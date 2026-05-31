"""
Pipecat TTSService backed by Qwen3-TTS.
"""

import asyncio
import os
import sys
import time
import queue as _queue
from typing import AsyncGenerator

import numpy as np
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tts_engine import TalkerDecoder

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService, TextAggregationMode

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

SAMPLE_RATE = 24000
CHUNK_FRAMES = 6
MAX_NEW_TOKENS = 4096


class MegakernelTTSService:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        language: str = "English",
        device: str = "cuda",
        verbose: bool = True,
    ):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.language = language
        self.sample_rate = SAMPLE_RATE

        if verbose:
            logger.info(f"Loading {model_name}...")

        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.decoder = TalkerDecoder(self.model, verbose=verbose)

    def synthesize(self, text: str, pcm_queue: _queue.Queue) -> None:
        """Runs entirely in a background thread. Puts np.ndarray chunks into pcm_queue, then None sentinel."""
        import torch

        t0 = time.perf_counter()
        first = True
        hf_model = self.model.model  # Qwen3TTSForConditionalGeneration
        talker = hf_model.talker     # Qwen3TTSTalkerForConditionalGeneration
        cfg = hf_model.config
        talker_cfg = cfg.talker_config
        device = talker.device
        dtype = talker.dtype

        logger.debug(f"synthesize: starting for {text!r}")

        # --- Step 1: build prefill embeddings (mirrors Qwen3TTSForConditionalGeneration.generate) ---
        input_id = self._encode(text)  # [1, seq_len]

        language = self.language.lower()
        if language == "auto":
            language_id = None
        else:
            language_id = talker_cfg.codec_language_id[language]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                    device=device, dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)  # 3 × [1, 1, D]

        if language_id is None:
            codec_prefill_ids = [[talker_cfg.codec_nothink_id, talker_cfg.codec_think_bos_id, talker_cfg.codec_think_eos_id]]
        else:
            codec_prefill_ids = [[talker_cfg.codec_think_id, talker_cfg.codec_think_bos_id, language_id, talker_cfg.codec_think_eos_id]]

        codec_emb0 = talker.get_input_embeddings()(
            torch.tensor(codec_prefill_ids, device=device, dtype=input_id.dtype)
        )
        codec_emb1 = talker.get_input_embeddings()(
            torch.tensor([[talker_cfg.codec_pad_id, talker_cfg.codec_bos_id]], device=device, dtype=input_id.dtype)
        )
        codec_input_emb = torch.cat([codec_emb0, codec_emb1], dim=1)  # [1, L_codec, D]

        # role prefix: <|im_start|>assistant\n  (first 3 tokens)
        role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))

        # tts_pad * (L_codec-1) + tts_bos, summed with codec_input_emb[:-1]
        pad_part = tts_pad_embed.expand(-1, codec_input_emb.shape[1] - 2, -1)
        prefill_codec = torch.cat([pad_part, tts_bos_embed], dim=1) + codec_input_emb[:, :-1]
        talker_input_embed = torch.cat([role_embed, prefill_codec], dim=1)

        # first text token fused with last codec embed
        first_text_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_emb[:, -1:]
        talker_input_embed = torch.cat([talker_input_embed, first_text_embed], dim=1)

        # trailing text hiddens: tokens [4:-5] + eos
        trailing_text_hidden = torch.cat((
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])),
            tts_eos_embed,
        ), dim=1)  # [1, T_trail, D]

        # --- Step 2: HF prefill — one forward pass to get first codec token ---
        with torch.inference_mode():
            prefill_out = talker(
                inputs_embeds=talker_input_embed,
                attention_mask=torch.ones(1, talker_input_embed.shape[1], device=device, dtype=torch.long),
                use_cache=False,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=-1,
            )

        first_token_id = int(prefill_out.logits[0, -1].argmax())
        logger.info(f"TTFC (prefill): {(time.perf_counter() - t0) * 1000:.1f} ms, first_token={first_token_id}")

        # --- Step 3: megakernel prefill — replay all prefill tokens to populate its KV cache ---
        # The megakernel has its own KV cache separate from HF; we must warm it up
        # by stepping through the prefill embedding token-by-token.
        # We use the talker's codec_embedding to map each prefill position to a token id,
        # but the prefill is in embedding space (not token ids). Instead we run the
        # megakernel's step() using the first_token_id we got from HF, and prime the
        # KV cache by running one step per prefill position using a dummy token.
        # The correct approach: copy HF KV cache tensors into decoder's k/v cache.
        eos_id = talker_cfg.codec_eos_token_id
        self.decoder.reset()

        # Transfer HF DynamicCache → megakernel KV cache
        # HF cache shape per layer: [1, num_kv_heads, seq_len, head_dim]
        # Megakernel cache shape: [num_layers, num_kv_heads, max_seq_len, head_dim]
        from transformers import DynamicCache
        past_kv = DynamicCache()
        with torch.inference_mode():
            prefill_with_cache = talker(
                inputs_embeds=talker_input_embed,
                attention_mask=torch.ones(1, talker_input_embed.shape[1], device=device, dtype=torch.long),
                use_cache=True,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=-1,
            )
        past_kv = prefill_with_cache.past_key_values
        prefill_len = past_kv.get_seq_length()

        for layer_idx in range(len(past_kv)):
            k_hf, v_hf = past_kv[layer_idx]  # [1, num_kv_heads, prefill_len, head_dim]
            self.decoder._k_cache[layer_idx, :, :prefill_len, :] = k_hf[0].to(torch.bfloat16)
            self.decoder._v_cache[layer_idx, :, :prefill_len, :] = v_hf[0].to(torch.bfloat16)
        self.decoder._position = prefill_len

        codes = [first_token_id]
        token = first_token_id
        for _ in range(MAX_NEW_TOKENS - 1):
            token = self.decoder.step(token)
            if token == eos_id:
                break
            codes.append(token)

        codes_tensor = torch.tensor(codes, dtype=torch.long)
        logger.debug(f"synthesize: {len(codes)} codec tokens, decoding {len(codes) // CHUNK_FRAMES} chunks")

        # --- Step 4: stream PCM chunks ---
        chunk_count = 0
        for start in range(0, len(codes), CHUNK_FRAMES):
            chunk = codes_tensor[start: start + CHUNK_FRAMES]
            # speech_tokenizer expects audio_codes: [batch, T, num_codebooks]
            audio_codes = chunk.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            wavs, _ = hf_model.speech_tokenizer.decode([{"audio_codes": audio_codes}])
            pcm = wavs[0].astype(np.float32)
            if first:
                logger.info(f"TTFC (first audio): {(time.perf_counter() - t0) * 1000:.1f} ms")
                first = False
            chunk_count += 1
            logger.debug(f"synthesize: chunk #{chunk_count} ({len(pcm)} samples)")
            pcm_queue.put(pcm)

        logger.debug(f"synthesize: done, {chunk_count} chunks")
        pcm_queue.put(None)

    def _encode(self, text: str):
        import torch
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inp = self.model.processor(text=formatted, return_tensors="pt", padding=True)
        device = next(self.model.model.parameters()).device
        input_id = inp["input_ids"].to(device)
        return input_id if input_id.dim() == 2 else input_id.unsqueeze(0)


class QwenTTSService(TTSService):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        language: str = "English",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(sample_rate=SAMPLE_RATE, push_stop_frames=False, text_aggregation_mode=TextAggregationMode.SENTENCE, **kwargs)
        self._model_name = model_name
        self._language = language
        self._device = device
        self._tts: MegakernelTTSService | None = None
        self._settings.model = model_name
        self._settings.voice = None
        self._settings.language = language

    def _ensure_loaded(self):
        if self._tts is None:
            self._tts = MegakernelTTSService(
                model_name=self._model_name,
                language=self._language,
                device=self._device,
            )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"QwenTTSService synthesizing: {text!r}")
        self._ensure_loaded()
        try:
            await self.create_audio_context(context_id)
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            loop = asyncio.get_running_loop()
            pcm_queue: _queue.Queue = _queue.Queue()

            def _generate():
                self._tts.synthesize(text, pcm_queue)

            loop.run_in_executor(None, _generate)

            first = True
            frame_count = 0
            while True:
                pcm = await loop.run_in_executor(None, pcm_queue.get)
                if pcm is None:
                    logger.info(f"run_tts: done, pushed {frame_count} audio frames")
                    break
                if first:
                    logger.info("run_tts: first audio frame ready, streaming started")
                    first = False
                frame_count += 1
                pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
                yield TTSAudioRawFrame(
                    audio=pcm_int16.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )

            yield TTSStoppedFrame(context_id=context_id)
            await self.remove_audio_context(context_id)
        except Exception as e:
            logger.error(f"QwenTTSService error: {e}", exc_info=True)
            yield ErrorFrame(str(e))
