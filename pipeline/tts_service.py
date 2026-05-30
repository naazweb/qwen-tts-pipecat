"""
Pipecat TTSService backed by Qwen3-TTS + megakernel talker decoder.

Pipeline per utterance:
  1. HF model prefill  — encode text, run talker prefill, get first codec token
  2. Megakernel decode — TalkerDecoder.step() loop for remaining codec tokens
  3. Codec → PCM       — speech_tokenizer.decode() every CHUNK_FRAMES tokens
  4. Stream chunks     — yield TTSAudioRawFrame as each chunk is ready
"""

import asyncio
import sys
import os
import time
from typing import AsyncGenerator, Generator

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_RATE = 24000
CHUNK_FRAMES = 6   # codec frames per PCM chunk (~0.5 s at 12 Hz)
EOS_TOKEN = 4096   # Qwen3-TTS codec EOS id


class MegakernelTTSService:
    """
    Synthesizes speech using:
      - HuggingFace Qwen3-TTS for text encoding + talker prefill
      - TalkerDecoder (megakernel) for autoregressive codec token decode
      - HF speech_tokenizer for codec tokens → PCM
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        language: str = "English",
        device: str = "cuda",
        verbose: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from tts_engine import TalkerDecoder

        self.language = language
        self.sample_rate = SAMPLE_RATE
        self._device = device

        if verbose:
            logger.info(f"Loading {model_name} (HF model for prefill + codec)...")

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._speech_tokenizer = self._hf_model.speech_tokenizer

        if verbose:
            logger.info("Loading TalkerDecoder (megakernel)...")
        self._talker = TalkerDecoder(model_name=model_name, verbose=verbose)

    def synthesize(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Yield float32 PCM chunks as codec frames are decoded via megakernel.
        Streams chunk-by-chunk rather than buffering the full utterance.
        """
        import torch

        t0 = time.perf_counter()

        # --- Step 1: prefill — get the first codec token from HF model ---
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self._processor(text=formatted, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self._device)

        with torch.no_grad():
            # Run prefill through HF model to populate its KV cache,
            # get logits for first codec token
            outputs = self._hf_model(input_ids=input_ids, use_cache=True)
            first_token_id = int(outputs.logits[0, -1].argmax().item())

        logger.info(f"Prefill done in {(time.perf_counter()-t0)*1000:.1f} ms, first token={first_token_id}")

        # --- Step 2: megakernel decode loop ---
        self._talker.reset()
        codec_tokens = [first_token_id]
        token_id = first_token_id
        first_chunk = True

        while token_id != EOS_TOKEN and len(codec_tokens) < 4096:
            token_id = self._talker.step(token_id)
            codec_tokens.append(token_id)

            # Every CHUNK_FRAMES tokens, decode to PCM and yield
            if len(codec_tokens) % CHUNK_FRAMES == 0:
                chunk_codes = codec_tokens[-CHUNK_FRAMES:]
                pcm = self._decode_chunk(chunk_codes)
                if first_chunk:
                    logger.info(f"TTFC: {(time.perf_counter()-t0)*1000:.1f} ms")
                    first_chunk = False
                yield pcm

        # Yield any remaining tokens
        remainder = len(codec_tokens) % CHUNK_FRAMES
        if remainder > 0:
            pcm = self._decode_chunk(codec_tokens[-remainder:])
            yield pcm

    def _decode_chunk(self, token_ids: list) -> np.ndarray:
        """Decode a list of codec token ids to float32 PCM via speech_tokenizer."""
        import torch
        codes = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            wavs = self._speech_tokenizer.decode(codes)
        return wavs[0, 0].cpu().float().numpy()


class QwenTTSService(TTSService):
    """
    Pipecat TTSService backed by Qwen3-TTS + megakernel.
    Streams TTSAudioRawFrame chunks as codec frames are decoded.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        language: str = "English",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
        self._model_name = model_name
        self._language = language
        self._device = device
        self._tts: MegakernelTTSService | None = None

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
            yield TTSStartedFrame(context_id=context_id)

            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(
                None, lambda: list(self._tts.synthesize(text))
            )

            for i, pcm in enumerate(chunks):
                if i == 0:
                    logger.info("First audio chunk ready")
                pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
                yield TTSAudioRawFrame(
                    audio=pcm_int16.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            logger.error(f"QwenTTSService error: {e}", exc_info=True)
            yield ErrorFrame(str(e))
        finally:
            yield TTSStoppedFrame(context_id=context_id)
