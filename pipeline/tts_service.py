"""
Pipecat TTSService backed by Qwen3-TTS.
"""

import asyncio
import os
import sys
import time
import threading
import queue as _queue
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

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

SAMPLE_RATE = 24000
CHUNK_FRAMES = 6


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
        )

    def synthesize(self, text: str) -> Generator[np.ndarray, None, None]:
        import torch
        from transformers import BaseStreamer

        t0 = time.perf_counter()
        first = True
        token_queue: _queue.Queue = _queue.Queue()

        class _CodeStreamer(BaseStreamer):
            def put(self, value):
                toks = value.squeeze(0) if value.dim() > 1 else value
                for tok in toks.tolist():
                    token_queue.put(tok)

            def end(self):
                token_queue.put(None)

        streamer = _CodeStreamer()

        def _run_generate():
            self.model.model.generate(
                input_ids=[self._encode(text)],
                languages=[self.language],
                streamer=streamer,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                subtalker_dosample=True,
                subtalker_temperature=0.9,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                max_new_tokens=4096,
            )

        t = threading.Thread(target=_run_generate, daemon=True)
        t.start()

        buf = []
        while True:
            tok = token_queue.get()
            if tok is None:
                break
            buf.append(tok)
            if len(buf) >= CHUNK_FRAMES:
                chunk = torch.tensor(buf, dtype=torch.long).unsqueeze(0)
                wavs, _ = self.model.model.speech_tokenizer.decode([{"audio_codes": chunk}])
                pcm = wavs[0].astype(np.float32)
                if first:
                    logger.info(f"TTFC: {(time.perf_counter() - t0) * 1000:.1f} ms")
                    first = False
                buf = []
                yield pcm

        if buf:
            chunk = torch.tensor(buf, dtype=torch.long).unsqueeze(0)
            wavs, _ = self.model.model.speech_tokenizer.decode([{"audio_codes": chunk}])
            yield wavs[0].astype(np.float32)

        t.join()

    def _encode(self, text: str):
        import torch
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inp = self.model.processor(text=formatted, return_tensors="pt", padding=True)
        input_id = inp["input_ids"].to(next(self.model.model.parameters()).device)
        return input_id if input_id.dim() == 2 else input_id.unsqueeze(0)


class QwenTTSService(TTSService):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        language: str = "English",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(sample_rate=SAMPLE_RATE, push_stop_frames=True, stop_frame_timeout_s=30.0, **kwargs)
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
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _generate():
                for pcm in self._tts.synthesize(text):
                    queue.put_nowait(pcm)
                queue.put_nowait(None)

            loop.run_in_executor(None, _generate)

            first = True
            while True:
                pcm = await queue.get()
                if pcm is None:
                    break
                if first:
                    logger.info("First audio chunk ready")
                    first = False
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
