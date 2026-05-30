"""
Pipecat TTSService backed by Qwen3-TTS.

Also exposes MegakernelTTSService for standalone use:
    svc = MegakernelTTSService()
    for pcm_chunk in svc.synthesize("Hello world"):
        play(pcm_chunk)
"""

import asyncio
import time
import logging
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

SAMPLE_RATE = 24000
CHUNK_FRAMES = 6  # 6 codec frames ≈ 0.5 s at 12 Hz


class MegakernelTTSService:
    """
    Standalone TTS service wrapping Qwen3-TTS-12Hz-0.6B-Base.
    Yields float32 PCM numpy arrays at 24 kHz, chunk by chunk.
    """

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
        """Yield float32 PCM chunks as codec frames are decoded."""
        t0 = time.perf_counter()
        first = True

        talker_codes, _ = self.model.model.generate(
            input_ids=[self._encode(text)],
            languages=[self.language],
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

        codes = talker_codes[0]
        for start in range(0, codes.shape[0], CHUNK_FRAMES):
            chunk = codes[start : start + CHUNK_FRAMES]
            wavs, _ = self.model.model.speech_tokenizer.decode([{"audio_codes": chunk}])
            pcm = wavs[0].astype(np.float32)

            if first:
                logger.info(f"TTFC: {(time.perf_counter() - t0) * 1000:.1f} ms")
                first = False

            yield pcm

    def _encode(self, text: str):
        import torch
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inp = self.model.processor(text=formatted, return_tensors="pt", padding=True)
        input_id = inp["input_ids"].to(next(self.model.model.parameters()).device)
        return input_id if input_id.dim() == 2 else input_id.unsqueeze(0)


class QwenTTSService(TTSService):
    """
    Pipecat TTSService backed by Qwen3-TTS.
    Streams TTSAudioRawFrame chunks as codec frames are decoded.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
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
            logger.error(f"QwenTTSService error: {e}")
            yield ErrorFrame(str(e))
        finally:
            yield TTSStoppedFrame(context_id=context_id)
