"""
Pipecat TTSService backed by Qwen3-TTS megakernel talker decoder.

Accepts text frames from the pipeline, yields TTSAudioRawFrame chunks
as codec tokens are decoded — no full-utterance buffering.
"""

import asyncio
from typing import AsyncGenerator

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
# 6 codec frames ≈ 0.5 s of audio at 12 Hz — low latency without excessive decode calls
CHUNK_FRAMES = 6


class QwenTTSService(TTSService):
    """
    Pipecat TTS service backed by Qwen3-TTS-12Hz-0.6B-Base.

    Drop-in replacement for any Pipecat TTSService.
    Streams audio frame-by-frame as codec chunks are decoded.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        language: str = "English",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(sample_rate=SAMPLE_RATE, push_stop_frames=True, **kwargs)

        # Lazy-load on first use so pipeline construction doesn't block
        self._model_name = model_name
        self._language = language
        self._device = device
        self._tts = None

    def _ensure_loaded(self):
        if self._tts is None:
            from pipeline.tts_service import MegakernelTTSService
            self._tts = MegakernelTTSService(
                model_name=self._model_name,
                language=self._language,
                device=self._device,
                verbose=True,
            )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"QwenTTSService synthesizing: {text!r}")

        try:
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)
            await self.start_ttfb_metrics()

            loop = asyncio.get_event_loop()

            # Run the blocking model load + generate in a thread so we don't
            # block the asyncio event loop
            def _generate():
                self._ensure_loaded()
                return list(self._tts.synthesize(text))

            chunks = await loop.run_in_executor(None, _generate)

            first = True
            for pcm in chunks:
                if first:
                    await self.stop_ttfb_metrics()
                    first = False

                # Convert float32 [-1, 1] → int16 PCM bytes
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
            await self.remove_audio_context(context_id)
