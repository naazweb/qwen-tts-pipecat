"""
Streaming TTS service backed by Qwen3-TTS.

Accepts text, yields PCM audio chunks as they're decoded (not buffered).

Usage (standalone test):
    service = MegakernelTTSService()
    for audio_chunk in service.synthesize("Hello world"):
        play(audio_chunk)
"""

import time
import logging
from typing import Generator

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
# Chunk size in codec frames. At 12 Hz, 12 frames = 1 second of audio.
# 6 frames = ~0.5 s — small enough for low latency, large enough to amortize decode overhead.
CHUNK_FRAMES = 6


class MegakernelTTSService:
    """
    Streaming TTS service wrapping Qwen3-TTS-12Hz-0.6B-Base.

    synthesize() yields raw PCM float32 numpy arrays at 24 kHz as soon as
    each chunk of codec frames is decoded — no full-utterance buffering.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        language: str = "English",
        device: str = "cuda",
        verbose: bool = True,
    ):
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
        """
        Yield PCM float32 chunks (numpy arrays, 24 kHz) as they are decoded.

        Measures and logs TTFC (time to first chunk).
        """
        t0 = time.perf_counter()
        first_chunk = True

        # Generate all codec codes at once (talker + code predictor run inside here).
        # We then stream the *decode* step chunk-by-chunk so audio starts playing
        # before the full waveform is ready.
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

        codes = talker_codes[0]  # (T, num_code_groups)
        total_frames = codes.shape[0]

        for start in range(0, total_frames, CHUNK_FRAMES):
            chunk_codes = codes[start : start + CHUNK_FRAMES]  # (C, G)

            wavs, fs = self.model.model.speech_tokenizer.decode(
                [{"audio_codes": chunk_codes}]
            )
            pcm = wavs[0].astype(np.float32)

            if first_chunk:
                ttfc_ms = (time.perf_counter() - t0) * 1000
                logger.info(f"TTFC: {ttfc_ms:.1f} ms")
                first_chunk = False

            yield pcm

    def _encode(self, text: str) -> torch.Tensor:
        """Tokenize text into talker input_ids."""
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inp = self.model.processor(text=formatted, return_tensors="pt", padding=True)
        input_id = inp["input_ids"].to(next(self.model.model.parameters()).device)
        return input_id if input_id.dim() == 2 else input_id.unsqueeze(0)
