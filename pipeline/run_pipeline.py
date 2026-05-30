"""
End-to-end voice agent pipeline.

    Microphone → Whisper STT → LLM (text echo for demo) → MegakernelTTSService → Speaker

No API keys required. Runs fully local.

Usage:
    python pipeline/run_pipeline.py
"""

import logging
import queue
import sys
import threading

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from tts_service import MegakernelTTSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Audio config
MIC_SAMPLE_RATE = 16000   # Whisper expects 16 kHz
MIC_CHANNELS = 1
MIC_BLOCK_DURATION = 0.5  # seconds per mic block
MIC_BLOCK_FRAMES = int(MIC_SAMPLE_RATE * MIC_BLOCK_DURATION)

# VAD: stop recording after this many silent blocks
SILENCE_BLOCKS = 4
SILENCE_THRESHOLD = 0.01  # RMS threshold


def record_utterance() -> np.ndarray:
    """Block until the user speaks, then return the utterance as float32 PCM."""
    logger.info("Listening... (speak now)")
    audio_blocks = []
    silent_count = 0
    speaking = False

    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS, dtype="float32") as stream:
        while True:
            block, _ = stream.read(MIC_BLOCK_FRAMES)
            rms = float(np.sqrt(np.mean(block ** 2)))

            if rms > SILENCE_THRESHOLD:
                speaking = True
                silent_count = 0
                audio_blocks.append(block.copy())
            elif speaking:
                audio_blocks.append(block.copy())
                silent_count += 1
                if silent_count >= SILENCE_BLOCKS:
                    break

    return np.concatenate(audio_blocks, axis=0).squeeze()


def transcribe(whisper: WhisperModel, audio: np.ndarray) -> str:
    segments, _ = whisper.transcribe(audio, beam_size=1, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


def play_audio_stream(tts: MegakernelTTSService, text: str):
    """Stream TTS output to speaker chunk by chunk."""
    logger.info(f"Synthesizing: {text!r}")
    for pcm_chunk in tts.synthesize(text):
        sd.play(pcm_chunk, samplerate=tts.sample_rate, blocking=True)


def main():
    logger.info("Loading Whisper (base.en)...")
    whisper = WhisperModel("base.en", device="cuda", compute_type="float16")

    logger.info("Loading TTS service...")
    tts = MegakernelTTSService()

    logger.info("Pipeline ready. Press Ctrl+C to quit.\n")

    while True:
        try:
            audio = record_utterance()
            text = transcribe(whisper, audio)
            if not text:
                logger.info("(no speech detected, skipping)")
                continue
            logger.info(f"Transcribed: {text!r}")
            play_audio_stream(tts, text)
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()
