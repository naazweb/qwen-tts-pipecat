"""
End-to-end voice agent pipeline.

    Microphone → Whisper STT → LLM (text echo for demo) → MegakernelTTSService → Speaker

No API keys required. Runs fully local.

Usage:
    python pipeline/run_pipeline.py

TODO:
  - Capture mic audio with sounddevice
  - Transcribe with faster-whisper
  - Pass text to MegakernelTTSService
  - Play back audio output
"""

# TODO: implement pipeline
