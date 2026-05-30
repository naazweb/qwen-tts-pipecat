"""
Streaming TTS service backed by the Qwen3-TTS megakernel talker decoder.

Accepts text, yields PCM audio chunks as they're decoded (not buffered).

Usage (standalone test):
    service = MegakernelTTSService()
    for audio_chunk in service.synthesize("Hello world"):
        play(audio_chunk)

TODO:
  - Load TalkerDecoder on init
  - synthesize(text): encode text → run talker step-by-step → decode codec tokens → yield PCM chunks
  - Measure and log TTFC on first chunk
"""

# TODO: implement MegakernelTTSService
