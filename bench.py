"""
Benchmark: megakernel talker decoder performance.

Reports:
  - tok/s  (codec tokens per second from the megakernel)
  - TTFC   (time to first audio chunk, target < 60ms)
  - RTF    (real-time factor, target < 0.15)
  - end-to-end latency (mic → audio out)

Usage:
    python bench.py

TODO:
  - Warm up TalkerDecoder
  - Run N decode steps, measure tok/s
  - Run full TTS on a fixed prompt, measure TTFC and RTF
  - Print results table
"""

# TODO: implement benchmarks
