"""
Benchmark: Qwen3-TTS megakernel talker decoder performance.

Reports:
  - tok/s  (codec tokens per second)
  - TTFC   (time to first audio chunk, ms)
  - RTF    (real-time factor = synthesis_time / audio_duration)
  - Audio duration per prompt

Targets (from task.md):
  - TTFC < 60 ms
  - RTF  < 0.15

Usage:
    cd /workspace/qwen-tts-pipecat
    python bench.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, "pipeline")

WARMUP = 1
RUNS = 3

PROMPTS = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Qwen3 TTS is a multilingual text to speech model running on a single RTX 5090 GPU.",
]

SAMPLE_RATE = 24000


def bench_tts():
    from tts_service import MegakernelTTSService

    print("Loading model...")
    svc = MegakernelTTSService(verbose=True)

    print(f"\nWarmup ({WARMUP} run)...")
    for _ in range(WARMUP):
        list(svc.synthesize(PROMPTS[0]))

    print(f"\nBenchmarking ({RUNS} runs × {len(PROMPTS)} prompts)...\n")

    results = []

    for prompt in PROMPTS:
        ttfcs, rtfs, tok_rates = [], [], []

        for _ in range(RUNS):
            t0 = time.perf_counter()
            first_chunk = True
            ttfc = None
            chunks = []

            for pcm in svc.synthesize(prompt):
                if first_chunk:
                    ttfc = (time.perf_counter() - t0) * 1000
                    first_chunk = False
                chunks.append(pcm)

            total_time = time.perf_counter() - t0
            audio = np.concatenate(chunks)
            audio_duration = len(audio) / SAMPLE_RATE

            # codec tokens ≈ audio_duration * 12 Hz (12 frames/sec)
            codec_tokens = audio_duration * 12
            tok_s = codec_tokens / total_time
            rtf = total_time / audio_duration

            ttfcs.append(ttfc)
            rtfs.append(rtf)
            tok_rates.append(tok_s)

        results.append({
            "prompt": prompt[:50],
            "ttfc_ms": np.mean(ttfcs),
            "rtf": np.mean(rtfs),
            "tok_s": np.mean(tok_rates),
            "audio_s": audio_duration,
        })

    # Print results table
    print("=" * 75)
    print(f"{'Prompt':<52} {'TTFC(ms)':>8} {'RTF':>6} {'tok/s':>7}")
    print("-" * 75)
    for r in results:
        ttfc_flag = "✓" if r["ttfc_ms"] < 60 else "✗"
        rtf_flag  = "✓" if r["rtf"] < 0.15 else "✗"
        print(f"{r['prompt']:<52} {r['ttfc_ms']:>7.1f}{ttfc_flag} {r['rtf']:>5.3f}{rtf_flag} {r['tok_s']:>7.1f}")
    print("=" * 75)

    avg_ttfc = np.mean([r["ttfc_ms"] for r in results])
    avg_rtf  = np.mean([r["rtf"] for r in results])
    avg_toks = np.mean([r["tok_s"] for r in results])
    print(f"\n{'Average':<52} {avg_ttfc:>7.1f}  {avg_rtf:>5.3f}  {avg_toks:>7.1f}")
    print(f"\nTargets:  TTFC < 60 ms {'✓' if avg_ttfc < 60 else '✗'}   RTF < 0.15 {'✓' if avg_rtf < 0.15 else '✗'}")


if __name__ == "__main__":
    bench_tts()
