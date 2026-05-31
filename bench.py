"""Benchmark: Qwen3-TTS pipeline vs PyTorch HuggingFace baseline.

Mirrors the format of qwen_megakernel/qwen_megakernel/bench.py so results
are directly comparable.

Reports per backend:
  - tok/s   (codec tokens per second from the talker decoder)
  - ms/tok  (milliseconds per codec token)
  - TTFC    (time to first audio chunk, ms)  — TTS only
  - RTF     (real-time factor)               — TTS only

Usage:
    cd /workspace/qwen-tts-pipecat
    python bench.py
"""

import gc
import queue
import sys
import threading
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, "pipeline")

TOKENS   = 100   # target codec tokens for decoder-only benchmark
WARMUP   = 3
RUNS     = 5
SAMPLE_RATE = 24000

PROMPTS = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Qwen3 TTS is a multilingual text to speech model running on a single RTX 5090 GPU.",
]


# ---------------------------------------------------------------------------
# HuggingFace PyTorch baseline (talker decoder only, no audio decode)
# ---------------------------------------------------------------------------

def bench_pytorch_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    input_ids = tokenizer("Hello", return_tensors="pt").input_ids.cuda()

    def run():
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return TOKENS / avg, avg * 1000 / TOKENS


# ---------------------------------------------------------------------------
# TTS pipeline benchmark
# ---------------------------------------------------------------------------

def synthesize_timed(svc, prompt):
    """Run synthesize in a thread, return (ttfc_ms, total_ms, chunks)."""
    pcm_queue = queue.Queue()
    t0 = time.perf_counter()

    t = threading.Thread(target=svc.synthesize, args=(prompt, pcm_queue), daemon=True)
    t.start()

    ttfc = None
    chunks = []
    while True:
        pcm = pcm_queue.get()
        if pcm is None:
            break
        if ttfc is None:
            ttfc = (time.perf_counter() - t0) * 1000
        chunks.append(pcm)

    total_ms = (time.perf_counter() - t0) * 1000
    t.join()
    return ttfc, total_ms, chunks


def bench_tts():
    from tts_service import MegakernelTTSService

    print("Loading TTS model...")
    svc = MegakernelTTSService(verbose=True)

    print(f"\nWarmup ({WARMUP} runs)...")
    for _ in range(WARMUP):
        synthesize_timed(svc, PROMPTS[0])

    print(f"\nBenchmarking ({RUNS} runs × {len(PROMPTS)} prompts)...\n")

    all_tok_s, all_ms_tok, all_ttfc, all_rtf = [], [], [], []
    prompt_results = []

    for prompt in PROMPTS:
        tok_rates, ms_toks, ttfcs, rtfs = [], [], [], []

        for _ in range(RUNS):
            ttfc, total_ms, chunks = synthesize_timed(svc, prompt)
            audio = np.concatenate(chunks)
            audio_duration = len(audio) / SAMPLE_RATE
            total_s = total_ms / 1000

            codec_tokens = audio_duration * 12  # 12 Hz tokenizer
            tok_s = codec_tokens / total_s
            ms_tok = total_s * 1000 / codec_tokens
            rtf = total_s / audio_duration

            tok_rates.append(tok_s)
            ms_toks.append(ms_tok)
            ttfcs.append(ttfc)
            rtfs.append(rtf)

        prompt_results.append({
            "prompt":   prompt[:48],
            "tok_s":    np.mean(tok_rates),
            "ms_tok":   np.mean(ms_toks),
            "ttfc_ms":  np.mean(ttfcs),
            "rtf":      np.mean(rtfs),
        })
        all_tok_s.extend(tok_rates)
        all_ms_tok.extend(ms_toks)
        all_ttfc.extend(ttfcs)
        all_rtf.extend(rtfs)

    return prompt_results, {
        "tok_s":   np.mean(all_tok_s),
        "ms_tok":  np.mean(all_ms_tok),
        "ttfc_ms": np.mean(all_ttfc),
        "rtf":     np.mean(all_rtf),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Qwen3-TTS Pipeline Benchmark")
    print("=" * 65)
    print()

    print("── HuggingFace PyTorch baseline (Qwen3-0.6B decoder, no audio) ──")
    hf_tok, hf_ms = bench_pytorch_hf()
    print()

    print("── Qwen3-TTS pipeline (talker + audio decode) ──")
    prompt_results, avg = bench_tts()

    # Per-prompt table
    print()
    print("=" * 65)
    print(f"{'Prompt':<50} {'tok/s':>6} {'ms/tok':>7}")
    print("-" * 65)
    for r in prompt_results:
        print(f"{r['prompt']:<50} {r['tok_s']:>6.1f} {r['ms_tok']:>7.2f}")
    print("=" * 65)

    # Summary comparison table (mirrors megakernel bench format)
    print()
    print("=" * 55)
    print(f"{'Backend':<25} {'tok/s':>8} {'ms/tok':>8}")
    print("-" * 55)
    print(f"{'HF PyTorch baseline':<25} {hf_tok:>8.1f} {hf_ms:>8.2f}")
    print(f"{'Qwen3-TTS pipeline':<25} {avg['tok_s']:>8.1f} {avg['ms_tok']:>8.2f}")
    print("=" * 55)

    # TTS-specific metrics
    print()
    print("=" * 55)
    print(f"{'TTS Metric':<25} {'Value':>10} {'Target':>10} {'Pass':>6}")
    print("-" * 55)
    ttfc_pass = avg['ttfc_ms'] < 90
    rtf_pass  = avg['rtf'] < 0.3
    print(f"{'TTFC (ms)':<25} {avg['ttfc_ms']:>10.1f} {'< 90':>10} {'✓' if ttfc_pass else '✗':>6}")
    print(f"{'RTF':<25} {avg['rtf']:>10.3f} {'< 0.3':>10} {'✓' if rtf_pass else '✗':>6}")
    print("=" * 55)
