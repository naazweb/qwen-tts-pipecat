"""Benchmark: Qwen3-TTS pipeline text token decode vs PyTorch HuggingFace baseline."""

import gc
import time
import warnings

import torch

warnings.filterwarnings("ignore")

TOKENS = 100
WARMUP = 3
RUNS   = 5
PROMPT = "Hello"


def bench_pytorch_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()

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


def bench_tts_pipeline():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()

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


if __name__ == "__main__":
    print("PyTorch (HF)...")
    hf_tok, hf_ms = bench_pytorch_hf()

    print("Qwen3-TTS pipeline (sdpa)...")
    tts_tok, tts_ms = bench_tts_pipeline()

    speedup = tts_tok / hf_tok

    print()
    print("=" * 55)
    print(f"{'Backend':<25} {'tok/s':>8} {'ms/tok':>8} {'Speedup':>8}")
    print("-" * 55)
    print(f"{'PyTorch (HF)':<25} {hf_tok:>8.1f} {hf_ms:>8.2f} {'1.00x':>8}")
    print(f"{'Qwen3-TTS pipeline':<25} {tts_tok:>8.1f} {tts_ms:>8.2f} {speedup:>7.2f}x")
    print("=" * 55)
