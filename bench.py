"""Benchmark: HF Qwen3-0.6B baseline vs TalkerDecoder megakernel."""

import gc
import os
import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

TOKENS = 200
WARMUP = 5
RUNS   = 10
PROMPT = "Hello, how are you today?"


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
                pad_token_id=tokenizer.eos_token_id,
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
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return TOKENS / avg, avg * 1000 / TOKENS


def bench_megakernel():
    from transformers import AutoTokenizer
    from tts_engine.model import TalkerDecoder, load_talker_weights, _pack_layer_weights
    from tts_engine.build import get_extension
    import math, struct

    # Load Qwen3-0.6B weights directly into TalkerDecoder-compatible format.
    # The talker and Qwen3-0.6B share the same architecture; we reuse
    # load_weights from qwen_megakernel but wire it through TalkerDecoder's
    # kernel (qwen_tts_talker_C) to prove the megakernel path works.
    from transformers import AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    hf_model.eval()

    # Build weight dict in the same shape TalkerDecoder expects,
    # but sourced from Qwen3-0.6B (identical architecture).
    from tts_engine.model import (
        NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        Q_SIZE, KV_SIZE, MAX_SEQ_LEN, _pack_layer_weights,
    )
    import math

    sd = hf_model.state_dict()

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend([
            sd[p + "input_layernorm.weight"].contiguous(),
            sd[p + "self_attn.q_proj.weight"].contiguous(),
            sd[p + "self_attn.k_proj.weight"].contiguous(),
            sd[p + "self_attn.v_proj.weight"].contiguous(),
            sd[p + "self_attn.q_norm.weight"].contiguous(),
            sd[p + "self_attn.k_norm.weight"].contiguous(),
            sd[p + "self_attn.o_proj.weight"].contiguous(),
            sd[p + "post_attention_layernorm.weight"].contiguous(),
            sd[p + "mlp.gate_proj.weight"].contiguous(),
            sd[p + "mlp.up_proj.weight"].contiguous(),
            sd[p + "mlp.down_proj.weight"].contiguous(),
        ])

    embed_weight      = sd["model.embed_tokens.weight"].contiguous()
    final_norm_weight = sd["model.norm.weight"].contiguous()
    lm_head_weight    = embed_weight  # tied

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # Build a TalkerDecoder-like object using the Qwen3-0.6B weights.
    # We call get_extension() to compile/load qwen_tts_talker_C.
    get_extension()
    _decode        = torch.ops.qwen_tts_talker_C.decode
    _gen_nosync    = torch.ops.qwen_tts_talker_C.generate_nosync
    layer_packed   = _pack_layer_weights(layer_weights)
    attn_scale     = 1.0 / math.sqrt(HEAD_DIM)

    k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros_like(k_cache)

    f32 = dict(dtype=torch.float32, device="cuda")
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    hidden    = torch.empty(HIDDEN_SIZE, **bf16)
    act       = torch.empty(HIDDEN_SIZE, **f32)
    res       = torch.empty(HIDDEN_SIZE, **f32)
    q         = torch.empty(Q_SIZE, **f32)
    k         = torch.empty(KV_SIZE, **f32)
    v         = torch.empty(KV_SIZE, **f32)
    attn_out  = torch.empty(Q_SIZE, **f32)
    mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
    norm_out  = torch.empty(HIDDEN_SIZE, **f32)
    bmax_vals = torch.empty(4096, **f32)
    bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
    out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_tokens = input_ids[0].tolist()

    def run():
        k_cache.zero_()
        v_cache.zero_()
        pos = 0
        # prefill: step through prompt tokens
        for tid in prompt_tokens[:-1]:
            _decode(out_token, tid, embed_weight, layer_packed,
                    final_norm_weight, lm_head_weight,
                    cos_table, sin_table, k_cache, v_cache,
                    hidden, act, res, q, k, v, attn_out, mlp_inter, norm_out,
                    bmax_vals, bmax_idxs,
                    NUM_LAYERS, pos, MAX_SEQ_LEN, attn_scale)
            pos += 1
        # generate TOKENS tokens via no-sync kernel
        _gen_nosync(prompt_tokens[-1], TOKENS,
                    embed_weight, layer_packed,
                    final_norm_weight, lm_head_weight,
                    cos_table, sin_table, k_cache, v_cache,
                    hidden, act, res, q, k, v, attn_out, mlp_inter, norm_out,
                    bmax_vals, bmax_idxs,
                    NUM_LAYERS, pos, MAX_SEQ_LEN, attn_scale)

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
    return TOKENS / avg, avg * 1000 / TOKENS


if __name__ == "__main__":
    print("PyTorch (HF)...")
    hf_tok, hf_ms = bench_pytorch_hf()

    print("Megakernel (qwen_tts_talker_C)...")
    mk_tok, mk_ms = bench_megakernel()

    speedup = mk_tok / hf_tok

    print()
    print("=" * 55)
    print(f"{'Backend':<28} {'tok/s':>7} {'ms/tok':>8} {'Speedup':>8}")
    print("-" * 55)
    print(f"{'PyTorch (HF)':<28} {hf_tok:>7.1f} {hf_ms:>8.2f} {'1.00x':>8}")
    print(f"{'Megakernel':<28} {mk_tok:>7.1f} {mk_ms:>8.2f} {speedup:>7.2f}x")
    print("=" * 55)
