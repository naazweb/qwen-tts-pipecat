"""
Weight loading and decode API for Qwen3-TTS talker decoder.

The talker has identical architecture to Qwen3-0.6B:
  hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
  num_key_value_heads=8, head_dim=128, intermediate_size=3072

Differences from the base megakernel model.py:
  - Weights are loaded from Qwen/Qwen3-TTS (talker sub-model)
  - vocab_size=3072 (codec tokens), lm_head is NOT tied to embeddings
  - rope_theta=1000000 (vs 10000 in original)
  - max_position_embeddings=32768 (vs 2048 in original)
"""

import math
import struct

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
Q_SIZE = NUM_Q_HEADS * HEAD_DIM    # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
MAX_SEQ_LEN = 32768
VOCAB_SIZE = 3072
ROPE_THETA = 1_000_000

_decode = None
_generate_nosync = None


def _get_ops():
    global _decode, _generate_nosync
    if _decode is None:
        from .build import get_extension
        get_extension()
        _decode = torch.ops.qwen_tts_C.decode
        _generate_nosync = torch.ops.qwen_tts_C.generate_nosync
    return _decode, _generate_nosync


def load_talker_weights(model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
    """Load Qwen3-TTS talker decoder weights into GPU tensors."""
    if not verbose:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging
            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    if verbose:
        print(f"Loading {model_name} talker weights...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    state = model.state_dict()

    # Detect talker prefix — Qwen3-TTS exposes the talker under
    # "talker_lm_model." or directly at root depending on model class.
    sample_key = next(k for k in state if "embed_tokens" in k)
    prefix = sample_key[: sample_key.index("embed_tokens")]  # e.g. "talker_lm_model.model."

    def w(key):
        full = prefix + key
        return state[full].contiguous()

    # RoPE tables with talker theta and max_seq_len
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    # Per-layer weights (11 tensors per layer, same order as LDGLayerWeights)
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"layers.{i}."
        layer_weights.extend([
            w(p + "input_layernorm.weight"),
            w(p + "self_attn.q_proj.weight"),
            w(p + "self_attn.k_proj.weight"),
            w(p + "self_attn.v_proj.weight"),
            w(p + "self_attn.q_norm.weight"),
            w(p + "self_attn.k_norm.weight"),
            w(p + "self_attn.o_proj.weight"),
            w(p + "post_attention_layernorm.weight"),
            w(p + "mlp.gate_proj.weight"),
            w(p + "mlp.up_proj.weight"),
            w(p + "mlp.down_proj.weight"),
        ])

    # lm_head is NOT tied to embeddings in the talker
    lm_head_key = next(k for k in state if "lm_head" in k and "talker" in k)
    lm_head_prefix = lm_head_key[: lm_head_key.index("lm_head")]

    weights = dict(
        embed_weight=w("embed_tokens.weight"),
        layer_weights=layer_weights,
        final_norm_weight=w("norm.weight"),
        lm_head_weight=state[lm_head_prefix + "lm_head.weight"].contiguous(),
        cos_table=cos_table,
        sin_table=sin_table,
    )

    del model
    torch.cuda.empty_cache()
    return weights


def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    ptr_size = 8
    n_ptrs = 11
    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class TalkerDecoder:
    """Stateful megakernel decoder for the Qwen3-TTS talker."""

    def __init__(self, weights: dict | None = None, model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
        if weights is None:
            weights = load_talker_weights(model_name, verbose=verbose)

        self._weights = weights
        self._position = 0

        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def step(self, token_id: int) -> int:
        """Decode one codec token. Returns next codec token id."""
        decode, _ = _get_ops()
        decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v,
            self._attn_out, self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += 1
        return int(self._out_token.item())

    def generate_tokens(self, first_token_id: int, num_steps: int) -> list[int]:
        """Decode num_steps tokens with no CPU sync between steps."""
        _, generate_nosync = _get_ops()
        output = generate_nosync(
            first_token_id, num_steps,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v,
            self._attn_out, self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += num_steps
        return output.cpu().tolist()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position
