"""TalkerDecoder: megakernel-backed decode for Qwen3-TTS talker.

Loads the talker's transformer weights (identical architecture to Qwen3-0.6B)
into the megakernel and exposes step(token_id) -> next_token_id, exactly like
qwen_megakernel.model.Decoder but using the talker's codec vocabulary (3072)
instead of the text vocabulary (151936).
"""

import math
import struct

import torch

from .build import get_extension

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM   # 2048
KV_SIZE = 8 * HEAD_DIM   # 1024
MAX_SEQ_LEN = 2048
TALKER_VOCAB_SIZE = 3072  # codec vocab, not text vocab


def load_talker_weights(tts_model, verbose: bool = True):
    """Extract talker transformer weights from a loaded Qwen3TTSModel."""
    if verbose:
        print("Extracting talker weights for megakernel...")

    sd = tts_model.model.state_dict()
    p = "talker.model.layers.{i}."

    # RoPE tables — same formula as Qwen3-0.6B
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    layer_weights = []
    for i in range(NUM_LAYERS):
        prefix = f"talker.model.layers.{i}."
        layer_weights.extend([
            sd[prefix + "input_layernorm.weight"].contiguous(),
            sd[prefix + "self_attn.q_proj.weight"].contiguous(),
            sd[prefix + "self_attn.k_proj.weight"].contiguous(),
            sd[prefix + "self_attn.v_proj.weight"].contiguous(),
            sd[prefix + "self_attn.q_norm.weight"].contiguous(),
            sd[prefix + "self_attn.k_norm.weight"].contiguous(),
            sd[prefix + "self_attn.o_proj.weight"].contiguous(),
            sd[prefix + "post_attention_layernorm.weight"].contiguous(),
            sd[prefix + "mlp.gate_proj.weight"].contiguous(),
            sd[prefix + "mlp.up_proj.weight"].contiguous(),
            sd[prefix + "mlp.down_proj.weight"].contiguous(),
        ])

    return dict(
        embed_weight=sd["talker.model.codec_embedding.weight"].contiguous(),
        layer_weights=layer_weights,
        final_norm_weight=sd["talker.model.norm.weight"].contiguous(),
        lm_head_weight=sd["talker.codec_head.weight"].contiguous(),
        cos_table=cos_table,
        sin_table=sin_table,
    )


def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class TalkerDecoder:
    """Megakernel-backed talker decoder. API mirrors qwen_megakernel.Decoder."""

    def __init__(self, tts_model, verbose: bool = True):
        get_extension()  # compile / load the kernel
        self._decode = torch.ops.qwen_tts_talker_C.decode
        self._generate_nosync = torch.ops.qwen_tts_talker_C.generate_nosync

        weights = load_talker_weights(tts_model, verbose=verbose)
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position = 0

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
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

        if verbose:
            print("TalkerDecoder ready.")

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def step(self, token_id: int) -> int:
        """Decode one codec token. Returns the next codec token id."""
        self._decode(
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
        return self._out_token.item()

    def generate_tokens(self, first_token_id: int, num_steps: int) -> list[int]:
        """Generate num_steps codec tokens using the no-sync kernel."""
        output_ids = self._generate_nosync(
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
        return output_ids.cpu().tolist()
