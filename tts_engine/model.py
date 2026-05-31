"""TalkerDecoder: wraps qwen_megakernel.Decoder with talker weights.

Uses the original megakernel from qwen_megakernel (no copy, no recompile).
The talker backbone is identical to Qwen3-0.6B so the kernel works as-is.
The lm_head_weight passed is talker.codec_head [3072, 1024] — smaller than
the text vocab but the kernel's bmax scratch buffer (4096) covers it fine.
"""

import math
import struct
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../qwen_megakernel"))
from qwen_megakernel.model import Decoder, _pack_layer_weights
from qwen_megakernel.build import get_extension  # ensures qwen_megakernel_C is compiled

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM
KV_SIZE = 8 * HEAD_DIM
MAX_SEQ_LEN = 2048


def load_talker_weights(tts_model, verbose: bool = True):
    """Extract talker transformer weights from a loaded Qwen3TTSModel."""
    if verbose:
        print("Extracting talker weights for megakernel...")

    sd = tts_model.model.state_dict()

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
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


class TalkerDecoder(Decoder):
    """Megakernel decoder loaded with talker weights instead of Qwen3-0.6B text weights.

    Inherits Decoder from qwen_megakernel — same kernel (qwen_megakernel_C),
    same step()/reset() API. Only the weights differ.
    """

    def __init__(self, tts_model, verbose: bool = True):
        get_extension()  # ensure qwen_megakernel_C is compiled
        weights = load_talker_weights(tts_model, verbose=verbose)
        super().__init__(weights=weights)
        if verbose:
            print("TalkerDecoder ready.")
