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

import torch

# Talker-specific constants (architecture identical to Qwen3-0.6B)
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM # 1024
MAX_SEQ_LEN = 32768
VOCAB_SIZE = 3072
ROPE_THETA = 1_000_000


def load_talker_weights(model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
    """
    Load Qwen3-TTS talker decoder weights into GPU tensors.

    Returns weights dict compatible with the megakernel Decoder,
    plus the full model's tokenizer.

    TODO:
      1. Load Qwen3TTSForConditionalGeneration from HuggingFace
      2. Extract state_dict keys under the talker sub-model prefix
      3. Build RoPE tables with ROPE_THETA=1_000_000, MAX_SEQ_LEN=32768
      4. Pack layer_weights list (same 11-tensor-per-layer format)
      5. Return weights dict + tokenizer
    """
    raise NotImplementedError


class TalkerDecoder:
    """
    Stateful megakernel decoder wrapping the Qwen3-TTS talker.

    Drop-in replacement for qwen_megakernel.model.Decoder,
    but loaded with talker weights and codec vocab.

    TODO: implement __init__ and step() mirroring Decoder in model.py
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
        raise NotImplementedError

    def step(self, token_id: int) -> int:
        """Decode one codec token. Returns next codec token id."""
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
