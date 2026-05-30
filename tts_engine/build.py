"""
Build the CUDA extension for the TTS megakernel.

Key difference from the original qwen_megakernel build:
  -DLDG_VOCAB_SIZE=3072  (talker codec vocab, not 151936)

Usage:
    python build.py
"""

# TODO: implement build using torch.utils.cpp_extension.load()
# Pass extra_cuda_cflags=["-DLDG_VOCAB_SIZE=3072", "-arch=sm_120", ...]
