"""JIT compilation of the megakernel for Qwen3-TTS talker decoder.

Identical to qwen_megakernel/build.py but compiled with -DLDG_VOCAB_SIZE=3072
to match the talker's codec vocabulary instead of the text vocabulary (151936).
"""

import os
from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_DIR, "../csrc")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


KERNEL_FLAGS = [
    f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
    f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
    f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 1280)}",
    f"-DLDG_LM_BLOCK_SIZE={_env_int('LDG_LM_BLOCK_SIZE', 384)}",
    f"-DLDG_LM_ROWS_PER_WARP={_env_int('LDG_LM_ROWS_PER_WARP', 2)}",
    f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
    f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
    f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
    f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
    f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
    f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
    f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
    f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
    "-DLDG_USE_UINT4",
    "-DLDG_ATTENTION_VEC4",
    "-DLDG_WEIGHT_LDCS",
    "-DLDG_MLP_SMEM",
    # Override vocab size for talker codec vocabulary
    "-DLDG_VOCAB_SIZE=3072",
]

CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-arch=sm_120a",
    f"-I{_CSRC}",
] + KERNEL_FLAGS


def get_extension():
    """Build (or return cached) the talker megakernel extension."""
    global _module
    if _module is not None:
        return _module

    _module = load(
        name="qwen_tts_talker_C",
        sources=[
            os.path.join(_CSRC, "torch_bindings.cpp"),
            os.path.join(_CSRC, "kernel.cu"),
        ],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=True,
    )
    return _module
