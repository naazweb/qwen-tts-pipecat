/**
 * Adapted megakernel for Qwen3-TTS talker decoder.
 *
 * Architecture is identical to Qwen3-0.6B (hidden=1024, layers=28, heads=16/8)
 * so the kernel body is unchanged. Only LDG_VOCAB_SIZE is overridden to 3072
 * (talker codec vocab) via compile flag: -DLDG_VOCAB_SIZE=3072
 *
 * Original: github.com/AlpinDale/qwen_megakernel
 */

// TODO: copy kernel.cu from qwen_megakernel/csrc/kernel.cu verbatim,
//       then verify -DLDG_VOCAB_SIZE=3072 is passed at compile time.
