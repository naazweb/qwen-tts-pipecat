# qwen-tts-pipecat

Qwen3-TTS voice agent pipeline using AlpinDale's RTX 5090 megakernel as the talker decoder backend.

## Architecture

```
Microphone → Deepgram STT → LLM → MegakernelTTSService → Speaker
                                         │
                              Qwen3-TTS talker decoder
                              (megakernel, ~1000 tok/s)
                                         │
                              Qwen3-TTS vocoder → PCM audio
```

### Kernel Adaptation

The megakernel targets Qwen3-0.6B. The Qwen3-TTS talker decoder has **identical architecture**:
- `hidden_size=1024`, `num_hidden_layers=28`, `num_attention_heads=16/8`, `intermediate_size=3072`

One change required: compile with `-DLDG_VOCAB_SIZE=3072` (talker codec vocab vs 151936 text vocab).

Other differences handled in Python weight loading:
- `rope_theta=1_000_000` (vs 10000)
- `max_position_embeddings=32768` (vs 2048)
- Separate `lm_head` weight (not tied to embeddings)

## Build

```bash
# Requires CUDA 12.8+, RTX 5090 (sm_120)
pip install -r requirements.txt
python tts_engine/build.py
```

## Run

```bash
# No API keys needed — fully local
python pipeline/run_pipeline.py
```

## Benchmark

```bash
python bench.py
```

## Performance

<!-- TODO: fill in after running on RTX 5090 -->

| Metric | Target | Measured |
|--------|--------|----------|
| tok/s  | ~1000  | TBD      |
| TTFC   | < 60ms | TBD      |
| RTF    | < 0.15 | TBD      |
| E2E latency | — | TBD  |
