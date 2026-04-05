# qwen-inference-lab

Exploration log: optimizing Qwen3.5-27B inference on Apple Silicon (M4 Max, 128 GB, 546 GB/s memory bandwidth).

## Goal

Push Qwen3.5-27B token generation as fast as possible on a single M4 Max, starting from stock `mlx_lm` and ending wherever the hardware limits take us.

## Results

| Configuration | tok/s | vs. baseline |
|---|---|---|
| Stock mlx_lm | 29.5 | 1.00x |
| V5 monolithic compile | 30.0 | 1.02x |
| Stock + spec decode (0.8B draft) | 37.6 | 1.27x |
| MTP head (self-speculative) | 36.9 | 1.25x |
| MTP + split-recurrence rollback | **42.7** | **1.45x** |

Starting point: **29.5 tok/s** (stock mlx_lm, async eval).
Ending point: **42.7 tok/s** (MTP with zero-cost DeltaNet rollback).

## What's in here

- `docs/TIMELINE.md` -- Every approach tried, in order, with results and reasoning
- `docs/BANDWIDTH_ANALYSIS.md` -- Profiling work: where the time actually goes
- `kernels/fused_gdn.py` -- The fused kernel code (V2-V7, MTP head, split-recurrence)
- `benchmarks/bench_v7.py` -- Speculative decoding benchmark harness
- `logs/llama_cpp_server.log` -- Server output showing MTP+draft performance

## The journey, briefly

**Phase 1: Kernel fusion (V2-V6).** Spent weeks fusing DeltaNet projections, writing custom Metal kernels, compiling entire layer stacks with `mx.compile`. Net gain: +1.7%. The GPU was already the bottleneck, and `mx.compile` was already fusing elementwise ops behind the scenes. Custom kernels actually *broke* its fusion graph and made things slower (V6: -1.5%).

**Phase 2: Speculative decoding.** Almost gave up on this after a broken benchmark showed 26.5 tok/s. Re-tested properly and got 37.6 tok/s. The lesson: measure correctly before declaring something dead.

**Phase 3: MTP (Multi-Token Prediction).** Discovered Qwen3.5 ships with MTP weights that both MLX and transformers strip during conversion. Reverse-engineered the architecture from HuggingFace weight shapes. Built a self-speculative decoder -- the model drafts its own next token using a small MTP head (3ms overhead vs. 34ms forward pass). 79% acceptance rate, no draft model needed.

**Phase 4: Split-recurrence rollback.** The DeltaNet layers have recurrent state that must be rolled back on rejection. Naive checkpoint/restore added 34ms * 21% = 7ms overhead. Realized MLX arrays are immutable -- just save references. Then split the GDN recurrence into per-token calls while keeping matmuls batched. Zero-cost rollback: 42.7 tok/s.

## Dead ends (the interesting parts)

- **V6 custom Metal kernels**: Wrote `fused_add_rms_norm`, `silu_mul_rms_norm`, `dual_rms_norm`. Slower than stock because they broke `mx.compile`'s automatic fusion.
- **qmv_fast kernel tuning**: Tried `results_per_simdgroup=8` and `num_simdgroups=4`. Register pressure killed occupancy. Apple's kernel is already well-tuned.
- **group_size=128 quantization**: 2.3x quantization error for 2.8% speed. Bad trade.
- **GPU-resident autoregressive loop**: `mx.compile` is a dispatch scheduler, not a kernel fuser. Unrolling N steps gives ~0% gain.
- **CPU draft model**: 3716 ms/tok. MLX CPU path has no Metal acceleration.
- **CoreML/ANE draft**: `coremltools` broken on Python 3.14, DeltaNet ops unsupported in CoreML converter.

## Hardware

- Apple M4 Max (16-core GPU, 128 GB unified memory)
- Theoretical peak bandwidth: 546 GB/s
- Model: Qwen3.5-27B-4bit (13.7 GB total weights)
- Theoretical minimum: 25.1 ms/tok (39.8 tok/s at 100% BW utilization)
