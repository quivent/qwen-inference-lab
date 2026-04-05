# Bandwidth Analysis: Qwen3.5-27B on M4 Max

Where the time actually goes during decode.

---

## Weight Budget

| Component | Size |
|---|---|
| Model weights (4-bit quantized) | 12.2 GB |
| Scales + biases (group_size=64) | 1.52 GB |
| **Total** | **13.7 GB** |

The metadata overhead (scales/biases) is 11.1% of total. At group_size=64, every 64 weights share one scale and one bias value. This is 12.4% of the weight data itself.

With group_size=128, metadata drops to ~0.76 GB, saving ~5.5% bandwidth. But quantization error increases 2.3x. Not worth it for this model.

---

## Theoretical Limits

| Metric | Value |
|---|---|
| M4 Max peak bandwidth | 546 GB/s |
| Total weight reads per token | 13.7 GB |
| Theoretical minimum latency | 25.1 ms/tok |
| Theoretical maximum throughput | 39.8 tok/s |
| At 85% sustained BW | 34.6 tok/s |
| At 80% sustained BW | 31.9 tok/s |

The theoretical maximum assumes reading every weight exactly once per token with perfect bandwidth utilization. In practice, you also need to read activations, KV cache entries, and intermediate buffers.

---

## Actual Measurements

### End-to-end

| Configuration | ms/tok | tok/s | BW utilization |
|---|---|---|---|
| Stock mlx_lm (async) | 33.8 | 29.5 | 74% |
| V5 monolithic (async) | 33.3 | 30.0 | 75% |
| Stock mlx_lm (sync) | 36.2 | 27.6 | 69% |
| V5 monolithic (sync) | 34.6 | 28.9 | 72% |

### Per-Matmul Profiling

Individual matmul bandwidth (measured via `mx.metal` timing):

| Matmul | Bandwidth |
|---|---|
| Fused input projection (DeltaNet) | 219 GB/s |
| Out projection | 228 GB/s |
| MLP gate+up (fused) | 234 GB/s |
| MLP down | 231 GB/s |
| QKV projection (attention) | 222 GB/s |

Individual matmuls achieve 40-43% of peak bandwidth. This is expected -- each matmul is a short-running kernel that can't fully saturate the memory controller.

When pipelined (consecutive matmuls dispatched without sync), effective bandwidth reaches ~439 GB/s (80% of peak). The memory controller stays busy because the next kernel starts before the previous one fully drains.

---

## Gap Breakdown

Total gap: 33.8ms actual vs 25.1ms theoretical = 8.7ms overhead.

| Source | Estimated Cost | Notes |
|---|---|---|
| Kernel dispatch overhead | 3-5 ms | ~930 operations per token (fewer actual GPU dispatches after mx.compile fusion) |
| Non-matmul compute | 2-3 ms | RMS norms, activations, transposes, concat, argmax |
| Memory controller inefficiency | 1-2 ms | Bank conflicts, TLB misses, non-sequential access patterns |
| KV cache reads | 0.5-1 ms | 16 attention layers, grows with context length |
| Python overhead (sync mode) | 1.5-2 ms | Eliminated by async_eval or V5 compile |

---

## CPU Graph Build Time

| Configuration | Graph build time |
|---|---|
| Stock mlx_lm | 1.71 ms |
| V5 monolithic compile | 0.64 ms |

The V5 monolithic compile reduces CPU graph build time by 63%, but this is entirely hidden by `async_eval` in practice. It only matters in sync mode.

---

## Why sync ~ async

The GPU takes ~34ms per token. The CPU takes ~1.7ms to build the graph (stock) or ~0.6ms (V5). Since GPU >> CPU, the CPU is never the bottleneck. `async_eval` hides the CPU time behind GPU execution, but there's almost nothing to hide.

The 2.4ms difference between sync and async modes (36.2 vs 33.8 ms/tok) represents the CPU graph build + Python dispatch time that async successfully overlaps with GPU execution.

---

## Implications

1. **Kernel optimization has diminishing returns.** At 80% pipelined bandwidth utilization, you're within 20% of the hardware limit. Getting to 90% would require changes to the MLX runtime itself (kernel launch coalescing, better memory pooling).

2. **Speculative decoding is the right lever.** Instead of making each forward pass faster, get more tokens per forward pass. Even at 79% MTP acceptance, you amortize the 34ms cost over 1.79 tokens instead of 1.

3. **The metadata tax is real but tolerable.** 11.1% of bandwidth goes to reading quantization scales/biases. group_size=128 halves this but the quality cost is too high. group_size=64 is the right trade-off.

4. **Memory bandwidth is the fundamental constraint.** The M4 Max reads 13.7 GB per token. At 546 GB/s, that's 25.1ms minimum. No amount of software optimization can beat this without reducing the amount of data read (smaller model, more aggressive quantization, or pruning).
