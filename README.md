<div align="center">

```
  ___  __  __ _____ ____  
 / _ \|  \/  |_   _|  _ \ 
| | | | |\/| | | | | |_) |
| |_| | |  | | | | |  __/ 
 \__\_\_|  |_| |_| |_|    
 I N F E R E N C E  L A B
```

**Exploration log: optimizing Qwen3.5-27B from 29.5 to 51.1 tok/s on Apple Silicon.**

*Pushing token generation as fast as possible on a single M4 Max.*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS-lightgrey.svg?style=for-the-badge&logo=apple)](https://apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📑 Table of Contents
- [🎯 Goal](#-goal)
- [📊 Results](#-results)
- [📖 The Journey](#-the-journey)
- [💀 Dead Ends](#-dead-ends)
- [📁 What's in Here](#-whats-in-here)
- [💻 Hardware Profile](#-hardware-profile)

---

## 🎯 Goal

Push Qwen3.5-27B token generation as fast as possible on a single M4 Max, starting from stock `mlx_lm` and ending wherever the hardware limits take us.

---

## 📊 Results

| Configuration | tok/s | vs. baseline |
|---|---|---|
| Stock mlx_lm | 29.5 | 1.00x |
| V5 monolithic compile | 30.0 | 1.02x |
| Stock + spec decode (0.8B draft) | 37.6 | 1.27x |
| MTP head (self-speculative) | 36.9 | 1.25x |
| MTP + split-recurrence rollback | 42.7 | 1.45x |
| Adaptive MTP chain (Huihui abliterated) | 49.5 | 1.68x |
| Adaptive MTP chain (vanilla) | **51.1** | **1.73x** |

> [!NOTE]
> Starting point: 29.5 tok/s. Current best: 51.1 tok/s (adaptive MTP confidence chain + batch verify).

---

## 📖 The Journey

**Phase 1: Kernel fusion (V2-V6)**
Spent weeks fusing DeltaNet projections and writing custom Metal kernels. Net gain: +1.7%. The GPU was already the bottleneck. Custom kernels actually broke the fusion graph, making things slower.

**Phase 2: Speculative decoding**
Initially hit 26.5 tok/s due to a broken benchmark. Re-tested and got 37.6 tok/s. Measure correctly before declaring it dead.

**Phase 3: MTP (Multi-Token Prediction)**
Discovered Qwen3.5 ships with MTP weights stripped by MLX. Built a self-speculative decoder drafting its own next token (3ms overhead vs 34ms pass). 79% acceptance rate.

**Phase 4: Split-recurrence rollback**
DeltaNet layers need recurrent state rolled back. Naive restore added 7ms overhead. By saving references and splitting GDN recurrence, we achieved zero-cost rollback at 42.7 tok/s.

---

## 💀 Dead Ends

- **V6 custom Metal kernels**: Slower than stock because they broke `mx.compile`'s fusion.
- **qmv_fast kernel tuning**: Register pressure killed occupancy.
- **group_size=128 quantization**: 2.3x error for 2.8% speed.
- **GPU-resident autoregressive loop**: ~0% gain.
- **CPU draft model**: 3716 ms/tok. No Metal acceleration.
- **CoreML/ANE draft**: `coremltools` broken, DeltaNet ops unsupported.

---

## 📁 What's in Here

- `docs/TIMELINE.md` -- Full history of approaches
- `docs/BANDWIDTH_ANALYSIS.md` -- Profiling work
- `docs/HUIHUI_ABLITERATED.md` -- Uncensored variant
- `kernels/fused_gdn.py` -- Fused kernel code
- `benchmarks/bench_v7.py` -- Speculative decoding benchmark harness
- `benchmarks/extract_mtp_huihui.py` -- Parametrized MTP head extractor
- `logs/` -- Server outputs and revalidation runs

---

## 💻 Hardware Profile

- **Device**: Apple M4 Max (16-core GPU, 128 GB unified memory)
- **Bandwidth**: 546 GB/s
- **Model**: Qwen3.5-27B-4bit (13.7 GB total weights)
- **Theoretical minimum**: 25.1 ms/tok (39.8 tok/s at 100% BW utilization)
