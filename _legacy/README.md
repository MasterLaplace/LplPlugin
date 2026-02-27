# LplPlugin

Adaptive closed-loop BCI platform — from raw EEG signal to real-time simulation, with zero-copy kernel pipeline and computational neuroscience metrics.

## Table of Contents

1. [Overview](#overview)
2. [Plugins](#plugins)
3. [BCI Scientific Contributions](#bci-scientific-contributions)
4. [Build & Run](#build--run)
5. [Technical Details](#technical-details)
6. [Performance](#performance)
7. [Roadmap](#roadmap)
8. [License](#license)

## Overview

LplPlugin is an **adaptive BCI platform** combined with a high-performance simulation engine. It implements a complete closed-loop: multi-channel EEG acquisition (OpenBCI Cyton, 8ch/250 Hz) → spectral and geometric metrics → real-time adaptation of visual feedback.

**Application domain:** BCI-assisted motor retraining, neuro-adaptive interfaces, haptic feedback conditioned by neural state.

**Key scientific contributions:**
- **Schumacher R(t)** — muscle tension indicator 40-70 Hz inter-channel
- **Riemannian Distance δ_R** — cognitive state change detection on SPD covariance matrices
- **Mahalanobis Distance D_M** — anomaly detection in EEG feature space

**System infrastructure:**
- **Zero-copy NIC → GPU** via Netfilter kernel module + `mmap` + CUDA
- **WorldPartition ECS** with Morton spatial hashing and lock-free hash map
- **SystemScheduler DAG** — automatic parallelism via R/W dependency resolution

## Plugins

| Plugin | Role | See |
|--------|------|-----|
| [`bci/`](bci/README.md) | OpenBCI driver, FFT, Schumacher, Riemann, Mahalanobis | [README](bci/README.md) |
| [`engine/`](engine/README.md) | ECS SoA, WorldPartition, SystemScheduler, CUDA physics | [README](engine/README.md) |
| [`kernel/`](kernel/README.md) | Netfilter kernel module + zero-copy ring buffer | [README](kernel/README.md) |
| [`shared/`](shared/README.md) | Binary protocol + CPU/GPU mathematics | [README](shared/README.md) |

## BCI Scientific Contributions

### Schumacher R(t)

$$R(t) = \frac{1}{N_{ch}} \sum_{i=1}^{N_{ch}} \int_{40}^{70} \mathrm{PSD}_i(f,t)\, df$$

Measures power in the 40-70 Hz band across all channels. Used as a proxy for **muscle tension** and fatigue. High R(t) triggers a pause or feedback reduction.

### Riemannian Distance δ_R

$$\delta_R(C_1, C_2) = \sqrt{\sum_i \ln^2(\lambda_i)}$$

Geodesic metric on the manifold of symmetric positive definite matrices. **Congruence invariant** — same distance before and after pre-multiplication by a non-singular matrix, making it robust to volume-conduction artifacts.

### Mahalanobis Distance D_M

$$D_M(x_t) = \sqrt{(x_t - \mu_c)^T \Sigma_c^{-1} (x_t - \mu_c)}$$

Anomaly detection in feature space. Reduces to Euclidean distance when Σ_c = I.

**References:**
- Schumacher et al., *Closed-loop BCI for fatigue monitoring*, 2015
- Blankertz et al., *Single-trial EEG using Riemannian covariance matrices*, 2011
- Moakher, *A Riemannian framework for the geometric mean of SPD matrices*, 2005

## Build & Run

### Prerequisites
- Linux kernel headers
- NVIDIA CUDA toolkit (nvcc) (Optional, auto-fallback to CPU)
- GCC 12+

### Quick Start
```bash
make            # Builds driver and engine
make install    # Installs kernel module (required)
make run        # Starts engine
```

If you wish to test with a visual client, then in another terminal, run:
```
make visual
./visual
```

## Technical Details

### 1. Networking (Kernel Bypass-like)
-   **Zero-Copy:** Packets are intercepted by `lpl_kmod` at the netfilter level and written directly to a shared memory ring buffer.
-   **Lockless:** The userspace `Network` class poll consumes packets from the ring buffer without system calls during the game loop.
-   **Protocol:** Custom binary protocol for `CONNECT`, `INPUT`, and `STATE` synchronization.

### 2. System Scheduler
The `SystemScheduler` builds a dependency graph (DAG) based on component access (`Read`/`Write`).
- **Automatic Parallelism:** Systems in the same stage run concurrently via `ThreadPool`.
- **Synchronization:** Stages are synchronized using `std::latch`.
- **Performance:** Uses `enqueueDetached` to minimize `std::future` overhead.

### 3. World Partitioning & Parallelism
- **Storage:** `FlatAtomicsHashMap` stores chunks with 22-bit indices.
- **Parallel Loop:** `forEachParallel` dynamically batches work across the `ThreadPool`.
- **Optimization:** If only 1 batch is needed (small workload or single core), it executes inline on the calling thread to avoid context switching.

### 4. Physics Pipeline
- **GPU Path:** Zero-copy `cudaHostGetDevicePointer` → Kernel launch per chunk.
- **CPU Path:** Parallel execution over chunks using the ThreadPool.

## Performance

**Test Run (CPU Fallback, 50 entities):**
- **Physics Step:** ~0.060 ms (avg)
- **Throughput:** Stable at high frame rates.
- **Scaling:** Linear scaling with entity count due to spatial partitioning.

**Previous Benchmarks (10k entities):**
- **Throughput:** ~38.6 M ops/sec
- **Lookup:** ~70ns

## Roadmap

- [x] Multi-channel OpenBCI Cyton driver (8 ch)
- [x] Schumacher R(t) with unit tests
- [x] Riemannian Distance (Jacobi eigenvalue, no external dependency)
- [x] Mahalanobis Distance
- [ ] Automatic calibration phase (30s baseline)
- [ ] Visual feedback conditioned by NeuralMetrics
- [ ] GPUDirect RDMA (NIC → GPU bypass)
- [ ] Stress test 100k+ entities

## License

See [LICENSE](LICENSE)

