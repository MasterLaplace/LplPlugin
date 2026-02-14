# LplPlugin

Low-latency real-time simulation engine using zero-copy architecture from NIC to GPU, with spatial partitioning for MMO-scale worlds.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Build & Run](#build--run)
5. [Technical Details](#technical-details)
6. [Performance](#performance)
7. [Roadmap](#roadmap)
8. [License](#license)

## Overview

LplPlugin is a high-performance engine designed to minimize latency in networked real-time simulations. It combines a Linux kernel module for zero-copy UDP packet ingestion with a WorldPartition system that spatially organizes entities into Morton-encoded chunks.

**Key features:**
- **Zero-copy packet path:** NIC → Kernel → Userspace → GPU
- **WorldPartition:** Morton-encoded spatial hashing with lock-free storage (`FlatAtomicsHashMap`).
- **Parallel ECS Scheduler:** DAG-based `SystemScheduler` with automatic dependency resolution and parallel stage execution.
- **Optimized ThreadPool:** Custom thread pool with `std::future` support and local-thread optimization for single-batch workloads.
- **Hybrid Physics:**
  - **GPU:** CUDA kernels (Euler semi-implicit) for massive scale.
  - **CPU:** Parallelized fallback via `ThreadPool` for compatibility.
- **EntityRegistry:** Generational sparse set O(1) lookup.

## Architecture

```
[UDP Packet] → [Kernel Module (lpl_kmod)] → [Ring Buffer (mmap)]
                                                   ↓
                                           [Engine Dispatch]
                                                   ↓
                                         [SystemScheduler DAG]
                                                   ↓
                                   [WorldPartition (Morton Chunks)]
                                                   ↓
                                        [Physics (CUDA/CPU)]
```

### Core Components

| File | Role |
|------|------|
| `lpl_kmod.c` | Linux kernel module — Netfilter hook, ring buffer via mmap |
| `SystemScheduler.hpp` | DAG-based ECS scheduler. Resolves R/W dependencies automatically. |
| `WorldPartition.hpp` | Spatially partitioned world. Orchestrates physics steps (CPU/GPU). |
| `ThreadPool.hpp` | High-performance thread pool with `enqueueDetached` for low-overhead tasks. |
| `FlatAtomicsHashMap.hpp` | Lock-free hash map with `forEachParallel` optimization. |
| `Partition.hpp` | Per-chunk SoA ECS storage. |

## Build & Run

### Prerequisites
- Linux kernel headers
- NVIDIA CUDA toolkit (nvcc) (Optional, auto-fallback to CPU)
- GCC 12+

### Quick Start
```bash
make            # Builds driver and engine
make install    # Loops kernel module (optional)
make run        # Starts engine
```

if you wish to test with a visual client, then in another terminal, run:
```
make visual
./visual
```

## Technical Details

### 1. System Scheduler
The `SystemScheduler` builds a dependency graph (DAG) based on component access (`Read`/`Write`).
- **Automatic Parallelism:** Systems in the same stage run concurrently via `ThreadPool`.
- **Synchronization:** Stages are synchronized using `std::latch`.
- **Performance:** Uses `enqueueDetached` to minimize `std::future` overhead.

### 2. World Partitioning & Parallelism
- **Storage:** `FlatAtomicsHashMap` stores chunks with 22-bit indices.
- **Parallel Loop:** `forEachParallel` dynamically batches work across the `ThreadPool`.
- **Optimization:** If only 1 batch is needed (small workload or single core), it executes inline on the calling thread to avoid context switching.

### 3. Physics Pipeline
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

- [ ] Client-side prediction & reconciliation
- [ ] GPUDirect RDMA support (NIC → GPU bypass)
- [ ] 100k+ entities stress testing

## License

See [LICENSE](LICENSE)
