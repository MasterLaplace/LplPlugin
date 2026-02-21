# engine — Real-time ECS Engine (header-only)

High-performance simulation engine based on **ECS/SoA** with Morton spatial partitioning, DAG scheduling, and hybrid CPU/GPU physics (CUDA).

## Contents

```
engine/
├── WorldPartition.hpp     — Partitioned world (Morton chunks, double buffer)
├── Partition.hpp          — SoA storage per chunk (ECS)
├── EntityRegistry.hpp     — Generational sparse set O(1)
├── SystemScheduler.hpp    — DAG scheduler with automatic parallelism
├── ThreadPool.hpp         — Custom thread pool (enqueueDetached, latch)
├── FlatAtomicsHashMap.hpp — Lock-free hash map (22-bit slots)
├── FlatDynamicOctree.hpp  — Dynamic octree for spatial queries
├── Network.hpp            — Kernel module interface + packet dispatch
├── PhysicsGPU.cu/.cuh     — CUDA kernels (semi-implicit Euler)
├── Morton.hpp             — 3D Morton encoding/decoding
├── SpinLock.hpp           — Lightweight spin lock
└── PinnedAllocator.hpp    — Pinned memory allocator (CUDA pinned)
```

## ECS Architecture

```
[EntityRegistry] ──── entities ────► [Partition (SoA)]
                                             │
[SystemScheduler] ──── DAG ──────► [WorldPartition]
                                             │
                                 ┌───────────┴───────────┐
                            [GPU Physics]           [CPU Physics]
                            (CUDA kernels)          (ThreadPool)
```

## Simulation Pipeline (per frame)

1. **Network** — `Network::poll()` consumes ring buffer, dispatches `STATE`/`INPUT`/`CONNECT`
2. **Scheduling** — `SystemScheduler::run()` launches systems in parallel by stage
3. **Partitioning** — `WorldPartition::step()` iterates over active chunks
4. **Physics** — CUDA kernels (if available) or `ThreadPool` fallback
5. **Double buffer** — atomic frame swap

## Technical Details

| Component | Detail |
|-----------|--------|
| `Morton.hpp` | 3D Z-order 64-bit encoding — cache locality for spatial traversal |
| `FlatAtomicsHashMap` | `std::atomic<uint64_t>` per slot — wait-free insertions |
| `PinnedAllocator` | `cudaHostAlloc` → GPU access zero-copy via `cudaHostGetDevicePointer` |
| `SystemScheduler` | R/W dependency graph → parallel stages synchronized by `std::latch` |

## Build

**Header-only** library — no separate compilation except `PhysicsGPU.cu` (CUDA optional).

```bash
make server     # Compiles apps/server with -I../../engine
```

CUDA is auto-detected by the Makefile; if `nvcc` is absent, CPU physics is used.
