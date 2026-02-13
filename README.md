# LplPlugin

Low-latency real-time simulation engine using zero-copy architecture from NIC to GPU, with spatial partitioning for MMO-scale worlds.

## Overview

LplPlugin is a high-performance engine designed to minimize latency in networked real-time simulations. It combines a Linux kernel module for zero-copy UDP packet ingestion with a WorldPartition system that spatially organizes entities into Morton-encoded chunks — each with its own SoA ECS storage and GPU-accelerated physics.

**Key features:**
- Zero-copy packet path: NIC → Kernel → Userspace → GPU
- WorldPartition with Morton-encoded spatial hashing (supports negative coordinates)
- Per-chunk SoA ECS with pinned memory (`cudaHostAllocMapped`)
- GPU physics via CUDA kernels (Euler semi-implicit, per-chunk dispatch)
- Lock-free chunk storage (`FlatAtomicsHashMap`) with wait-free reads
- Automatic entity migration between chunks (swap-and-pop)
- EntityRegistry: generational sparse set + chunk routing for O(1) lookup
- Dynamic octree for spatial region queries
- Sub-100μs kernel-to-userspace latency
- Dynamic packet format for extensibility

## Architecture

```
UDP Packet (port 7777)
    ↓
Netfilter Hook (lpl_kmod.c kernel module)
    ↓
Ring Buffer (mmap'd, lockless and zero-copy)
    ↓
engine_consume_packets() → dispatch to WorldPartition chunks
    ↓
engine_physics_tick() → CUDA kernel per chunk (gravity + Euler)
    ↓
migrateEntities() → bounds check + swap-and-pop + chunk reassignment
```

### File Structure

| File | Role |
|------|------|
| `lpl_protocol.h` | Network structures shared with kernel (C-compatible) |
| `lpl_kmod.c` | Linux kernel module — Netfilter hook, ring buffer via mmap |
| `Engine.cuh` | Engine header — CUDA macros, ECS constants, API declarations |
| `Engine.cu` | CUDA kernels (physics) + network→WorldPartition dispatch |
| `main.cpp` | Unified entry point: kernel link → simulation loop (CUDA) |
| `server.cpp` | Standalone CPU-only demo (no CUDA, no kernel module) |
| `WorldPartition.hpp` | Chunk orchestrator — Morton keys, migration, registry |
| `Partition.hpp` | Per-chunk SoA ECS — entities, physics, setters, migration |
| `EntityRegistry.hpp` | Sparse set + chunk routing — O(1) entity lookup |
| `FlatAtomicsHashMap.hpp` | Lock-free hash map with pool-based storage |
| `FlatDynamicOctree.hpp` | Dynamic octree for spatial region queries |
| `Math.hpp` | Vec3, Quat, BoundaryBox (`__host__ __device__` annotated) |
| `Morton.hpp` | Morton encoding (2D/3D) |
| `SpinLock.hpp` | SpinLock + RAII LocalGuard |
| `PinnedAllocator.hpp` | CUDA pinned memory allocator for `std::vector` |
| `World/` | Legacy implementation (kept for reference) |

## Build & Run

### Prerequisites
- Linux kernel headers
- NVIDIA CUDA toolkit (nvcc)
- GCC 12+

### Full build (engine + kernel module)
```bash
make            # Builds both 'driver' and 'engine' targets
make install    # Loads kernel module (sudo)
make run        # Starts the engine
```

### CPU-only demo (no CUDA, no kernel module)
```bash
make server     # Builds standalone WorldPartition demo
./server        # Runs the simulation with CPU physics
```

### Cleanup
```bash
make uninstall  # Removes kernel module
make clean      # Removes build artifacts
```

### Debug
```bash
dmesg | tail -n 50  # Kernel module logs
```

## Technical Details

### ECS Architecture
- **Storage:** Structure of Arrays (SoA) per chunk, using `PinnedAllocator` (CUDA pinned memory)
- **Components:** position (Vec3), rotation (Quat), velocity (Vec3), mass (float), force (Vec3), size (Vec3), health (int32)
- **Entity IDs:** Generational sparse set (`EntityRegistry`) with chunk routing
- **Pipeline:** Network → Dispatch to chunk → GPU Physics → Migration check → Re-insertion

### Packet Format
Dynamic format supporting variable components per entity:
```
[EntityID(4B)][CompID(1B)][Data]...[CompID(1B)][Data]
```
Component IDs: `COMP_TRANSFORM(1)`, `COMP_HEALTH(2)`, `COMP_VELOCITY(3)`, `COMP_MASS(4)`

### Memory Management
- **Kernel:** Static ring buffer, atomic head/tail (lockless, mmap'd)
- **Userspace:** Pinned memory via `PinnedAllocator` (`cudaHostAllocMapped`) for zero-copy GPU access, PCIe optimization
- **GPU:** Device pointers obtained via `cudaHostGetDevicePointer` on pinned SoA vectors

### World Partitioning
- **Chunks:** 255×∞×255 unit partitions, indexed by Morton 2D codes with bias for negative coordinates
- **Capacity:** Up to 65,536 chunks (`FlatAtomicsHashMap` with 22-bit pool indices)
- **Physics:** CUDA kernel `kernel_physics_tick` launched per chunk, then CPU migration pass
- **Migration:** Backward-iterating bounds check + swap-and-pop, avoiding double-tick of swapped entities
- **Lookup:** `EntityRegistry` maps `publicId → chunkKey` in O(1), `Partition::_idToLocal` maps `entityId → localIndex` in O(1)

## Performance

Benchmark results (13 February 2026):

| Test Case | Duration | Throughput | Lookup Time | Status |
|-----------|----------|-----------|------------|--------|
| **10k entities, 100 frames** | 0.026 sec | 38.6 M ops/sec | 70.7 ns | ✅ Stable |
| **50k entities, 200 frames** | 0.180 sec | 55.6 M ops/sec | 71.7 ns | ✅ Excellent |

**Key metrics:**
- **EntityRegistry lookup:** 0.059-0.073 µs (O(1) sparse set)
- **Partition::findEntityIndex:** 71 ns (adaptive sparse set, O(1) guaranteed)
- **Active chunks:** 1598 (10k) → 1700 (50k)
- **Avg entities per chunk:** 6.3 (10k) → 29.4 (50k)
- **Stability:** Zero crashes, linear scaling, no memory hangs
- **Deprecated:** Previous unordered_map caused 4MB allocations per chunk; adaptive sparse set reduces to 64-256KB

## Roadmap

- [ ] Client-side prediction & reconciliation
- [ ] Double buffering per-chunk (write/read separation for concurrent GPU render)
- [ ] GPUDirect RDMA support (NIC → GPU bypass)
- [ ] Distributed physics with server meshing (chunk offloading)
- [ ] 100k+ entities stress testing

## License

See [LICENSE](LICENSE)
