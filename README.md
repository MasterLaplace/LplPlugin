# LplPlugin

Low-latency real-time simulation engine using zero-copy architecture from NIC to GPU.

## Overview

LplPlugin is a proof-of-concept engine designed to minimize latency in networked real-time simulations. It uses a Linux kernel module to intercept UDP packets and deliver them to userspace with minimal overhead using zero-copy techniques.

**Key features:**
- Zero-copy packet path: NIC → Kernel → Userspace → GPU
- ECS (Entity Component System) with GPU acceleration
- Sub-100μs kernel-to-userspace latency
- Dynamic packet format for extensibility

## Architecture

```
UDP Packet (port 7777)
    ↓
Netfilter Hook (kernel module)
    ↓
Ring Buffer (mmap'd, zero-copy)
    ↓
Userspace ECS Engine
    ↓
CUDA Kernels (physics on GPU)
```

**Components:**
- `lpl_kmod.c`: Linux kernel module with Netfilter hook
- `main.c`: Userspace simulation loop
- `plugin.cu`: ECS implementation + CUDA physics kernels
- `World/`: Spatial partitioning (Octree, Morton codes) (obsolete, being refactored)

## Current Performance

Measured on a standard consumer system:
- **Latency:** ~62.55μs average frame time
- **Packet loss:** 0% (1000 packets tested)
- **Throughput:** ~495 packets/sec
- **Stability:** <15% variance

## Build & Run

### Prerequisites
- Linux kernel headers
- NVIDIA CUDA toolkit
- GCC

### Compilation
```bash
make
```

### Installation
```bash
make install  # Loads kernel module
```

### Run
```bash
make run      # Starts simulation
```

### Cleanup
```bash
make uninstall  # Removes kernel module
```

### Debug
```bash
dmesg | tail -n 50  # Kernel logs
```

## Technical Details

### ECS Architecture
- **Storage:** Structure of Arrays (SoA) for GPU coalescing
- **Entity IDs:** Sparse sets with generational indices
- **Pipeline:** Network → Physics → Render

### Packet Format
Dynamic format supporting variable components per entity:
```
[EntityID][ComponentID1][Data1]...[ComponentIDN][DataN]
```

### Memory Management
- **Kernel:** Static ring buffer, atomic head/tail (lockless)
- **Userspace:** Pinned memory (`cudaHostAllocMapped`) for PCIe optimization
- **GPU:** Double buffering with atomic swaps

### World Partitioning System *(in integration)*
High-performance entity management layer for MMO-scale simulations:
- **WorldPartition:** Chunk orchestrator with Morton-encoded spatial hashing
- **Partition:** Per-chunk ECS with SoA storage (cache-friendly for GPU transfer)
- **FlatAtomicsHashMap:** Lock-free chunk storage with wait-free reads
- **Migration System:** Automatic entity transfer between chunks (swap-and-pop)
- **Spatial Indexing:** Dynamic octree for region queries

**Current Status:**  
Core implementation complete (WorldPartition.hpp, Partition.hpp, FlatDynamicOctree.hpp). Integration with the CUDA plugin pipeline is in progress—some components may be ported to GPU kernels for parallel processing.

## Roadmap

- [ ] Client-side prediction & reconciliation
- [ ] Spatial partitioning for large-scale simulations (100k+ entities)
- [ ] GPUDirect RDMA support (NIC → GPU bypass)
- [ ] Distributed physics with spatial sharding

## License

See [LICENSE](LICENSE)
