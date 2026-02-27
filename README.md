# FullDive Engine (LPL Plugin)

Welcome to the **FullDive Engine**, the highly optimized, C++23 foundation for neuro-immersive simulations.

## Architecture Highlights
This codebase has been thoroughly refactored from its legacy roots into a modular, horizontally scalable architecture powered by:
- **Lockless Zero-Copy IPC**: Utilizing Linux kernel modules (`/dev/lpl0`) with `vmalloc_user` mapped memory and `smp_*` lockless ring buffers. Bypasses standard VFS syscall constraints for microsecond-latency network broadcasting. 
- **Data-Oriented ECS**: A highly optimized Entity Component System (`lpl::ecs`) using flat archetypes, atomic concurrent registries, and strictly cache-aligned component layouts.
- **DDA Task Scheduler**: System dependency graph scheduling execution over an integrated Thread Pool leveraging work-stealing and concurrent atomic queues.
- **Deterministic Fixed-Point Math**: Replaced floating-point drift with `Fixed32` types combined with custom CORDIC trigonometry functions for guaranteed deterministic cross-platform physics validation.
- **Flat Dynamic Octree**: Fast, recursive 3D spatial partitioning utilizing Morton (Z-order) encoded bit-interleaving and high-performance LSD Radix Sorting variants.

## Directory Structure
- `apps/` — Entry points (Server headless, Client visual, and Benchmarking suites).
- `bci/` — Brain-Computer Interface adapters for non-invasive (EEG) hardware, integrating OpenBCI via LSL with DSP processing lines.
- `kernel/` — Linux kernel module source (`lpl_kmod.c`) injecting directly into the Netfilter stack (`NF_INET_PRE_ROUTING`) for instant UDP ingestion.
- `ecs/`, `physics/`, `net/`, `math/`, `core/` — Individual modules comprising the core runtime.
- `_legacy/` — The previous generation's prototype codebase preserved for reference.

## Building the Engine

The project relies entirely on **xmake**. 

```bash
# Clean configuration and cache
$ xmake f -c

# Build all targets (lpl-server, lpl-client, lpl-benchmark)
$ xmake build -j$(nproc)
```

## Running the Verification Suite
To gauge component-level latency and general runtime throughput:
```bash
$ xmake run lpl-benchmark
```
