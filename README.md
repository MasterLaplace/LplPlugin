# FullDive Engine — LplPlugin

> **A modular, experimental ultra-optimized C++23 engine for neuro-immersive simulations.**

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Build: xmake](https://img.shields.io/badge/build-xmake-brightgreen)](https://xmake.io)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)]()

---

## Architecture Highlights

This codebase has been thoroughly refactored from its legacy roots into a **flat, modular architecture** of 16 independent libraries orchestrated by xmake:

- **Lockless Zero-Copy IPC** — Linux kernel module (`/dev/lpl0`) with `vmalloc_user` mapped memory and `smp_*` lockless ring buffers for microsecond-latency network I/O.
- **Data-Oriented ECS** — Flat archetypes, atomic concurrent registries, cache-aligned SoA component layouts with double-buffered hot data.
- **DDA Task Scheduler** — System dependency graph (DAG) split into PreSwap/PostSwap phases, executed over an integrated ThreadPool with work-stealing.
- **Deterministic Fixed-Point Math** — `Fixed32` types with custom CORDIC trigonometry for cross-platform deterministic physics.
- **Flat Dynamic Octree** — Morton (Z-order) encoded spatial partitioning with LSD Radix Sort and adaptive brute-force/octree broadphase.
- **Vulkan Renderer** — Integrated Vulkan pipeline (ported from [VkWrapper](https://github.com/EngineSquared/VkWrapper.git)) with ImGui support.
- **BCI Integration** — Brain-Computer Interface module (OpenBCI Cyton, 8ch/250Hz) with real-time DSP: Schumacher $R(t)$, Riemannian $\delta_R$, Mahalanobis $D_M$.

> [!NOTE]
> This is an experimental project currently under development, not a distributable software product.
> I am currently testing an architecture that combines various personal projects I have undertaken. It is still in the testing phase and is not yet fully functional.
> However, the `_legacy` folder contains a functional version of the original proof of concept for this repository.

---

## Directory Structure

```
LplPlugin/
├── core/           — Platform abstraction, types, assertions, logging
├── math/           — Vec3, Quat, Fixed32, CORDIC, BoundaryBox
├── memory/         — PinnedAllocator (CUDA zero-copy), pool allocators
├── container/      — FlatAtomicsHashMap, Morton encoding, sparse sets
├── concurrency/    — ThreadPool, SpinLock, atomic utilities
├── ecs/            — Entity registry, Partition (SoA double-buffered chunks)
├── physics/        — WorldPartition, collision (AABB, octree broadphase)
├── net/            — UDP transport (kernel driver / socket fallback), protocol
├── gpu/            — CUDA physics kernels, GPU lifecycle
├── input/          — InputManager (keys, axes, neural state per entity)
├── render/         — Vulkan pipeline, ImGui integration
├── audio/          — Spatial audio (stub)
├── haptic/         — Haptic/vestibular feedback (stub)
├── bci/            — OpenBCI driver, FFT, SignalMetrics, RiemannianGeometry, NeuralMetrics
├── serial/         — Serial port abstraction
├── engine/         — Top-level facade aggregating all modules, game loop
├── kernel/         — Linux kernel module (lpl_kmod.c) — Netfilter + ring buffers
├── apps/           — Executables (lpl-server, lpl-client, lpl-benchmark)
└── _legacy/        — Previous-generation prototype preserved for reference
```

Each module is a **static library** (`lpl-<name>`) with its own `xmake.lua`, `include/lpl/<name>/`, and `src/`.

---

## Building the Engine

The project uses **[xmake](https://xmake.io)** exclusively.

### Prerequisites

- **Linux** with kernel headers (for the kernel module)
- **xmake** ≥ 2.9.0
- **GCC 13+** or **Clang 17+** (C++23 support)
- **Vulkan SDK** (headers + loader)
- NVIDIA CUDA Toolkit *(optional — automatic CPU fallback)*

### Build Commands

```bash
# Configure (clean cache)
xmake f -c

# Build all targets
xmake build -j$(nproc)

# Build specific targets
xmake build lpl-server
xmake build lpl-client
xmake build lpl-benchmark

# Build with options
xmake f --renderer=n    # Disable Vulkan (useful for server)
xmake f --cuda=y        # Enable CUDA physics

# Kernel Module Management
xmake kmod-build        # Build the kernel module
xmake kmod-install      # Load into kernel (insmod)
xmake kmod-uninstall    # Unload from kernel (rmmod)

# Run
xmake run lpl-server
xmake run lpl-client
xmake run lpl-benchmark
```

### Build Modes

```bash
xmake f -m debug      # Debug symbols, no optimization, LPL_DEBUG
xmake f -m release    # Full optimization, stripped, NDEBUG
xmake f -m profile    # Debug symbols + full optimization, LPL_PROFILE
```

---

## Documentation

📖 Full documentation available in the **[Wiki](LplPlugin.wiki/Home.md)** — architecture deep-dives, module reference, ADRs, benchmarks, and scientific contributions.

---

## License

This project is licensed under the **GPL-3.0** — see [LICENSE](LICENSE) for details.
