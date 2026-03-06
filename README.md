# FullDive Engine — LplPlugin

> A modular, experimental ultra-optimized C++23 engine for neuro-immersive simulations.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Build: xmake](https://img.shields.io/badge/build-xmake-brightgreen)](https://xmake.io)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)]()

---

## Architecture Highlights

This codebase has been thoroughly refactored from its legacy roots into a **flat, modular architecture** of 16 independent libraries orchestrated by xmake:

- **Lockless Zero-Copy IPC** — Linux kernel module (`/dev/lpl0`) with `vmalloc_user` mapped memory and `smp_*` lockless ring buffers for microsecond-latency network I/O.
- **Data-Oriented ECS** — Flat archetypes, atomic concurrent registries, cache-aligned SoA double-buffered component chunks. O(1) entity lookup via generational sparse sets.
- **DAG Task Scheduler** — System dependency graph split into PreSwap/PostSwap phases, executed over a ThreadPool with MPSC job submission.
- **Deterministic Fixed-Point Math** — `Fixed32` types with custom CORDIC trigonometry for cross-platform deterministic physics.
- **Morton Spatial Partitioning** — Z-order encoded world partitioning with broad-phase collision and conditional GPU dispatch above a configurable entity threshold.
- **Vulkan Renderer** — Integrated Vulkan pipeline (VkWrapper v0.0.4) with ImGui, SPIR-V shaders, and cube debug rendering.
- **BCI Integration** — Brain-Computer Interface module (OpenBCI Cyton, 8ch/250Hz) with real-time DSP: Schumacher $R(t)$, Riemannian $\delta_R$, Mahalanobis $D_M$.

> This is an experimental project under active development — not a distributable product.

---

## Directory Structure

```
LplPlugin/
├── core/           — Platform abstraction, types, assertions, logging
├── math/           — Vec3, Quat, Fixed32, CORDIC, Morton
├── memory/         — PinnedAllocator (CUDA zero-copy), pool/arena allocators
├── container/      — FlatAtomicHashMap, SparseSet, RingBuffer
├── concurrency/    — ThreadPool, SpinLock, atomic utilities
├── ecs/            — Entity registry, Partition (SoA double-buffered chunks)
├── physics/        — CpuPhysicsBackend, SpatialGrid (deterministic sorted)
├── net/            — UDP transport (kernel driver / socket fallback), protocol
├── gpu/            — IComputeBackend, CudaBackend, VulkanComputeBackend
├── input/          — InputManager (keys, axes, neural state per entity)
├── render/         — Vulkan pipeline (VkWrapper), ImGui, SPIR-V shaders
├── audio/          — Spatial audio (stub)
├── haptic/         — Haptic/vestibular feedback (stub)
├── bci/            — OpenBCI driver, FFT, SignalMetrics, RiemannianGeometry
├── serial/         — Serial port abstraction
├── engine/         — Top-level facade, game loop, SystemScheduler
├── kernel/         — Linux kernel module (lpl_kmod.c)
├── apps/           — Executables: lpl-server, lpl-client, lpl-benchmark
├── tests/          — Parity regression tests (lpl-test-*)
└── shaders/        — Compiled SPIR-V shaders (vert.spv, frag.spv)
```

Each module is a **static library** (`lpl-<name>`) with its own `xmake.lua`, `include/lpl/<name>/`, and `src/`.

---

## Building the Engine

The project uses **[xmake](https://xmake.io)** exclusively.

### Prerequisites

- **Linux** with kernel headers (for the kernel module)
- **xmake** >= 2.9.0
- **GCC 13+** or **Clang 17+** (C++23 support)
- **Vulkan SDK** (headers + loader)
- **glslangValidator** (compile shaders — `apt install glslang-tools`)
- NVIDIA CUDA Toolkit *(optional — automatic CPU fallback)*

### Quick Start

```bash
# Clone and build
xmake f -c
xmake build -j$(nproc)

# Run parity regression tests (must all pass before any commit)
xmake run test-fixed32-parity
xmake run test-morton-parity
xmake run test-physics-parity

# Run the engine
xmake run lpl-server         # terminal 1
xmake run lpl-client         # terminal 2

# Benchmarks
xmake run lpl-benchmark
```

### Build Options

```bash
xmake f --renderer=n         # Disable Vulkan (headless server)
xmake f --cuda=y             # Enable CUDA physics dispatch
xmake f -m debug             # Debug symbols, LPL_DEBUG
xmake f -m release           # Full optimization, NDEBUG (default)
xmake f -m profile           # Debug symbols + full optimization

# Kernel Module
xmake kmod-build             # Build lpl_kmod.ko
xmake kmod-install           # insmod lpl_kmod
xmake kmod-uninstall         # rmmod lpl_kmod
```

---

## Documentation

Full documentation is available in the **[Wiki](LplPlugin.wiki/Home.md)** — architecture deep-dives, module reference, ADRs, benchmarks, and roadmap.

## License

This project is licensed under the **GPL-3.0** — see [LICENSE](LICENSE) for details.
