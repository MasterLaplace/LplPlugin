# shared — Shared Headers

Common C/C++ headers for all plugins (binary protocol, mathematics).

## Contents

```
shared/
├── lpl_protocol.h   — Binary protocol kernel↔userspace (pure C)
└── Math.hpp         — Vec3, Quat, BoundaryBox (__host__ __device__)
```

## `lpl_protocol.h`

Pure C header (`#pragma once` + C99 types) used in both the kernel module (`kernel/`) and userspace plugins.

| Structure | Role |
|-----------|------|
| `RingHeader` | Atomic head/tail of the shared ring buffer |
| `RxPacket` | Incoming packet received from network |
| `TxPacket` | Outgoing packet to send |
| `LplSharedMemory` | Complete layout of shared memory (mmap) |

Useful macros: `smp_load_acquire`, `smp_store_release` (portable memory barriers).

## `Math.hpp`

Mathematical primitives compatible with CPU and CUDA GPU (`__host__ __device__`).

| Type | Operations |
|------|-----------|
| `Vec3` | `+`, `-`, `*`, `dot`, `cross`, `normalize`, `length` |
| `Quat` | Multiplication, normalization, vector rotation |
| `BoundaryBox` | AABB — `contains(Vec3)`, `intersects(BoundaryBox)` |

## Include Rule

All plugins add `-I../../shared` (or `-I../shared` for `bci/`) to their compilation flags.  
No include should use relative paths to `shared/` — this ensures header portability.
