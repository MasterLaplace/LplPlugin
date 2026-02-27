# kernel — Linux Kernel Module (lpl_kmod)

Linux kernel module implementing a **zero-copy** pipeline for UDP packet ingestion with shared memory to userspace.

## Contents

```
kernel/
├── lpl_kmod.c   — Module source (Netfilter hook + char device + mmap)
└── Makefile
```

## Architecture

```
[NIC] → [Netfilter NF_INET_PRE_ROUTING] → [Ring Buffer (shared memory)]
                                                       ↓
                                           [mmap() userspace]
                                                       ↓
                                           [Network.hpp — polling lockless]
```

## Operation

| Mechanism | Details |
|-----------|---------|
| **Netfilter Hook** | `NF_INET_PRE_ROUTING` — captures packets before kernel routing |
| **Ring Buffer** | `RxRingBuffer` / `TxRingBuffer` structure shared via `mmap` |
| **Char Device** | `/dev/lpl` exposed with `open`, `mmap`, `ioctl` |
| **TX Thread** | Dedicated kernel thread for sending response packets |
| **Zero-copy** | No `copy_to_user` — memory is directly mapped |

## Build

```bash
# Prerequisites: linux-headers-$(uname -r)
make -C kernel   # or `make driver` from root
```

Produces `lpl_kmod.ko` in `/tmp/lpl_kernel_build/`.

```bash
make install     # insmod + chown /dev/lpl
make uninstall   # rmmod
```

## Shared Dependency

The module uses `lpl_protocol.h` (from `shared/`) for structure definitions `RingHeader`, `RxPacket`, `TxPacket`, and binary protocol constants.

The Makefile automatically copies `../shared/lpl_protocol.h` to the temporary build directory before kernel compilation.

