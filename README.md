# LplPlugin

Plate-forme BCI adaptative en boucle fermée — du signal EEG brut à la simulation temps-réel, avec un pipeline noyau zéro-copie et des métriques de neuroscience computationnelle.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Plugins](#plugins)
4. [BCI Scientific Contributions](#bci-scientific-contributions)
5. [Build & Run](#build--run)
6. [Technical Details](#technical-details)
7. [Performance](#performance)
8. [Roadmap](#roadmap)
9. [License](#license)

## Overview

LplPlugin est une **plate-forme BCI adaptative** combinée à un moteur de simulation hautes performances. Elle implémente une boucle fermée complète : acquisition EEG multi-canal (OpenBCI Cyton, 8ch/250 Hz) → métriques spectrales et géométriques → adaptation du retour visuel en temps réel.

**Domaine d'application :** réentraînement moteur assisté par BCI, interfaces neuro-adaptatives, retour hàptique conditionné par l'état neural.

**Contributions scientifiques clés :**
- **Schumacher R(t)** — indicateur de tension musculaire 40-70 Hz inter-canaux
- **Distance de Riemann δ_R** — détection de changement d'état cognitif sur les matrices de covariance SPD
- **Distance de Mahalanobis D_M** — détection d'anomalie dans l'espace des caractéristiques EEG

**Infrastructure système :**
- **Zéro-copie NIC → GPU** via module noyau Netfilter + `mmap` + CUDA
- **WorldPartition ECS** avec hachage spatial Morton et hash map lock-free
- **SystemScheduler DAG** — parallélisme automatique par résolution de dépendances R/W

## Architecture

```
[OpenBCI Cyton]
  8 canaux / 250 Hz
       |
[OpenBCIDriver] ── FFT multi-canal ──► [SignalMetrics : R(t) Schumacher]
       |                               [RiemannianGeometry : δ_R, D_M    ]
       ▼                                          |
[NeuralMetrics]                                   ▼
    muscle_tension / stability / concentration    |
       |                                          |
       ▼                                          ▼
[Network Packet] → [Kernel Module (lpl_kmod)] → [Shared Ring Buffer]
                                                      ↓
                                         [Network (Userspace)]
                                                      ↓
                                         [SystemScheduler DAG]
                                                      ↓
                                   [WorldPartition (Morton Chunks)]
                                                      ↓
                                        [Physics (CUDA / CPU)]
```

## Plugins

| Plugin | Rôle | Voir |
|--------|------|------|
| [`bci/`](bci/README.md) | Driver OpenBCI, FFT, Schumacher, Riemann, Mahalanobis | [README](bci/README.md) |
| [`engine/`](engine/README.md) | ECS SoA, WorldPartition, SystemScheduler, CUDA physics | [README](engine/README.md) |
| [`kernel/`](kernel/README.md) | Module noyau Netfilter + ring buffer zéro-copie | [README](kernel/README.md) |
| [`shared/`](shared/README.md) | Protocole binaire + mathématiques CPU/GPU | [README](shared/README.md) |

## BCI Scientific Contributions

### Schumacher R(t)

$$R(t) = \frac{1}{N_{ch}} \sum_{i=1}^{N_{ch}} \int_{40}^{70} \mathrm{PSD}_i(f,t)\, df$$

Mesure la puissance dans la bande 40-70 Hz sur tous les canaux. Utilisé comme proxy de la **tension musculaire** et de la fatigue. Un R(t) élevé déclenche une pause ou une réduction du feedback.

### Distance Riemannienne δ_R

$$\delta_R(C_1, C_2) = \sqrt{\sum_i \ln^2(\lambda_i)}$$

Métrique géodésique sur la variété des matrices symétriques définies positives. **Invariante par congruence** — mème distance avant et après pré-multiplication par une matrice non singulière, ce qui la rend robuste aux artfacts de volume-conduit.

### Distance de Mahalanobis D_M

$$D_M(x_t) = \sqrt{(x_t - \mu_c)^T \Sigma_c^{-1} (x_t - \mu_c)}$$

Détection d'anomalie dans l'espace des caractéristiques. Se réduit à la distance euclidienne quand Σ_c = I.

**Références :**
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
make install    # Libs kernel module (required)
make run        # Starts engine
```

if you wish to test with a visual client, then in another terminal, run:
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

- [x] Driver OpenBCI Cyton multi-canal (8 ch)
- [x] Schumacher R(t) avec tests unitaires
- [x] Distance de Riemann (Jacobi eigenvalue, sans dépendance extérieure)
- [x] Distance de Mahalanobis
- [ ] Phase de calibration automatique (baseline 30s)
- [ ] Visual feedback conditionné par NeuralMetrics
- [ ] GPUDirect RDMA (NIC → GPU bypass)
- [ ] Stress test 100k+ entités

## License

See [LICENSE](LICENSE)
