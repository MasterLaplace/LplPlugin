# engine — Moteur ECS temps-réel (header-only)

Moteur de simulation hautes performances basé sur un **ECS/SoA** avec partitionnement spatial Morton, ordonnancement DAG et physique hybride CPU/GPU (CUDA).

## Contenu

```
engine/
├── WorldPartition.hpp     — Monde partitionné (Morton chunks, double buffer)
├── Partition.hpp          — Stockage SoA par chunk (ECS)
├── EntityRegistry.hpp     — Sparse set générationnel O(1)
├── SystemScheduler.hpp    — Ordonnanceur DAG avec parallélisme automatique
├── ThreadPool.hpp         — Thread pool custom (enqueueDetached, latch)
├── FlatAtomicsHashMap.hpp — Hash map lock-free (22-bit slots)
├── FlatDynamicOctree.hpp  — Octree dynamique pour requêtes spatiales
├── Network.hpp            — Interface kernel module + dispatch de paquets
├── PhysicsGPU.cu/.cuh     — Kernels CUDA (Euler semi-implicite)
├── Morton.hpp             — Encodage/décodage Morton 3D
├── SpinLock.hpp           — Spin lock léger
└── PinnedAllocator.hpp    — Allocateur mémoire paginée (CUDA pinned)
```

## Architecture ECS

```
[EntityRegistry] ──── entités ────► [Partition (SoA)]
                                          │
[SystemScheduler] ──── DAG ──────► [WorldPartition]
                                          │
                              ┌───────────┴───────────┐
                         [GPU Physics]           [CPU Physics]
                         (CUDA kernels)          (ThreadPool)
```

## Pipeline de simulation (par frame)

1. **Réseau** — `Network::poll()` consomme le ring buffer, dispatche `STATE`/`INPUT`/`CONNECT`
2. **Ordonnancement** — `SystemScheduler::run()` lance les systèmes en parallèle par stage
3. **Partitionnement** — `WorldPartition::step()` itère sur les chunks actifs
4. **Physique** — kernels CUDA (si disponible) ou `ThreadPool` en fallback
5. **Double buffer** — swap atomique de la frame rendue

## Points techniques

| Composant | Détail |
|-----------|--------|
| `Morton.hpp` | Encodage Z-order 3D 64-bit — localité cache pour la traversée spatiale |
| `FlatAtomicsHashMap` | `std::atomic<uint64_t>` par slot — insertions wait-free |
| `PinnedAllocator` | `cudaHostAlloc` → accès GPU zero-copy via `cudaHostGetDevicePointer` |
| `SystemScheduler` | Graphe de dépendances R/W → stages parallèles synchronisés par `std::latch` |

## Build

Bibliothèque **header-only** — aucune compilation séparée hormis `PhysicsGPU.cu` (CUDA optionnel).

```bash
make server     # Compile apps/server avec -I../../engine
```

CUDA est détecté automatiquement par le Makefile ; si `nvcc` est absent, la physique CPU est utilisée.
