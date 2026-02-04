# Projet Laplace : Roadmap & Objectifs

## 1. Vision & Objectifs
*   **But Ultime** : Créer un **Sword Art Online (SAO)** réel (FullDive VR).
*   **Ambition** : Viser le **Prix Turing** via une révolution technique dans la simulation massive et la latence.
*   **Philosophie** : Performance absolue, "Close to Metal", Convergence Silicium/Cerveau.

## 2. Stack Technique
*   **Langage** : **C** (Puriste, Kernel, Drivers).
*   **Architecture** : **ECS (Entity Component System)** avec mémoire **SoA (Structure of Arrays)**.
*   **Compute** : Physique et Collisions déportées sur **GPU** (CUDA/Vulkan).
*   **Réseau** : UDP Fiable, **Zero-Copy**, Ring Buffers, RDMA (Remote Direct Memory Access).

## 3. État d'Avancement (Current State)
### Modules Implémentés
*   **Core Engine (`plugin.c`)** :
    *   Gestionnaire ECS bas niveau (Sparse Sets, Generations).
    *   **Ring Buffer** atomique pour l'ingestion réseau sans allocation.
    *   **Double Buffering** pour la synchronisation CPU/GPU sans Mutex bloquants.
*   **Simulation Loop (`main.c`)** : Threading réseau vs rendu, stress tests.

## 4. Roadmap Technique
### Phase 1 : Fondations (Actuel)
- [x] Preuve de concept ECS + Ring Buffer.
- [ ] Kernel CUDA basique.
- [ ] Optimisation du transfert CPU -> GPU (Pinned Memory / Unified Memory).
- [ ] Gestion des "Race Conditions" entre écriture réseau et lecture GPU.

### Phase 2 : Kernel & Hardware
- [ ] Développement d'un module Kernel (LKM) pour l'injection directe des paquets dans le Ring Buffer.
- [ ] Implémentation du **GPUDirect** (NIC -> GPU sans passer par RAM CPU).
- [ ] Gestionnaire de mémoire custom (Slab Allocator).

### Phase 3 : Simulation Massive
- [ ] Physique distribuée (Serveur autoritaire vs Prédiction client).
- [ ] Spatial Partitioning sur GPU (BVH / Octree).
- [ ] Gestion de 100 000+ entités dynamiques.

### Phase 4 : Immersion & BCI
- [ ] Moteur de rendu NeRF (Neural Radiance Fields).
- [ ] Audio Spatial (Propagation physique du son).
- [ ] Interface Cerveau-Machine (Lecture signaux EEG/EMG).
