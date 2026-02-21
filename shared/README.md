# shared — Headers partagés

Headers C/C++ communs à tous les plugins (protocole binaire, mathématiques).

## Contenu

```
shared/
├── lpl_protocol.h   — Protocole binaire kernel↔userspace (C pur)
└── Math.hpp         — Vec3, Quat, BoundaryBox (__host__ __device__)
```

## `lpl_protocol.h`

Header C pur (`#pragma once` + types C99) utilisé à la fois dans le module noyau (`kernel/`) et les plugins userspace.

| Structure | Rôle |
|-----------|------|
| `RingHeader` | Head/tail atomiques du ring buffer partagé |
| `RxPacket` | Paquet entrant reçu du réseau |
| `TxPacket` | Paquet sortant à envoyer |
| `LplSharedMemory` | Layout complet de la mémoire partagée (mmap) |

Macros utiles : `smp_load_acquire`, `smp_store_release` (barrières mémoire portables).

## `Math.hpp`

Primitives mathématiques compatibles CPU et GPU CUDA (`__host__ __device__`).

| Type | Opérations |
|------|-----------|
| `Vec3` | `+`, `-`, `*`, `dot`, `cross`, `normalize`, `length` |
| `Quat` | Multiplication, normalisation, rotation de vecteur |
| `BoundaryBox` | AABB — `contains(Vec3)`, `intersects(BoundaryBox)` |

## Règle d'inclusion

Tous les plugins ajoutent `-I../../shared` (ou `-I../shared` pour `bci/`) à leurs flags de compilation.  
Aucun include ne doit utiliser de chemin relatif vers `shared/` — cela garantit la portabilité des headers.
