# kernel — Module noyau Linux (lpl_kmod)

Module noyau Linux implémentant un pipeline **zéro-copie** pour l'ingestion de paquets UDP avec partage mémoire vers l'espace utilisateur.

## Contenu

```
kernel/
├── lpl_kmod.c   — Source du module (hook Netfilter + char device + mmap)
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

## Fonctionnement

| Mécanisme | Détail |
|-----------|--------|
| **Hook Netfilter** | `NF_INET_PRE_ROUTING` — capture les paquets avant le routage noyau |
| **Ring Buffer** | Structure `RxRingBuffer` / `TxRingBuffer` partagée via `mmap` |
| **Char Device** | `/dev/lpl` exposé avec `open`, `mmap`, `ioctl` |
| **Thread TX** | Thread noyau dédié à l'envoi des paquets de réponse |
| **Zéro-copie** | Aucun `copy_to_user` — la mémoire est directement mappée |

## Build

```bash
# Prérequis : linux-headers-$(uname -r)
make -C kernel   # ou `make driver` depuis la racine
```

Produit `lpl_kmod.ko` dans `/tmp/lpl_kernel_build/`.

```bash
make install     # insmod + chown /dev/lpl
make uninstall   # rmmod
```

## Dépendance partagée

Le module utilise `lpl_protocol.h` (depuis `shared/`) pour les définitions des structures `RingHeader`, `RxPacket`, `TxPacket`, et les constantes du protocole binaire.

Le Makefile copie automatiquement `../shared/lpl_protocol.h` dans le répertoire de build temporaire avant la compilation noyau.
