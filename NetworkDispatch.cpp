// --- LAPLACE NETWORK DISPATCH --- //
// File: NetworkDispatch.cpp
// Description: Routage des paquets réseau vers le WorldPartition + gestion driver
// Auteur: MasterLaplace

#include "NetworkDispatch.hpp"
#include "WorldPartition.hpp"
#include "Math.hpp"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// ─── Driver file descriptor (module-scoped) ──────────────────

static int g_driverFd = -1;

// ─── Driver Lifecycle ─────────────────────────────────────────

NetworkRingBuffer *network_init()
{
    g_driverFd = open("/dev/lpl_driver", O_RDWR);
    if (g_driverFd < 0)
    {
        perror("[ERROR] network_init: Impossible d'ouvrir /dev/lpl_driver");
        return nullptr;
    }

    auto *ring = static_cast<NetworkRingBuffer *>(
        mmap(nullptr, sizeof(NetworkRingBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, g_driverFd, 0));

    if (ring == MAP_FAILED)
    {
        perror("[ERROR] network_init: Erreur de mappage mémoire");
        close(g_driverFd);
        g_driverFd = -1;
        return nullptr;
    }

    printf("[NET] Driver connecté. Ring Buffer mappé à %p (%zu bytes)\n", ring, sizeof(NetworkRingBuffer));
    printf("[NET] Initial head=%u, tail=%u\n",
           lpl_atomic_load(&ring->head), lpl_atomic_load(&ring->tail));
    return ring;
}

void network_cleanup(NetworkRingBuffer *ring)
{
    if (ring && ring != MAP_FAILED)
        munmap(ring, sizeof(NetworkRingBuffer));

    if (g_driverFd >= 0)
    {
        close(g_driverFd);
        g_driverFd = -1;
    }
}

// ─── Packet Dispatch ──────────────────────────────────────────

/**
 * @brief Dispatch un paquet réseau vers le WorldPartition.
 *
 * Les écritures hot (position, velocity) ciblent le write buffer.
 * Les écritures cold (health, mass) ciblent le buffer unique.
 * Si l'entité est inconnue, addEntity l'insère dans les deux buffers.
 */
static void dispatch_packet_world(DynamicPacket *pkt, WorldPartition &world)
{
    if (pkt->msgType != RING_MSG_DYNAMIC)
    {
        printf("[WARNING] dispatch_packet_world: unknown msgType(%u)\n", pkt->msgType);
        return;
    }

    uint8_t *cursor = pkt->data;
    uint32_t public_id = *(uint32_t *)cursor;
    cursor += sizeof(uint32_t);

    uint32_t writeIdx = world.getWriteIdx();
    int localIdx = -1;
    Partition *chunk = world.findEntity(public_id, localIdx);

    if (chunk && localIdx >= 0)
    {
        // Entité existante → mise à jour des composants dans le write buffer
        while (cursor < pkt->data + pkt->size)
        {
            uint8_t comp = *cursor++;
            switch (comp)
            {
            case COMP_TRANSFORM: {
                Vec3 pos{*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                chunk->setPosition(static_cast<uint32_t>(localIdx), pos, writeIdx);
                break;
            }
            case COMP_HEALTH: {
                int32_t hp = *(int32_t *)(cursor);
                cursor += 4;
                chunk->setHealth(static_cast<uint32_t>(localIdx), hp);
                break;
            }
            case COMP_VELOCITY: {
                Vec3 vel{*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                chunk->setVelocity(static_cast<uint32_t>(localIdx), vel, writeIdx);
                break;
            }
            case COMP_MASS: {
                float m = *(float *)(cursor);
                cursor += 4;
                chunk->setMass(static_cast<uint32_t>(localIdx), m);
                break;
            }
            case COMP_SIZE: {
                Vec3 s{*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                chunk->setSize(static_cast<uint32_t>(localIdx), s);
                break;
            }
            default:
                printf("[WARNING] dispatch_packet_world: unknown component_id(%d)\n", comp);
                return;
            }
        }
    }
    else
    {
        // Nouvelle entité → addEntity écrit dans les deux buffers automatiquement
        Partition::EntitySnapshot snap{};
        snap.id = public_id;
        snap.rotation = {0.f, 0.f, 0.f, 1.f};
        snap.mass = 1.0f;
        snap.size = {1.f, 1.f, 1.f};
        snap.health = 100;

        while (cursor < pkt->data + pkt->size)
        {
            uint8_t comp = *cursor++;
            switch (comp)
            {
            case COMP_TRANSFORM:
                snap.position = {*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                break;
            case COMP_HEALTH:
                snap.health = *(int32_t *)(cursor);
                cursor += 4;
                break;
            case COMP_VELOCITY:
                snap.velocity = {*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                break;
            case COMP_MASS:
                snap.mass = *(float *)(cursor);
                cursor += 4;
                break;
            case COMP_SIZE:
                snap.size = {*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                break;
            default:
                printf("[WARNING] dispatch_packet_world: unknown component_id(%d) for new entity %u\n", comp, public_id);
                return;
            }
        }

        world.addEntity(snap);
    }
}

// ─── Public API ───────────────────────────────────────────────

void network_consume_packets(NetworkRingBuffer *ring, WorldPartition &world)
{
    uint32_t tail = lpl_atomic_load(&ring->tail);
    uint32_t head = lpl_atomic_load(&ring->head);

    while (tail != head)
    {
        dispatch_packet_world(&ring->packets[tail], world);
        tail = (tail + 1u) & (RING_SIZE - 1u);
    }

    lpl_atomic_store(&ring->tail, tail);
}
