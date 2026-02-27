/**
 * @file Systems.hpp
 * @brief Systèmes ECS réutilisables extraits de l'ancien Network.hpp et des applications.
 *
 * Chaque fonction retourne un SystemDescriptor prêt à être enregistré dans le scheduler.
 * Les systèmes sont regroupés par responsabilité :
 *
 *   ── PreSwap ──
 *   - NetworkReceiveSystem  : réceptionne les paquets réseau → PacketQueue
 *   - SessionSystem         : traite les connexions (serveur)
 *   - InputProcessingSystem : désérialise les InputEvents → InputManager (serveur)
 *   - WelcomeSystem         : traite MSG_WELCOME (client)
 *   - StateReconciliationSystem : applique MSG_STATE au monde (client)
 *   - MovementSystem        : calcule la vélocité depuis InputManager → entité
 *   - PhysicsSystem         : world.step(dt)
 *
 *   ── PostSwap ──
 *   - BroadcastSystem       : sérialise et broadcast l'état du monde (serveur)
 *
 * Les systèmes spécifiques au client (BCI, Camera, Render, InputSend) restent dans
 * l'application car ils dépendent de libs externes (GLFW, OpenGL, BCI).
 *
 * @author MasterLaplace
 */

#pragma once

#include "SystemScheduler.hpp"
#include "WorldPartition.hpp"
#include "Network.hpp"
#include "PacketQueue.hpp"
#include "InputManager.hpp"
#include "SessionManager.hpp"

namespace Systems {

// ═══════════════════════════════════════════════════════════════
//  PreSwap Systems
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Système de réception réseau.
 *
 * Consomme les paquets en attente (socket ou ring buffer) et les
 * désérialise dans les queues typées de PacketQueue.
 *
 * Utilisé côté serveur ET client.
 */
inline SystemDescriptor NetworkReceiveSystem(Network &network, PacketQueue &queue)
{
    return {
        "NetworkReceive",
        -20,
        [&network, &queue](WorldPartition &/*w*/, float /*dt*/) {
            network.network_consume_packets(queue);
        },
        {}, // Pas d'accès composant direct (écrit dans les queues, pas dans le monde)
        SchedulePhase::PreSwap
    };
}

/**
 * @brief Système de gestion de session (serveur).
 *
 * Traite les ConnectEvent : crée l'entité joueur, enregistre le client,
 * envoie MSG_WELCOME.
 */
inline SystemDescriptor SessionSystem(SessionManager &session, PacketQueue &queue,
                                      Network &network, InputManager &inputManager)
{
    return {
        "Session",
        -15,
        [&session, &queue, &network, &inputManager](WorldPartition &w, float /*dt*/) {
            session.handleConnections(queue, w, network, inputManager);
        },
        {
            {ComponentId::Position, AccessMode::Write},
            {ComponentId::Velocity, AccessMode::Write},
            {ComponentId::Health,   AccessMode::Write},
            {ComponentId::Mass,     AccessMode::Write},
            {ComponentId::Size,     AccessMode::Write},
        },
        SchedulePhase::PreSwap
    };
}

/**
 * @brief Système de traitement des inputs réseau (serveur).
 *
 * Consomme les InputEvents de la queue et met à jour l'InputManager
 * avec l'état des touches, axes et données neurales de chaque client.
 */
inline SystemDescriptor InputProcessingSystem(PacketQueue &queue, InputManager &inputManager)
{
    return {
        "InputProcessing",
        -10,
        [&queue, &inputManager](WorldPartition &/*w*/, float /*dt*/) {
            auto events = queue.inputs.drain();
            for (const auto &ev : events)
            {
                for (const auto &k : ev.keys)
                    inputManager.setKeyState(ev.entityId, k.key, k.pressed);

                for (const auto &a : ev.axes)
                    inputManager.setAxis(ev.entityId, a.axisId, a.value);

                if (ev.hasNeural)
                {
                    inputManager.setNeural(ev.entityId,
                        ev.neural.alpha, ev.neural.beta,
                        ev.neural.concentration, ev.neural.blink);
                }
            }
        },
        {},
        SchedulePhase::PreSwap
    };
}

/**
 * @brief Système de mouvement (serveur ET client prediction).
 *
 * Pour chaque entité dans l'InputManager, met à jour l'état grounded,
 * puis calcule la vélocité depuis les inputs WASD + modulation neurale et l'applique.
 */
inline SystemDescriptor MovementSystem(InputManager &inputManager)
{
    return {
        "Movement",
        -5,
        [&inputManager](WorldPartition &w, float /*dt*/) {
            uint32_t writeIdx = w.getWriteIdx();

            // Iterate over all entities that have input state
            // We need to find them in the world and update velocity
            w.forEachChunk([&](Partition &p) {
                for (size_t i = 0; i < p.getEntityCount(); ++i)
                {
                    uint32_t entId = p.getEntityId(i);
                    if (!inputManager.hasEntity(entId))
                        continue;

                    Vec3 currentVel = p.getEntity(i, writeIdx).velocity;

                    // Mettre à jour le grounded state basé sur la vélocité Y actuelle
                    // L'entité est grounded si elle se déplace lentement verticalement
                    inputManager.updateGroundedState(entId, currentVel.y, 0.5f);

                    Vec3 newVel = inputManager.computeMovementVelocity(entId, currentVel);

                    p.setVelocity(static_cast<uint32_t>(i), newVel, writeIdx);
                    p.wakeEntity(static_cast<uint32_t>(i));
                }
            });
        },
        {
            {ComponentId::Velocity, AccessMode::Write},
        },
        SchedulePhase::PreSwap
    };
}

/**
 * @brief Système de physique.
 *
 * Utilisé côté serveur ET client.
 */
inline SystemDescriptor PhysicsSystem()
{
    return {
        "Physics",
        0,
        [](WorldPartition &w, float dt) {
            w.step(dt);
        },
        {
            {ComponentId::Position, AccessMode::Write},
            {ComponentId::Velocity, AccessMode::Write},
            {ComponentId::Forces,   AccessMode::Write},
            {ComponentId::Mass,     AccessMode::Read},
        },
        SchedulePhase::PreSwap
    };
}

// ─── Client-only PreSwap Systems ─────────────────────────────

/**
 * @brief Système de bienvenue (client).
 *
 * Consomme les WelcomeEvents et met à jour l'état de connexion.
 */
inline SystemDescriptor WelcomeSystem(PacketQueue &queue, uint32_t &myEntityId, bool &connected)
{
    return {
        "Welcome",
        -18,
        [&queue, &myEntityId, &connected](WorldPartition &/*w*/, float /*dt*/) {
            auto events = queue.welcomes.drain();
            for (const auto &ev : events)
            {
                myEntityId = ev.entityId;
                connected = true;
                printf("[CLIENT] Connected! Entity ID: %u\n", ev.entityId);
            }
        },
        {},
        SchedulePhase::PreSwap
    };
}

/**
 * @brief Système de réconciliation d'état (client).
 *
 * Consomme les StateUpdateEvents et met à jour le monde local.
 * Crée les entités manquantes (entités distantes non encore spawnées).
 */
inline SystemDescriptor StateReconciliationSystem(PacketQueue &queue)
{
    return {
        "StateReconciliation",
        -15,
        [&queue](WorldPartition &w, float /*dt*/) {
            auto events = queue.states.drain();
            uint32_t writeIdx = w.getWriteIdx();

            for (const auto &ev : events)
            {
                for (const auto &ent : ev.entities)
                {
                    int localIdx = -1;
                    Partition *chunk = w.findEntity(ent.id, localIdx);

                    if (chunk && localIdx >= 0)
                    {
                        chunk->setPosition(static_cast<uint32_t>(localIdx), ent.pos, writeIdx);
                        chunk->setSize(static_cast<uint32_t>(localIdx), ent.size);
                        chunk->setHealth(static_cast<uint32_t>(localIdx), ent.hp);
                    }
                    else
                    {
                        Partition::EntitySnapshot snap{};
                        snap.id = ent.id;
                        snap.position = ent.pos;
                        snap.size = ent.size;
                        snap.health = ent.hp;
                        snap.mass = 1.0f;
                        snap.rotation = {0, 0, 0, 1};
                        w.addEntity(snap);
                    }
                }
            }
        },
        {
            {ComponentId::Position, AccessMode::Write},
            {ComponentId::Size,     AccessMode::Write},
            {ComponentId::Health,   AccessMode::Write},
        },
        SchedulePhase::PreSwap
    };
}

// ═══════════════════════════════════════════════════════════════
//  PostSwap Systems
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Système de broadcast (serveur, PostSwap).
 *
 * Sérialise l'état du monde (read buffer) et l'envoie à tous les clients.
 * Doit être exécuté après swapBuffers() pour lire le snapshot stable.
 */
inline SystemDescriptor BroadcastSystem(SessionManager &session, Network &network)
{
    return {
        "Broadcast",
        0,
        [&session, &network](WorldPartition &w, float /*dt*/) {
            session.broadcast_state(w, network);
        },
        {
            {ComponentId::Position, AccessMode::Read},
            {ComponentId::Size,     AccessMode::Read},
            {ComponentId::Health,   AccessMode::Read},
        },
        SchedulePhase::PostSwap
    };
}

} // namespace Systems
