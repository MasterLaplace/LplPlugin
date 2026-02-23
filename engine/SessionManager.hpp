/**
 * @file SessionManager.hpp
 * @brief Gestion des sessions clients côté serveur.
 *
 * Stocke la liste des clients connectés (entityId, ip, port).
 * Gère les connexions (MSG_CONNECT → création d'entité + MSG_WELCOME).
 * Fournit broadcast_state() pour sérialiser et envoyer l'état du monde.
 *
 * Extrait de l'ancien Network.hpp (handle_connect, broadcast_state, _clients).
 *
 * @author MasterLaplace
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>
#include "Network.hpp"
#include "PacketQueue.hpp"
#include "WorldPartition.hpp"
#include "InputManager.hpp"

/**
 * @brief Endpoint d'un client connecté au serveur.
 */
struct ClientEndpoint {
    uint32_t entityId;
    uint32_t ip;
    uint16_t port;
};

/**
 * @brief Gestionnaire de sessions serveur.
 *
 * Responsabilités :
 *   - Traiter les ConnectEvent : créer l'entité, envoyer MSG_WELCOME, enregistrer le client
 *   - Stocker la liste des clients connectés
 *   - Broadcast de l'état du monde à tous les clients (PostSwap)
 */
class SessionManager {
public:
    /**
     * @brief Traite les événements de connexion en attente.
     *
     * Pour chaque ConnectEvent :
     *   1. Vérifie que le client n'est pas déjà connecté
     *   2. Crée une entité joueur dans le monde
     *   3. Crée un InputState dans l'InputManager
     *   4. Envoie MSG_WELCOME au client
     *   5. Enregistre le ClientEndpoint
     */
    void handleConnections(PacketQueue &queue, WorldPartition &world,
                           Network &network, InputManager &inputManager)
    {
        auto events = queue.connects.drain();

        for (const auto &ev : events)
        {
            // Skip duplicate connections
            bool duplicate = false;
            for (const auto &c : _clients)
            {
                if (c.ip == ev.srcIp && c.port == ev.srcPort)
                {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate)
                continue;

            uint32_t newId = _nextEntityId++;

            // Create player entity
            Partition::EntitySnapshot player{};
            player.id = newId;
            player.position = {0.f, 10.f, 0.f};
            player.rotation = {0.f, 0.f, 0.f, 1.f};
            player.velocity = {0.f, 0.f, 0.f};
            player.mass = 1.f;
            player.force = {0.f, 0.f, 0.f};
            player.size = {1.f, 2.f, 1.f};
            player.health = 100;
            world.addEntity(player);

            // Create input state for this player
            inputManager.getOrCreate(newId);

            // Register client endpoint
            _clients.push_back({newId, ev.srcIp, ev.srcPort});

            // Send MSG_WELCOME
            network.send_welcome(ev.srcIp, ev.srcPort, newId);

            printf("[SESSION] Client connected: %u -> Entity %u\n", ev.srcIp, newId);
        }
    }

    /**
     * @brief Sérialise et envoie l'état du monde à tous les clients connectés.
     *
     * Lit depuis le read buffer (doit être appelé après swapBuffers — PostSwap).
     * Protocol: [MSG_STATE (1)][Count (2)][EntityData...]
     * EntityData: [ID (4)][Pos (12)][Size (12)][HP (4)] = 32 bytes
     */
    void broadcast_state(WorldPartition &world, Network &network)
    {
        if (_clients.empty()) return;

        uint32_t readIdx = world.getReadIdx();
        constexpr size_t MAX_UDP = MAX_PACKET_SIZE;
        uint8_t pkt[MAX_UDP];

        constexpr size_t HEADER_SIZE = 3u;
        constexpr size_t ENTITY_SIZE = 32u;
        constexpr size_t MAX_ENTITIES_PER_PACKET = (MAX_UDP - HEADER_SIZE) / ENTITY_SIZE;

        uint16_t count = 0u;
        uint8_t *cursor = pkt + HEADER_SIZE;

        auto flush_packet = [&](bool force = false) {
            if (count == 0u && !force) return;

            pkt[0] = MSG_STATE;
            *reinterpret_cast<uint16_t*>(pkt + 1) = count;
            uint16_t len = static_cast<uint16_t>(cursor - pkt);

            for (const auto &client : _clients)
                network.send_packet(client.ip, client.port, len, pkt);

            count = 0u;
            cursor = pkt + HEADER_SIZE;
        };

        world.forEachChunk([&](Partition &p) {
            for (size_t i = 0u; i < p.getEntityCount(); ++i)
            {
                if (count >= MAX_ENTITIES_PER_PACKET)
                    flush_packet();

                auto ent = p.getEntity(i, readIdx);
                uint32_t entId = p.getEntityId(i);

                *reinterpret_cast<uint32_t*>(cursor) = entId; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.position.x; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.position.y; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.position.z; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.size.x; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.size.y; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.size.z; cursor += 4;
                *reinterpret_cast<int32_t*>(cursor) = ent.health; cursor += 4;

                count++;
            }
        });

        if (count > 0u)
            flush_packet();
    }

    // ─── Queries ─────────────────────────────────────────────

    [[nodiscard]] size_t getClientCount() const noexcept { return _clients.size(); }
    [[nodiscard]] const std::vector<ClientEndpoint> &getClients() const noexcept { return _clients; }

private:
    std::vector<ClientEndpoint> _clients;
    uint32_t _nextEntityId = 100u; ///< IDs 0-99 réservés (NPCs, etc.)
};
