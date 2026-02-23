/**
 * @file PacketQueue.hpp
 * @brief Queues typées pour le dispatch de paquets réseau désérialisés.
 *
 * Network écrit les événements désérialisés dans ces queues.
 * Les systèmes ECS les consomment via drain() chaque frame.
 *
 * Architecture découplée : Network ne connaît plus WorldPartition ni InputManager.
 * Le routage se fait via les systèmes qui consomment les events.
 *
 * @author MasterLaplace
 */

#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include "Math.hpp"

// ─── Event Types ─────────────────────────────────────────────

/**
 * @brief Événement de connexion client (MSG_CONNECT côté serveur).
 */
struct ConnectEvent {
    uint32_t srcIp;
    uint16_t srcPort;
};

/**
 * @brief Événement de bienvenue (MSG_WELCOME côté client).
 */
struct WelcomeEvent {
    uint32_t entityId;
};

/**
 * @brief Données d'une entité dans un paquet MSG_STATE.
 */
struct StateEntityData {
    uint32_t id;
    Vec3 pos;
    Vec3 size;
    int32_t hp;
};

/**
 * @brief Événement de mise à jour d'état (MSG_STATE côté client).
 */
struct StateUpdateEvent {
    std::vector<StateEntityData> entities;
};

/**
 * @brief Données d'input d'une touche.
 */
struct KeyInput {
    uint16_t key;
    bool pressed;
};

/**
 * @brief Données d'input d'un axe.
 */
struct AxisInput {
    uint8_t axisId;
    float value;
};

/**
 * @brief Données neurales d'un input.
 */
struct NeuralInput {
    float alpha;
    float beta;
    float concentration;
    bool blink;
};

/**
 * @brief Événement d'inputs (MSG_INPUTS côté serveur).
 */
struct InputEvent {
    uint32_t entityId;
    std::vector<KeyInput> keys;
    std::vector<AxisInput> axes;
    NeuralInput neural;
    bool hasNeural = false;
};

// ─── Typed Queue ─────────────────────────────────────────────

/**
 * @brief Queue thread-safe pour un type d'événement donné.
 *
 * Producteur unique (Network::network_consume_packets) — consommateur unique (système ECS).
 * Un simple mutex suffit (faible contention, une frame par cycle).
 */
template <typename T>
class TypedQueue {
public:
    void push(T &&event)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push_back(std::move(event));
    }

    void push(const T &event)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push_back(event);
    }

    /**
     * @brief Vide la queue et retourne tous les événements.
     * Appelé une fois par frame par le système consommateur.
     */
    [[nodiscard]] std::vector<T> drain()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<T> result;
        result.swap(_queue);
        return result;
    }

    /**
     * @brief Nombre d'événements en attente.
     */
    [[nodiscard]] size_t size() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

    [[nodiscard]] bool empty() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }

private:
    std::vector<T> _queue;
    mutable std::mutex _mutex;
};

// ─── Packet Queue ────────────────────────────────────────────

/**
 * @brief Agrégateur de toutes les queues d'événements réseau.
 *
 * Network y écrit après désérialisation.
 * Les systèmes ECS consomment via les drain() individuels.
 *
 * Usage :
 * @code
 *   PacketQueue pq;
 *
 *   // Producteur (Network)
 *   pq.connects.push({srcIp, srcPort});
 *
 *   // Consommateur (SessionSystem)
 *   auto events = pq.connects.drain();
 *   for (auto &ev : events) handleConnect(ev);
 * @endcode
 */
struct PacketQueue {
    TypedQueue<ConnectEvent>     connects;   ///< MSG_CONNECT (serveur)
    TypedQueue<WelcomeEvent>     welcomes;   ///< MSG_WELCOME (client)
    TypedQueue<StateUpdateEvent> states;     ///< MSG_STATE   (client)
    TypedQueue<InputEvent>       inputs;     ///< MSG_INPUTS  (serveur)
};
