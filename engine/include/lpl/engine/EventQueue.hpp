/**
 * @file EventQueue.hpp
 * @brief Typed event queues for deserialized network packets.
 *
 * Mirrors the legacy PacketQueue typed-queue design. Each queue is
 * thread-safe (mutex + drain-swap) and stores deserialized events
 * for consumption by ECS systems.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_EVENTQUEUE_HPP
#    define LPL_ENGINE_EVENTQUEUE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/net/Endpoint.hpp>

#    include <array>
#    include <lpl/std/mutex.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::engine {

// ========================================================================== //
//  Event types                                                               //
// ========================================================================== //

/** @brief A client requests to connect. */
struct ConnectEvent {
    net::Endpoint source{}; ///< Address the handshake arrived from.
};

/** @brief A client asks to disconnect cleanly (or is dropped). */
struct DisconnectEvent {
    net::Endpoint source{}; ///< Address the disconnect arrived from.
};

/** @brief Server acknowledges a connection with an entity ID. */
struct WelcomeEvent {
    core::u32 entityId;
};

/** @brief A single entity's snapshot within a state update. */
struct StateEntity {
    core::u32 id;
    math::Vec3<float> pos;
    math::Vec3<float> size;
    core::i32 hp;
};

/** @brief Full state update from the server. */
struct StateUpdateEvent {
    pmr::vector<StateEntity> entities;
};

/**
 * @brief Entities that just entered the client's interest radius (AOI).
 *
 * Sent by systems::AoiBroadcastSystem when a client's radius query first sees an
 * entity. Carries the full snapshot so the client can create it locally, exactly
 * as a StateUpdateEvent would for an entity it does not yet hold.
 */
struct EntitySpawnEvent {
    pmr::vector<StateEntity> entities;
};

/**
 * @brief Position/state update for entities already known to the client (AOI).
 *
 * The delta half of area-of-interest broadcasting: entities the client already
 * holds and that remain in range get their current transform, not a re-spawn.
 */
struct StateDeltaEvent {
    pmr::vector<StateEntity> entities;
};

/**
 * @brief Entities that just left the client's interest radius (AOI).
 *
 * Only the ids travel: the client removes them from its local world. The mirror
 * image of EntitySpawnEvent.
 */
struct EntityDestroyEvent {
    pmr::vector<core::u32> ids;
};

/** @brief A single key press/release. */
struct KeyInput {
    core::u16 key;
    bool pressed;
};

/** @brief A single analog axis value. */
struct AxisInput {
    core::u8 axisId;
    float value;
};

/** @brief Neural data from a BCI headset. */
struct NeuralInput {
    float alpha;
    float beta;
    float concentration;
    bool blink;
};

/**
 * @brief A client reporting the digest it computed for one of its past ticks.
 *
 * Desync detection (§6.4): the server looks the tick up in its own digest
 * history and compares. Carries the sender so the server knows whose
 * simulation diverged.
 */
struct StateHashReportEvent {
    net::Endpoint source{}; ///< Who reported it.
    core::u64 tick{0};      ///< The tick the client hashed.
    core::u64 digest{0};    ///< What the client computed for it.
};

/** @brief Input event from a remote client. */
struct InputEvent {
    core::u32 entityId;
    pmr::vector<KeyInput> keys;
    pmr::vector<AxisInput> axes;
    bool hasNeural;
    NeuralInput neural;
};

// ========================================================================== //
//  TypedQueue<T>                                                             //
// ========================================================================== //

/**
 * @class TypedQueue
 * @brief Thread-safe queue with O(1) amortised drain via swap.
 *
 * Writers push events one at a time (protected by mutex).
 * Readers call @ref drain() to atomically swap and retrieve all pending
 * events with minimal lock contention.
 *
 * @tparam T Event type (must be movable).
 */
template <typename T> class TypedQueue {
public:
    /** @brief Pushes a single event into the queue. */
    void push(T event)
    {
        pmr::lock_guard<pmr::mutex> lock{mutex_};
        queue_.push_back(std::move(event));
    }

    /**
     * @brief Atomically drains all pending events.
     * @return Vector of events (empty if none pending).
     */
    [[nodiscard]] pmr::vector<T> drain()
    {
        pmr::lock_guard<pmr::mutex> lock{mutex_};
        pmr::vector<T> result;
        result.swap(queue_);
        return result;
    }

    /** @brief Returns true if no events are pending. */
    [[nodiscard]] bool empty() const
    {
        pmr::lock_guard<pmr::mutex> lock{mutex_};
        return queue_.empty();
    }

private:
    mutable pmr::mutex mutex_;
    pmr::vector<T> queue_;
};

// ========================================================================== //
//  EventQueues aggregate                                                     //
// ========================================================================== //

/**
 * @struct EventQueues
 * @brief Aggregate of all typed event queues used by the engine.
 *
 * The NetworkReceiveSystem fills these queues from raw transport data.
 * Other systems (Session, InputProcessing, Welcome, StateReconciliation)
 * drain the relevant queue each tick.
 */
struct EventQueues {
    TypedQueue<ConnectEvent> connects;
    TypedQueue<DisconnectEvent> disconnects;
    TypedQueue<WelcomeEvent> welcomes;
    TypedQueue<StateUpdateEvent> states;
    TypedQueue<EntitySpawnEvent> spawns;
    TypedQueue<StateDeltaEvent> deltas;
    TypedQueue<EntityDestroyEvent> destroys;
    TypedQueue<InputEvent> inputs;
    TypedQueue<StateHashReportEvent> stateHashReports;
};

} // namespace lpl::engine

#endif // LPL_ENGINE_EVENTQUEUE_HPP
