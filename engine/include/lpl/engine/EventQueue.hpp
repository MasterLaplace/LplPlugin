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

#    include <array>
#    include <mutex>
#    include <vector>

namespace lpl::engine {

// ========================================================================== //
//  Event types                                                               //
// ========================================================================== //

/** @brief Maximum size for a raw network address (covers sockaddr_storage). */
static constexpr core::u32 kMaxAddrSize = 128;

/** @brief A client requests to connect. */
struct ConnectEvent {
    core::u32 srcIp;
    core::u16 srcPort;
    std::array<core::byte, kMaxAddrSize> rawAddr{};
    core::u32 rawAddrLen{0};
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
    std::vector<StateEntity> entities;
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

/** @brief Input event from a remote client. */
struct InputEvent {
    core::u32 entityId;
    std::vector<KeyInput> keys;
    std::vector<AxisInput> axes;
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
        std::lock_guard<std::mutex> lock{mutex_};
        queue_.push_back(std::move(event));
    }

    /**
     * @brief Atomically drains all pending events.
     * @return Vector of events (empty if none pending).
     */
    [[nodiscard]] std::vector<T> drain()
    {
        std::lock_guard<std::mutex> lock{mutex_};
        std::vector<T> result;
        result.swap(queue_);
        return result;
    }

    /** @brief Returns true if no events are pending. */
    [[nodiscard]] bool empty() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::vector<T> queue_;
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
    TypedQueue<WelcomeEvent> welcomes;
    TypedQueue<StateUpdateEvent> states;
    TypedQueue<InputEvent> inputs;
};

} // namespace lpl::engine

#endif // LPL_ENGINE_EVENTQUEUE_HPP
