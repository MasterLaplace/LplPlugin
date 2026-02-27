/**
 * @file Session.hpp
 * @brief Represents a single client session on the server.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_SESSION_SESSION_HPP
    #define LPL_NET_SESSION_SESSION_HPP

#include <lpl/ecs/Entity.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <chrono>

namespace lpl::net::session {

/**
 * @enum SessionState
 * @brief Lifecycle states of a client session.
 */
enum class SessionState : core::u8
{
    Connecting,
    Connected,
    Disconnecting,
    Disconnected
};

/**
 * @class Session
 * @brief Per-client state including entity binding, RTT, and sequence
 *        tracking.
 */
class Session final : public core::NonCopyable<Session>
{
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /**
     * @brief Constructs a session for the given player ID.
     * @param playerId Unique player identifier.
     */
    explicit Session(core::u32 playerId);
    ~Session();

    /** @brief Returns the player ID. */
    [[nodiscard]] core::u32 playerId() const noexcept;

    /** @brief Returns the current session state. */
    [[nodiscard]] SessionState state() const noexcept;

    /** @brief Sets the session state. */
    void setState(SessionState newState) noexcept;

    /** @brief Binds an entity to this session (the player's avatar). */
    void bindEntity(ecs::EntityId entity) noexcept;

    /** @brief Returns the bound entity (may be null). */
    [[nodiscard]] ecs::EntityId boundEntity() const noexcept;

    /** @brief Records a ping measurement. */
    void recordRtt(core::f32 rttMs) noexcept;

    /** @brief Returns the smoothed round-trip time (ms). */
    [[nodiscard]] core::f32 smoothedRtt() const noexcept;

    /** @brief Returns the last input sequence received from this client. */
    [[nodiscard]] core::u32 lastInputSequence() const noexcept;

    /** @brief Updates the last input sequence. */
    void setLastInputSequence(core::u32 seq) noexcept;

    /** @brief Returns time of last packet received. */
    [[nodiscard]] TimePoint lastActivity() const noexcept;

    /** @brief Marks activity (heartbeat). */
    void touch() noexcept;

private:
    core::u32     _playerId;
    SessionState  _state{SessionState::Connecting};
    ecs::EntityId _entity{};
    core::f32     _smoothedRtt{0.0f};
    core::u32     _lastInputSeq{0};
    TimePoint     _lastActivity;
};

} // namespace lpl::net::session

#endif // LPL_NET_SESSION_SESSION_HPP
