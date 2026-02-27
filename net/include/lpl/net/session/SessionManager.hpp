/**
 * @file SessionManager.hpp
 * @brief Manages all active client sessions on the server.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_SESSION_SESSIONMANAGER_HPP
    #define LPL_NET_SESSION_SESSIONMANAGER_HPP

#include <lpl/net/session/Session.hpp>
#include <lpl/net/transport/ITransport.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>
#include <span>
#include <vector>

namespace lpl::net::session {

/**
 * @class SessionManager
 * @brief Active session registry with timeout-based disconnection.
 *
 * Uses Read-Copy-Update (RCU) semantics for concurrent read access during
 * tick while mutations (connect/disconnect) happen on the main thread.
 */
class SessionManager final : public core::NonCopyable<SessionManager>
{
public:
    SessionManager();
    ~SessionManager();

    /**
     * @brief Creates a new session for a connecting client.
     * @param playerId Unique player identifier.
     * @return Reference to the new session, or error if already exists.
     */
    [[nodiscard]] core::Expected<Session*> connect(core::u32 playerId);

    /**
     * @brief Disconnects a client session.
     * @param playerId Player to disconnect.
     */
    [[nodiscard]] core::Expected<void> disconnect(core::u32 playerId);

    /**
     * @brief Finds a session by player ID.
     * @return Pointer to the session or nullptr.
     */
    [[nodiscard]] Session* find(core::u32 playerId) noexcept;

    /** @brief Finds a session by player ID (const). */
    [[nodiscard]] const Session* find(core::u32 playerId) const noexcept;

    /**
     * @brief Iterates all active sessions.
     * @param callback Called for each active session.
     */
    void forEach(const std::function<void(Session&)>& callback);

    /**
     * @brief Cleans up sessions that have been inactive for too long.
     * @param timeoutMs Inactivity timeout in milliseconds.
     * @return Number of sessions reaped.
     */
    core::u32 reapTimedOut(core::f64 timeoutMs);

    /**
     * @brief Broadcasts state data to all active sessions with UDP fragmentation.
     *
     * Splits the data into packets of at most @c kMaxPayloadSize bytes each
     * (legacy MAX_ENTITIES_PER_PACKET logic), and sends each fragment with
     * the PacketFlag::Fragment bit set.
     *
     * @param transport Transport to send through.
     * @param data      Raw state data to broadcast.
     */
    void broadcastState(transport::ITransport& transport,
                        std::span<const core::byte> data);

    /**
     * @brief Checks if a connection from the same endpoint already exists.
     * @param playerId  Player to check.
     * @return True if already connected (duplicate).
     */
    [[nodiscard]] bool isDuplicate(core::u32 playerId) const noexcept;

    /** @brief Returns the number of active sessions. */
    [[nodiscard]] core::u32 activeCount() const noexcept;

    /** @brief Maximum payload size per UDP packet (minus header). */
    static constexpr core::u32 kMaxPayloadSize = 1400;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net::session

#endif // LPL_NET_SESSION_SESSIONMANAGER_HPP
