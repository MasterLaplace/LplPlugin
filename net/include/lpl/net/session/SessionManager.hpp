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
#    define LPL_NET_SESSION_SESSIONMANAGER_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/net/session/Session.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>
#    include <span>
#    include <vector>

namespace lpl::net::session {

/**
 * @class SessionManager
 * @brief Active session registry with timeout-based disconnection.
 */
class SessionManager final : public core::NonCopyable<SessionManager> {
public:
    SessionManager();
    ~SessionManager();

    /**
     * @brief Creates a new session for a connecting client.
     * @param playerId Unique player identifier.
     * @return Reference to the new session, or error if already exists.
     */
    [[nodiscard]] core::Expected<Session *> connect(core::u32 playerId);

    /**
     * @brief Disconnects a client session.
     * @param playerId Player to disconnect.
     */
    [[nodiscard]] core::Expected<void> disconnect(core::u32 playerId);

    /**
     * @brief Finds a session by player ID.
     * @return Pointer to the session or nullptr.
     */
    [[nodiscard]] Session *find(core::u32 playerId) noexcept;

    /**
     * @brief Finds a session by player ID (const).
     * @return Pointer to the session or nullptr.
     */
    [[nodiscard]] const Session *find(core::u32 playerId) const noexcept;

    /**
     * @brief Iterates all active sessions.
     * @param callback Called for each active session.
     */
    void forEach(const std::function<void(Session &)> &callback);

    /**
     * @brief Cleans up sessions that have been inactive for too long.
     *
     * @param timeoutMs Inactivity timeout in milliseconds.
     * @param onReap    Called with each session just before it is erased, so the
     *                  caller can tear down what the session owns beyond the net
     *                  layer — its avatar entity, input state and spatial-index
     *                  entry — instead of leaking them. Runs before erasure so the
     *                  session (its bound entity, its address) is still valid.
     * @return Number of sessions reaped.
     */
    core::u32 reapTimedOut(core::f64 timeoutMs, const std::function<void(const Session &)> &onReap = {});

    /**
     * @brief Broadcasts one state packet to every connected session, as one datagram each.
     *
     * A packet longer than @c kMaxPayloadSize is refused and nothing is sent. Cut
     * into slices without a header each, it reached the client as a truncated
     * first packet and pieces nobody could put back; a caller with more state
     * sends several packets, each usable alone, as BroadcastSystem does. Large
     * blocks that must arrive whole belong to a reliable channel (#147).
     *
     * Every client's datagram is ready at this instant, so the whole burst is
     * handed to the transport at once instead of one packet at a time: one
     * sendmmsg per 64 datagrams (sockets) or at most one ring kick (kernel
     * module). The datagrams borrow their bytes from @p data and their addresses
     * from the sessions, both of which outlive the call. A transport that sends
     * fewer than it was handed is logged.
     *
     * @param transport Transport to send through.
     * @param data      The state packet, header included.
     * @return Nothing, or InvalidArgument when @p data does not fit in one datagram.
     */
    [[nodiscard]] core::Expected<void> broadcastState(transport::ITransport &transport,
                                                      std::span<const core::byte> data);

    /**
     * @brief Checks if a connection from the same endpoint already exists.
     * @param playerId  Player to check.
     * @return True if already connected (duplicate).
     */
    [[nodiscard]] bool isDuplicate(core::u32 playerId) const noexcept;

    /**
     * @brief Finds the session bound to @p address, if any.
     *
     * A client is identified on the wire by its endpoint, not by a player id it
     * does not choose — so this is what tells a retransmitted handshake apart
     * from a genuine new player (legacy SessionManager::handleConnections
     * rejected a duplicate by comparing ip+port for exactly this reason).
     *
     * @return The session, or nullptr if no client at @p address is connected.
     */
    [[nodiscard]] Session *findByAddress(const Endpoint &address) noexcept;

    /**
     * @brief Returns the number of active sessions.
     * @return The count of active sessions.
     */
    [[nodiscard]] core::u32 activeCount() const noexcept;

    /** @brief Maximum payload size per UDP packet (minus header). */
    static constexpr core::u32 kMaxPayloadSize = 1400;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net::session

#endif // LPL_NET_SESSION_SESSIONMANAGER_HPP
