// /////////////////////////////////////////////////////////////////////////////
/// @file SessionManager.hpp
/// @brief Manages all active client sessions on the server.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/net/session/Session.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>
#include <span>
#include <vector>

namespace lpl::net::session {

// /////////////////////////////////////////////////////////////////////////////
/// @class SessionManager
/// @brief Active session registry with timeout-based disconnection.
///
/// Uses Read-Copy-Update (RCU) semantics for concurrent read access during
/// tick while mutations (connect/disconnect) happen on the main thread.
// /////////////////////////////////////////////////////////////////////////////
class SessionManager final : public core::NonCopyable<SessionManager>
{
public:
    SessionManager();
    ~SessionManager();

    /// @brief Creates a new session for a connecting client.
    /// @param playerId Unique player identifier.
    /// @return Reference to the new session, or error if already exists.
    [[nodiscard]] core::Expected<Session*> connect(core::u32 playerId);

    /// @brief Disconnects a client session.
    /// @param playerId Player to disconnect.
    [[nodiscard]] core::Expected<void> disconnect(core::u32 playerId);

    /// @brief Finds a session by player ID.
    /// @return Pointer to the session or nullptr.
    [[nodiscard]] Session* find(core::u32 playerId) noexcept;

    /// @brief Finds a session by player ID (const).
    [[nodiscard]] const Session* find(core::u32 playerId) const noexcept;

    /// @brief Iterates all active sessions.
    /// @param callback Called for each active session.
    void forEach(const std::function<void(Session&)>& callback);

    /// @brief Cleans up sessions that have been inactive for too long.
    /// @param timeoutMs Inactivity timeout in milliseconds.
    /// @return Number of sessions reaped.
    core::u32 reapTimedOut(core::f64 timeoutMs);

    /// @brief Returns the number of active sessions.
    [[nodiscard]] core::u32 activeCount() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::net::session
