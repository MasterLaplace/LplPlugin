/**
 * @file SessionManager.cpp
 * @brief SessionManager implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/session/SessionManager.hpp>
#include <lpl/core/Log.hpp>

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace lpl::net::session {

struct SessionManager::Impl
{
    std::unordered_map<core::u32, std::unique_ptr<Session>> sessions;
};

SessionManager::SessionManager()
    : _impl{std::make_unique<Impl>()}
{}

SessionManager::~SessionManager() = default;

core::Expected<Session*> SessionManager::connect(core::u32 playerId)
{
    if (_impl->sessions.contains(playerId))
    {
        return core::makeError(core::ErrorCode::AlreadyExists,
                               "Session already exists for player");
    }

    auto session = std::make_unique<Session>(playerId);
    session->setState(SessionState::Connected);
    auto* ptr = session.get();
    _impl->sessions.emplace(playerId, std::move(session));

    core::Log::info("SessionManager: player connected");
    return ptr;
}

core::Expected<void> SessionManager::disconnect(core::u32 playerId)
{
    auto it = _impl->sessions.find(playerId);
    if (it == _impl->sessions.end())
    {
        return core::makeError(core::ErrorCode::NotFound, "Session not found");
    }

    it->second->setState(SessionState::Disconnected);
    _impl->sessions.erase(it);

    core::Log::info("SessionManager: player disconnected");
    return {};
}

Session* SessionManager::find(core::u32 playerId) noexcept
{
    auto it = _impl->sessions.find(playerId);
    return (it != _impl->sessions.end()) ? it->second.get() : nullptr;
}

const Session* SessionManager::find(core::u32 playerId) const noexcept
{
    auto it = _impl->sessions.find(playerId);
    return (it != _impl->sessions.end()) ? it->second.get() : nullptr;
}

void SessionManager::forEach(const std::function<void(Session&)>& callback)
{
    for (auto& [id, session] : _impl->sessions)
    {
        callback(*session);
    }
}

core::u32 SessionManager::reapTimedOut(core::f64 timeoutMs)
{
    const auto now = Session::Clock::now();
    core::u32 reaped = 0;

    for (auto it = _impl->sessions.begin(); it != _impl->sessions.end(); )
    {
        const auto elapsed = std::chrono::duration<core::f64, std::milli>(
            now - it->second->lastActivity()).count();

        if (elapsed > timeoutMs)
        {
            it->second->setState(SessionState::Disconnected);
            it = _impl->sessions.erase(it);
            ++reaped;
        }
        else
        {
            ++it;
        }
    }

    return reaped;
}

core::u32 SessionManager::activeCount() const noexcept
{
    return static_cast<core::u32>(_impl->sessions.size());
}

void SessionManager::broadcastState(transport::ITransport& transport,
                                     std::span<const core::byte> data)
{
    // Fragmentation: split data into kMaxPayloadSize chunks
    const core::u32 totalSize = static_cast<core::u32>(data.size());
    core::u32 offset = 0;

    while (offset < totalSize)
    {
        const core::u32 chunkSize = std::min(kMaxPayloadSize, totalSize - offset);
        auto fragment = data.subspan(offset, chunkSize);

        // Send to each active session
        for (auto& [id, session] : _impl->sessions)
        {
            if (session->state() != SessionState::Connected)
            {
                continue;
            }

            // Use the session's stored address (player ID as address proxy)
            auto result = transport.send(fragment, nullptr);
            if (!result.has_value())
            {
                core::Log::warn("SessionManager: broadcast fragment failed");
            }
        }

        offset += chunkSize;
    }
}

bool SessionManager::isDuplicate(core::u32 playerId) const noexcept
{
    return _impl->sessions.contains(playerId);
}

} // namespace lpl::net::session
