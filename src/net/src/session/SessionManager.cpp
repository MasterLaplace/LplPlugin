// /////////////////////////////////////////////////////////////////////////////
/// @file SessionManager.cpp
/// @brief SessionManager implementation.
// /////////////////////////////////////////////////////////////////////////////

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
    : impl_{std::make_unique<Impl>()}
{}

SessionManager::~SessionManager() = default;

core::Expected<Session*> SessionManager::connect(core::u32 playerId)
{
    if (impl_->sessions.contains(playerId))
    {
        return core::makeError(core::ErrorCode::AlreadyExists,
                               "Session already exists for player");
    }

    auto session = std::make_unique<Session>(playerId);
    session->setState(SessionState::Connected);
    auto* ptr = session.get();
    impl_->sessions.emplace(playerId, std::move(session));

    core::Log::info("SessionManager: player connected");
    return ptr;
}

core::Expected<void> SessionManager::disconnect(core::u32 playerId)
{
    auto it = impl_->sessions.find(playerId);
    if (it == impl_->sessions.end())
    {
        return core::makeError(core::ErrorCode::NotFound, "Session not found");
    }

    it->second->setState(SessionState::Disconnected);
    impl_->sessions.erase(it);

    core::Log::info("SessionManager: player disconnected");
    return {};
}

Session* SessionManager::find(core::u32 playerId) noexcept
{
    auto it = impl_->sessions.find(playerId);
    return (it != impl_->sessions.end()) ? it->second.get() : nullptr;
}

const Session* SessionManager::find(core::u32 playerId) const noexcept
{
    auto it = impl_->sessions.find(playerId);
    return (it != impl_->sessions.end()) ? it->second.get() : nullptr;
}

void SessionManager::forEach(const std::function<void(Session&)>& callback)
{
    for (auto& [id, session] : impl_->sessions)
    {
        callback(*session);
    }
}

core::u32 SessionManager::reapTimedOut(core::f64 timeoutMs)
{
    const auto now = Session::Clock::now();
    core::u32 reaped = 0;

    for (auto it = impl_->sessions.begin(); it != impl_->sessions.end(); )
    {
        const auto elapsed = std::chrono::duration<core::f64, std::milli>(
            now - it->second->lastActivity()).count();

        if (elapsed > timeoutMs)
        {
            it->second->setState(SessionState::Disconnected);
            it = impl_->sessions.erase(it);
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
    return static_cast<core::u32>(impl_->sessions.size());
}

} // namespace lpl::net::session
