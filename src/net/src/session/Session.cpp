// /////////////////////////////////////////////////////////////////////////////
/// @file Session.cpp
/// @brief Session implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/session/Session.hpp>

namespace lpl::net::session {

Session::Session(core::u32 playerId)
    : playerId_{playerId}
    , lastActivity_{Clock::now()}
{}

Session::~Session() = default;

core::u32    Session::playerId() const noexcept       { return playerId_; }
SessionState Session::state() const noexcept          { return state_; }
void         Session::setState(SessionState s) noexcept { state_ = s; }

void         Session::bindEntity(ecs::EntityId e) noexcept { entity_ = e; }
ecs::EntityId Session::boundEntity() const noexcept       { return entity_; }

void Session::recordRtt(core::f32 rttMs) noexcept
{
    constexpr core::f32 alpha = 0.125f;
    smoothedRtt_ = smoothedRtt_ * (1.0f - alpha) + rttMs * alpha;
}

core::f32 Session::smoothedRtt() const noexcept { return smoothedRtt_; }

core::u32 Session::lastInputSequence() const noexcept     { return lastInputSeq_; }
void      Session::setLastInputSequence(core::u32 s) noexcept { lastInputSeq_ = s; }

Session::TimePoint Session::lastActivity() const noexcept { return lastActivity_; }
void               Session::touch() noexcept              { lastActivity_ = Clock::now(); }

} // namespace lpl::net::session
