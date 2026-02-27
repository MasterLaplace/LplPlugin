/**
 * @file Session.cpp
 * @brief Session implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/session/Session.hpp>

namespace lpl::net::session {

Session::Session(core::u32 playerId)
    : _playerId{playerId}
    , _lastActivity{Clock::now()}
{}

Session::~Session() = default;

core::u32    Session::playerId() const noexcept       { return _playerId; }
SessionState Session::state() const noexcept          { return _state; }
void         Session::setState(SessionState s) noexcept { _state = s; }

void         Session::bindEntity(ecs::EntityId e) noexcept { _entity = e; }
ecs::EntityId Session::boundEntity() const noexcept       { return _entity; }

void Session::recordRtt(core::f32 rttMs) noexcept
{
    constexpr core::f32 alpha = 0.125f;
    _smoothedRtt = _smoothedRtt * (1.0f - alpha) + rttMs * alpha;
}

core::f32 Session::smoothedRtt() const noexcept { return _smoothedRtt; }

core::u32 Session::lastInputSequence() const noexcept     { return _lastInputSeq; }
void      Session::setLastInputSequence(core::u32 s) noexcept { _lastInputSeq = s; }

Session::TimePoint Session::lastActivity() const noexcept { return _lastActivity; }
void               Session::touch() noexcept              { _lastActivity = Clock::now(); }

} // namespace lpl::net::session
