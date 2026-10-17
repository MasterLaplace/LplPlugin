#ifndef LPL_ENGINE_CHARACTER_CONTROLLER_INL
#define LPL_ENGINE_CHARACTER_CONTROLLER_INL

namespace lpl::engine {

namespace detail {

/**
 * @brief Moves @p value toward @p target by at most @p maximum.
 *
 * @param value The current value.
 * @param target The target value.
 * @param maximum The maximum amount to move.
 * @return The new value.
 */
[[nodiscard]] inline math::Fixed32 approach(math::Fixed32 value, math::Fixed32 target, math::Fixed32 maximum) noexcept
{
    const math::Fixed32 difference = target - value;
    if (difference > maximum)
        return value + maximum;
    if (difference < -maximum)
        return value - maximum;
    return target;
}

[[nodiscard]] inline math::Fixed32 absolute(math::Fixed32 value) noexcept
{
    return value < math::Fixed32{} ? -value : value;
}

} // namespace detail

template <typename SpaceAt>
void CharacterController::placeAt(math::Fixed32 worldX, math::Fixed32 worldZ, math::Fixed32 startY,
                                  SpaceAt &&spaceAt) noexcept
{
    _x = worldX;
    _z = worldZ;
    const procgen::VerticalSpan span = spaceAt(worldX.toInt(), worldZ.toInt(), startY);
    _groundHeight = span.floor;
    _ceiling = span.ceiling;
    _enclosed = span.enclosed;
    _y = _groundHeight;
    _vx = math::Fixed32{};
    _vy = math::Fixed32{};
    _vz = math::Fixed32{};
    _grounded = true;
    _sliding = false;
    _coyote = 0u;
    _jumpBuffer = 0u;
    _airborneTicks = 0u;
}

template <typename SpaceAt>
void CharacterController::moveAxis(const CharacterParams &params, math::Fixed32 &coordinate, math::Fixed32 delta,
                                   SpaceAt &&spaceAt)
{
    if (delta == math::Fixed32{})
        return;

    const math::Fixed32 previous = coordinate;
    coordinate = coordinate + delta;

    if (!_grounded)
        return;

    const procgen::VerticalSpan ahead = spaceAt(_x.toInt(), _z.toInt(), _y);
    const math::Fixed32 rise = ahead.floor - _y;
    if (rise > params.stepHeight)
    {
        coordinate = previous;
        ++_blocked;
        return;
    }

    const math::Fixed32 standingAt = rise > math::Fixed32{} ? ahead.floor : _y;
    if (ahead.ceiling - standingAt < params.standHeight)
    {
        coordinate = previous;
        ++_ducked;
        return;
    }

    if (rise > math::Fixed32{})
        _y = ahead.floor;
}

template <typename SpaceAt>
void CharacterController::step(const CharacterParams &params, const CharacterIntent &intent, math::Fixed32 dt,
                               SpaceAt &&spaceAt)
{
    _yaw = _yaw + intent.turn;
    math::Fixed32 sinYaw{};
    math::Fixed32 cosYaw{};
    math::Cordic::sincos(_yaw, sinYaw, cosYaw);

    const math::Fixed32 speed = intent.sprint ? params.walkSpeed * params.sprintScale : params.walkSpeed;
    math::Fixed32 wishX = (-intent.forward * sinYaw) + (intent.strafe * cosYaw);
    math::Fixed32 wishZ = (-intent.forward * cosYaw) - (intent.strafe * sinYaw);

    const math::Fixed32 wishLength = math::fixedSqrt(wishX * wishX + wishZ * wishZ);
    if (wishLength > math::Fixed32::one())
    {
        wishX = wishX / wishLength;
        wishZ = wishZ / wishLength;
    }
    const math::Fixed32 targetX = wishX * speed;
    const math::Fixed32 targetZ = wishZ * speed;

    const bool pushing = wishLength > math::Fixed32::fromFloat(0.01f);
    math::Fixed32 rate = params.acceleration;
    if (!_grounded)
        rate = params.acceleration * params.airControl;
    else if (!pushing)
        rate = params.groundFriction;

    const math::Fixed32 maximum = rate * dt;
    _vx = detail::approach(_vx, targetX, maximum);
    _vz = detail::approach(_vz, targetZ, maximum);

    if (intent.jump)
        _jumpBuffer = params.jumpBufferTicks;
    else if (_jumpBuffer != 0u)
        --_jumpBuffer;

    const bool mayJump = (_grounded || _coyote != 0u) && !_sliding;
    if (_jumpBuffer != 0u && mayJump)
    {
        _vy = params.jumpSpeed;
        _grounded = false;
        _coyote = 0u;
        _jumpBuffer = 0u;
        ++_jumps;
    }

    if (!_grounded)
    {
        _vy = _vy - params.gravity * dt;
        if (_vy < -params.terminalFall)
            _vy = -params.terminalFall;
    }

    moveAxis(params, _x, _vx * dt, spaceAt);
    moveAxis(params, _z, _vz * dt, spaceAt);

    _sliding = false;
    if (_grounded && !_enclosed)
    {
        const math::Fixed32 here = spaceAt(_x.toInt(), _z.toInt(), _y).floor;
        const math::Fixed32 eastward = spaceAt(_x.toInt() + 1, _z.toInt(), _y).floor - here;
        const math::Fixed32 southward = spaceAt(_x.toInt(), _z.toInt() + 1, _y).floor - here;

        if (detail::absolute(eastward) > params.maxSlope || detail::absolute(southward) > params.maxSlope)
        {
            _sliding = true;
            const math::Fixed32 slide = params.slideSpeed * dt;
            if (detail::absolute(eastward) > params.maxSlope)
                _x = _x - (eastward > math::Fixed32{} ? slide : -slide);
            if (detail::absolute(southward) > params.maxSlope)
                _z = _z - (southward > math::Fixed32{} ? slide : -slide);
        }
    }

    _y = _y + _vy * dt;
    const procgen::VerticalSpan here = spaceAt(_x.toInt(), _z.toInt(), _y);
    _groundHeight = here.floor;
    _ceiling = here.ceiling;
    _enclosed = here.enclosed;

    const bool wasGrounded = _grounded;
    if (_y <= _groundHeight)
    {
        _y = _groundHeight;
        if (_vy < math::Fixed32{})
            _vy = math::Fixed32{};
        _grounded = true;
        _airborneTicks = 0u;
        _coyote = params.coyoteTicks;
    }
    else
    {
        _grounded = false;
        _sliding = false;
        if (_airborneTicks < 0xFFFFFFFFu)
            ++_airborneTicks;
        if (wasGrounded)
            _coyote = params.coyoteTicks;
        else if (_coyote != 0u)
            --_coyote;
    }

    if (_enclosed)
    {
        math::Fixed32 highest = _ceiling - params.standHeight;
        if (highest < _groundHeight)
            highest = _groundHeight;
        if (_y > highest)
        {
            _y = highest;
            if (_vy > math::Fixed32{})
                _vy = math::Fixed32{};
            ++_headBumps;
        }
    }
}

inline core::u32 CharacterController::fold(core::u32 seed) const noexcept
{
    core::u32 hash = seed;
    const auto mix = [&hash](core::u32 word) {
        hash ^= word;
        hash *= 0x01000193u;
    };
    mix(static_cast<core::u32>(_x.raw()));
    mix(static_cast<core::u32>(_y.raw()));
    mix(static_cast<core::u32>(_z.raw()));
    mix(static_cast<core::u32>(_vx.raw()));
    mix(static_cast<core::u32>(_vy.raw()));
    mix(static_cast<core::u32>(_vz.raw()));
    mix(static_cast<core::u32>(_yaw.raw()));
    mix(_grounded ? 1u : 0u);
    mix(_sliding ? 1u : 0u);
    mix(_enclosed ? 1u : 0u);
    return hash;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_CHARACTER_CONTROLLER_INL
