/**
 * @file CharacterController.inl
 * @brief Out-of-line definitions for engine::CharacterController.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-31
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_CHARACTER_CONTROLLER_INL
#    define LPL_ENGINE_CHARACTER_CONTROLLER_INL

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

    // Only a body ON the ground can be stopped by a step. In mid-air the same rise
    // is a hillside the jump is arcing over, and refusing the move there pins the
    // player against invisible walls in open sky.
    if (!_grounded)
        return;

    const procgen::VerticalSpan ahead = spaceAt(_x.toInt(), _z.toInt(), _y);
    const math::Fixed32 rise = ahead.floor - _y;
    if (rise > params.stepHeight)
    {
        // A wall. Undo the axis rather than the whole move: refusing both axes at
        // once makes a body walking into a corner stop dead, when what it should do
        // is slide along the face it did not hit. Two independent axis tests give
        // that for free — which is the same reason the collision resolver tests
        // axes separately.
        coordinate = previous;
        ++_blocked;
        return;
    }

    // A gap too low to stand in is a wall the OTHER way round, and it has to be
    // refused here rather than resolved vertically afterwards. Let the move through
    // and the body is inside rock: the vertical pass then has to put it somewhere,
    // and every somewhere is either a teleport onto the hill or a fall through the
    // floor. Refusing the step is the only answer that leaves the body where it was.
    //
    // Measured against the FLOOR IT WOULD STAND ON, not against its current height:
    // stepping up into a gap and standing in it are the same act, and a test against
    // the old height passes a body under a slab it cannot then fit beneath.
    const math::Fixed32 standingAt = rise > math::Fixed32{} ? ahead.floor : _y;
    if (ahead.ceiling - standingAt < params.standHeight)
    {
        coordinate = previous;
        ++_ducked;
        return;
    }

    // A rise the body CAN take is taken immediately, feet first: a heightfield has
    // no ramp between two cells, so a walker that waited for gravity would spend
    // every step falling a few centimetres and never report as grounded.
    if (rise > math::Fixed32{})
        _y = ahead.floor;
}

template <typename SpaceAt>
void CharacterController::step(const CharacterParams &params, const CharacterIntent &intent, math::Fixed32 dt,
                               SpaceAt &&spaceAt)
{
    // ── Heading ─────────────────────────────────────────────────────────────
    //
    // Fixed32 and CORDIC, not a float yaw and a libm sine. This is what keeps the
    // POSITION authoritative: the walk direction comes from here.
    _yaw = _yaw + intent.turn;
    math::Fixed32 sinYaw{};
    math::Fixed32 cosYaw{};
    math::Cordic::sincos(_yaw, sinYaw, cosYaw);

    // ── Desired horizontal velocity ─────────────────────────────────────────
    //
    // The same convention OrbitCamera walks by, so the body and the view agree
    // about which way "forward" points.
    const math::Fixed32 speed = intent.sprint ? params.walkSpeed * params.sprintScale : params.walkSpeed;
    math::Fixed32 wishX = (-intent.forward * sinYaw) + (intent.strafe * cosYaw);
    math::Fixed32 wishZ = (-intent.forward * cosYaw) - (intent.strafe * sinYaw);

    // Diagonal input must not be faster than straight input. Normalising only when
    // the magnitude exceeds one keeps a half-pressed direction slow, which an
    // unconditional normalise would silently turn into a full-speed walk.
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

    // ── Jump ────────────────────────────────────────────────────────────────
    //
    // Buffered on the way in and forgiven on the way out, so the two ways a player
    // legitimately mistimes a jump both work. Ordered BEFORE gravity: a jump taken
    // this tick should get its whole upward speed, not one already shaved by a
    // tick of fall.
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

    // ── Gravity ─────────────────────────────────────────────────────────────
    if (!_grounded)
    {
        _vy = _vy - params.gravity * dt;
        if (_vy < -params.terminalFall)
            _vy = -params.terminalFall;
    }

    // ── Move ────────────────────────────────────────────────────────────────
    //
    // Horizontally first and one axis at a time, so a wall on one axis does not
    // cancel the other. Then vertically, against the ground under wherever the
    // horizontal move ended up — the other order tests the fall against the cell
    // being left rather than the one being entered.
    moveAxis(params, _x, _vx * dt, spaceAt);
    moveAxis(params, _z, _vz * dt, spaceAt);

    // ── Slope ───────────────────────────────────────────────────────────────
    //
    // BEFORE the vertical resolution, and horizontal only. Both of those are the
    // fix to a bug the probe caught: the slide used to end by snapping the body onto
    // whatever ground it had slid over, so walking off a cliff teleported the walker
    // to the bottom of it in a single tick, still reported as standing. Sliding down
    // something steep is a FALL that has started — it moves you sideways, and gravity
    // does the rest through the ordinary ground check below.
    //
    // The steepness is measured by sampling one cell along each axis rather than by
    // an angle: comparing a rise against a run is a subtraction, and an angle would
    // need an arctangent nothing here is allowed to call.
    //
    // NOT under a roof, and that exclusion is load-bearing rather than tidy. A gallery
    // floor is quantised to whole voxel levels, so every level change reads as a slope
    // steeper than any walkable one and a body underground would slide continuously —
    // into a wall, where the vertical pass has to put it somewhere and every somewhere
    // is wrong. A cave floor is a floor: it is what it is, and you stand on it.
    _sliding = false;
    if (_grounded && !_enclosed)
    {
        const math::Fixed32 here = spaceAt(_x.toInt(), _z.toInt(), _y).floor;
        const math::Fixed32 eastward = spaceAt(_x.toInt() + 1, _z.toInt(), _y).floor - here;
        const math::Fixed32 southward = spaceAt(_x.toInt(), _z.toInt() + 1, _y).floor - here;

        if (detail::absolute(eastward) > params.maxSlope || detail::absolute(southward) > params.maxSlope)
        {
            _sliding = true;
            // Downhill is where the ground DROPS, so the step is against the
            // gradient — sliding uphill would be the sign error to make here.
            const math::Fixed32 slide = params.slideSpeed * dt;
            if (detail::absolute(eastward) > params.maxSlope)
                _x = _x - (eastward > math::Fixed32{} ? slide : -slide);
            if (detail::absolute(southward) > params.maxSlope)
                _z = _z - (southward > math::Fixed32{} ? slide : -slide);
        }
    }

    // ── Vertical ────────────────────────────────────────────────────────────
    //
    // Last, against the ground under wherever the horizontal move and the slide
    // ended up. The other order tests the fall against the cell being LEFT rather
    // than the one being entered, which is how a body walks into the air and stays
    // there — or, as above, drops into the floor.
    _y = _y + _vy * dt;
    const procgen::VerticalSpan here = spaceAt(_x.toInt(), _z.toInt(), _y);
    _groundHeight = here.floor;
    _ceiling = here.ceiling;
    _enclosed = here.enclosed;

    const bool wasGrounded = _grounded;
    if (_y <= _groundHeight)
    {
        _y = _groundHeight;
        // Landing kills downward speed only. Clearing an UPWARD velocity here would
        // eat the first tick of every jump, because a jump starts at ground level
        // and is therefore still "at or below" it.
        if (_vy < math::Fixed32{})
            _vy = math::Fixed32{};
        _grounded = true;
        _airborneTicks = 0u;
        _coyote = params.coyoteTicks;
    }
    else
    {
        _grounded = false;
        _sliding = false; // in the air, nothing is underfoot to slide on
        if (_airborneTicks < 0xFFFFFFFFu)
            ++_airborneTicks;
        // Coyote time counts down only while falling OFF something. It was granted
        // on the last landing; a jump already consumed it.
        if (wasGrounded)
            _coyote = params.coyoteTicks;
        else if (_coyote != 0u)
            --_coyote;
    }

    // ── Head ────────────────────────────────────────────────────────────────
    //
    // After the ground, because a body that is both standing and scraping is standing:
    // clamping to the ceiling first and then to the floor would let a gap shorter than
    // the body push it under its own floor, which is the one place it can never be.
    // The clamp is floored at the ground for the same reason — in a gap it should not
    // have been able to enter, it stands crushed rather than sinking.
    if (_enclosed)
    {
        math::Fixed32 highest = _ceiling - params.standHeight;
        if (highest < _groundHeight)
            highest = _groundHeight;
        if (_y > highest)
        {
            _y = highest;
            // Upward speed only. Killing a downward one here would hold a body up
            // against a ceiling it is falling away from.
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
    // Enclosed decides whether the body slides, so it decides where the body will be:
    // it is state, not presentation, and a signature that omitted it would call two
    // diverging runs identical for exactly one tick.
    mix(_enclosed ? 1u : 0u);
    return hash;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_CHARACTER_CONTROLLER_INL
