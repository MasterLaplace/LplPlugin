/**
 * @file Cordic.cpp
 * @brief CORDIC implementation with pre-computed atan lookup table.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#include "lpl/math/Cordic.hpp"

#include <lpl/std/cmath.hpp>

#include <array>

namespace lpl::math {

namespace {

constexpr core::u32 kIterations = 16;

constexpr auto kAtanTable = [] {
    std::array<core::i32, kIterations> tbl{};
    for (core::u32 i = 0; i < kIterations; ++i)
        tbl[i] = static_cast<core::i32>(lpl::pmr::atan(1.0 / (1 << i)) * Fixed32::kOne);
    return tbl;
}();

constexpr core::i32 kGain = static_cast<core::i32>(0.607252935 * Fixed32::kOne);

} // anonymous namespace

Fixed32 Cordic::sin([[maybe_unused]] Fixed32 angle)
{
    Fixed32 s, c;
    sincos(angle, s, c);
    return s;
}

Fixed32 Cordic::cos([[maybe_unused]] Fixed32 angle)
{
    Fixed32 s, c;
    sincos(angle, s, c);
    return c;
}

void Cordic::sincos([[maybe_unused]] Fixed32 angle, [[maybe_unused]] Fixed32 &outSin, [[maybe_unused]] Fixed32 &outCos)
{
    core::i32 x = kGain;
    core::i32 y = 0;
    core::i32 z = angle.raw();

    // ── Getting the angle into the range the rotation can actually reach ─────
    //
    // The loop below steers z to zero using a table of atan(2^-i), so the largest
    // angle it can consume is their sum: 1.7433 radians, a little under a hundred
    // degrees. Past that it runs out of table and simply stops turning — sin and cos
    // freeze at their values there and never move again.
    //
    // That was live, and in the authoritative path. CharacterController accumulates
    // its heading without bound and resolves the walk direction through here, so a
    // player who turned more than a hundred degrees kept walking in the direction they
    // had at a hundred degrees. It reads as the controls going dead rather than as a
    // maths fault, which is why it survived: the first sixty degrees of every turn
    // work perfectly.
    //
    // Reduction is exact and integer, so it costs the determinism contract nothing:
    // wrap into (-pi, pi] with one remainder, then fold the two outer quadrants into
    // the inner two by half a turn and a sign flip — cos and sin are both odd about a
    // half-turn. What is left is at most pi/2 = 1.5708, inside the table's reach with
    // room to spare.
    constexpr core::i32 kPiRaw = static_cast<core::i32>(3.14159265358979323846 * Fixed32::kOne);
    constexpr core::i32 kHalfPiRaw = kPiRaw / 2;
    constexpr core::i32 kTwoPiRaw = kPiRaw * 2;

    z %= kTwoPiRaw;
    if (z > kPiRaw)
        z -= kTwoPiRaw;
    else if (z < -kPiRaw)
        z += kTwoPiRaw;

    bool opposite = false;
    if (z > kHalfPiRaw)
    {
        z -= kPiRaw;
        opposite = true;
    }
    else if (z < -kHalfPiRaw)
    {
        z += kPiRaw;
        opposite = true;
    }

    for (core::u32 i = 0; i < kIterations; ++i)
    {
        core::i32 dx = y >> i;
        core::i32 dy = x >> i;
        if (z >= 0)
        {
            x -= dx;
            y += dy;
            z -= kAtanTable[i];
        }
        else
        {
            x += dx;
            y -= dy;
            z += kAtanTable[i];
        }
    }

    outCos = Fixed32::fromRaw(opposite ? -x : x);
    outSin = Fixed32::fromRaw(opposite ? -y : y);
}

Fixed32 Cordic::atan2([[maybe_unused]] Fixed32 yVal, [[maybe_unused]] Fixed32 xVal)
{
    core::i32 x = xVal.raw();
    core::i32 y = yVal.raw();
    core::i32 z = 0;

    if (x == 0 && y == 0)
        return Fixed32{};

    // ── The quadrant, which the rotation below cannot reach on its own ───────
    //
    // Vectoring CORDIC accumulates angles from a table of atan(2^-i), so the largest
    // angle it can ever produce is their sum — 1.7433 radians, a shade under a hundred
    // degrees. That is the RIGHT half-plane and nothing else, and for x < 0 the loop
    // simply saturates there: atan2(0, -1) came back as 1.7433 instead of pi.
    //
    // Which is to say this function was wrong over half its domain, and wrong in the
    // one way its NAME exists to rule out — resolving all four quadrants is the whole
    // difference between atan2 and atan. It survived because nothing called it; the
    // first caller walked a body due north and it set off west.
    //
    // The fix is the standard one: reflect the input through the origin into the half
    // the rotation can reach, and put the half-turn back afterwards. The sign of that
    // half-turn comes from the ORIGINAL y, so the result stays in (-pi, pi].
    constexpr core::i32 kPiRaw = static_cast<core::i32>(3.14159265358979323846 * Fixed32::kOne);
    core::i32 halfTurn = 0;
    if (x < 0)
    {
        halfTurn = y >= 0 ? kPiRaw : -kPiRaw;
        x = -x;
        y = -y;
    }

    for (core::u32 i = 0; i < kIterations; ++i)
    {
        core::i32 dx = y >> i;
        core::i32 dy = x >> i;
        if (y >= 0)
        {
            x += dx;
            y -= dy;
            z += kAtanTable[i];
        }
        else
        {
            x -= dx;
            y += dy;
            z -= kAtanTable[i];
        }
    }

    return Fixed32::fromRaw(z + halfTurn);
}

} // namespace lpl::math
