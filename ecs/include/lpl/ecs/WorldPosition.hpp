/**
 * @file WorldPosition.hpp
 * @brief The one place that knows a position is a chunk plus an offset.
 *
 * @warning **One place, and that is the whole point of the file.** A body that walks past a chunk
 * boundary has to move to the next cell and fold its local coordinate back into range; two
 * implementations of that would drift, and a drift in a position is a body that is somewhere
 * else. Everything that moves a `WorldCell`-carrying entity normalises here.
 *
 * **Why the split exists at all.** Q16.16 spans +/-32768 units, so a flat world position cannot
 * address a planet -- Earth's circumference alone is a thousand times that. Widening the word is
 * the wrong fix twice: it doubles every position in the game, and precision far from the origin
 * still degrades, which is the jitter every large world has at its edges. A chunk index plus a
 * local offset has **constant** precision everywhere and unlimited range, and it is the decision
 * `procgen/Chunking.hpp` already argues for terrain -- this is the same one for bodies.
 *
 * @warning It also removes a class of bug rather than a bug: a product of two world-scale Q16.16 values
 * overflows at about 181 units, so `dx*dx` on real spacings is noise. Local coordinates are
 * bounded by the chunk, so the operands are small by construction.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_WORLDPOSITION_HPP
#    define LPL_ECS_WORLDPOSITION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/ComponentData.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::ecs {

/**
 * @struct WorldPosition
 * @brief A chunk and an offset inside it, which together address anywhere.
 */
struct WorldPosition {
    core::i32 chunkX{0};
    core::i32 chunkZ{0};
    math::Fixed32 localX{};
    math::Fixed32 localZ{};
};

/**
 * @brief Folds a local coordinate back into its chunk, carrying into the cell index.
 *
 * @warning Called by everything that moves such a body, and by nothing else -- a second implementation
 * would eventually disagree about which side of a boundary a coordinate belongs to, and a body
 * would jump a chunk width.
 *
 * @warning The fold uses a loop rather than a division because `Fixed32` division rounds, and a rounding
 * here does not lose a fraction of a unit -- it puts the body in the wrong chunk. A step is at most
 * a few units, so the loop runs once at a boundary and not at all otherwise.
 *
 * @param position  The position to normalise, in place.
 * @param chunkSize Width of a chunk in world units. Must be positive.
 */
constexpr void normaliseWorldPosition(WorldPosition &position, math::Fixed32 chunkSize) noexcept
{
    if (chunkSize <= math::Fixed32{})
        return;
    while (position.localX >= chunkSize)
    {
        position.localX = position.localX - chunkSize;
        ++position.chunkX;
    }
    while (position.localX < math::Fixed32{})
    {
        position.localX = position.localX + chunkSize;
        --position.chunkX;
    }
    while (position.localZ >= chunkSize)
    {
        position.localZ = position.localZ - chunkSize;
        ++position.chunkZ;
    }
    while (position.localZ < math::Fixed32{})
    {
        position.localZ = position.localZ + chunkSize;
        --position.chunkZ;
    }
}

/**
 * @brief The offset from one world position to another.
 *
 * @warning **This is where the overflow used to live.** Two positions a thousand units apart cannot
 * have their difference expressed in Q16.16 as a single flat coordinate -- but the difference of
 * their CHUNKS is an integer, and only the leftover fits in a fixed word. So the delta is returned
 * as a chunk count plus a remainder, and a caller that only needs a direction never multiplies two
 * large numbers.
 *
 * @param from      Where the body is.
 * @param to        Where it is going.
 * @param chunkSize Width of a chunk in world units.
 * @param outChunkX Receives the whole chunks between them, x.
 * @param outChunkZ Receives the whole chunks between them, z.
 * @param outLocalX Receives the leftover, x. Always inside one chunk.
 * @param outLocalZ Receives the leftover, z.
 */
constexpr void worldDelta(const WorldPosition &from, const WorldPosition &to, math::Fixed32 chunkSize,
                          core::i32 &outChunkX, core::i32 &outChunkZ, math::Fixed32 &outLocalX,
                          math::Fixed32 &outLocalZ) noexcept
{
    (void) chunkSize;
    outChunkX = to.chunkX - from.chunkX;
    outChunkZ = to.chunkZ - from.chunkZ;
    outLocalX = to.localX - from.localX;
    outLocalZ = to.localZ - from.localZ;
}

/**
 * @brief Whether two world positions are the same place.
 *
 * @param a First.
 * @param b Second.
 * @return true when both the cell and the offset match.
 */
[[nodiscard]] constexpr bool sameWorldPosition(const WorldPosition &a, const WorldPosition &b) noexcept
{
    return a.chunkX == b.chunkX && a.chunkZ == b.chunkZ && a.localX == b.localX && a.localZ == b.localZ;
}

} // namespace lpl::ecs

#endif // LPL_ECS_WORLDPOSITION_HPP
