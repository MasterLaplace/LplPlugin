/**
 * @file SpringBody.hpp
 * @brief A body made of masses and springs, so it fits terrain nobody authored.
 *
 * Keyframed animation assumes the ground was designed. On procedural terrain it
 * cannot be: a walk cycle authored for a flat plane slides over a slope, floats
 * over a step and clips through a boulder, and no amount of blending fixes it
 * because the animation does not know the boulder exists.
 *
 * The alternative is to stop drawing a creature and start simulating one. A body
 * is a handful of masses (`BodyChunk`) joined by springs (`BodyChunkConnection`)
 * obeying Hooke's law. The AI does not play a walk animation; it picks a *place
 * to put a foot*, an inverse-kinematics solve finds the joint angles that reach
 * it, and pulling on that anchor drags the rest of the body along. Contact with
 * the world is then an input to the motion instead of a decoration on it.
 *
 * **This is authoritative.** Chunk positions decide collisions, so this is
 * Fixed32 like everything else the simulation depends on — which rules out the
 * usual `acos` in the two-bone solve. The angle comes from CORDIC, and the one
 * square root is the hardware instruction the determinism contract permits.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_SPRINGBODY_HPP
#    define LPL_AI_SPRINGBODY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/**
 * @struct BodyChunk
 * @brief One mass of a body: a circle with velocity.
 */
struct BodyChunk {
    math::Fixed32 x{};
    math::Fixed32 z{};
    math::Fixed32 vx{};
    math::Fixed32 vz{};
    math::Fixed32 radius{math::Fixed32::half()};
    math::Fixed32 inverseMass{math::Fixed32::one()}; ///< 0 pins the chunk in place.
};

/**
 * @struct BodyChunkConnection
 * @brief A spring between two chunks.
 */
struct BodyChunkConnection {
    core::u32 a{0u};
    core::u32 b{0u};
    math::Fixed32 restLength{math::Fixed32::one()};
    math::Fixed32 stiffness{math::Fixed32::half()}; ///< In [0, 1]; 1 is rigid.
};

/**
 * @struct SpringBodyParams
 * @brief Global forces and the integrator's damping.
 */
struct SpringBodyParams {
    math::Fixed32 damping{math::Fixed32::fromRaw(0xF333)};   ///< Velocity retained per step (~0.95).
    math::Fixed32 gravityZ{};                                ///< Constant pull; zero for a top-down world.
    core::u32 relaxations{4u};                               ///< Constraint iterations per step.
    math::Fixed32 maxSpeed{math::Fixed32::fromInt(4)};       ///< Clamp, so a stiff spring cannot explode.
};

/**
 * @class SpringBody
 * @brief Chunks, connections, and the integrator that keeps them together.
 */
class SpringBody {
public:
    /// @brief Adds a chunk and returns its index.
    core::u32 addChunk(const BodyChunk &chunk);

    /// @brief Joins two chunks with a spring at their current separation.
    void connect(core::u32 a, core::u32 b, math::Fixed32 stiffness);

    /**
     * @brief Advances the body one step.
     *
     * Position-based: velocities are integrated, then the spring constraints are
     * relaxed a fixed number of times. Relaxation rather than force accumulation
     * because a stiff force-based spring needs a small timestep to stay stable,
     * and a body that explodes when the frame rate changes is not authoritative.
     *
     * @param params Damping, gravity and iteration count.
     */
    void step(const SpringBodyParams &params);

    /**
     * @brief Pulls a chunk toward a point — the effect of a limb gripping.
     *
     * The whole motor model: the AI chooses where to place a foot, and the body
     * follows the foot. Nothing here knows what "walking" is.
     *
     * @param chunk    Index of the anchored chunk.
     * @param targetX  Where it is being pulled.
     * @param targetZ  Where it is being pulled.
     * @param strength Share of the gap closed per step, in [0, 1].
     */
    void pull(core::u32 chunk, math::Fixed32 targetX, math::Fixed32 targetZ, math::Fixed32 strength);

    [[nodiscard]] core::u32 chunkCount() const noexcept { return static_cast<core::u32>(_chunks.size()); }
    [[nodiscard]] const BodyChunk &chunk(core::u32 i) const { return _chunks[i]; }
    [[nodiscard]] BodyChunk &chunk(core::u32 i) { return _chunks[i]; }

    /**
     * @brief Total spring energy, as a stability measure.
     *
     * The number to watch: a correct integrator keeps it bounded, and a diverging
     * one shows here long before the body visibly explodes.
     *
     * @return Sum of squared extensions, weighted by stiffness.
     */
    [[nodiscard]] math::Fixed32 strainEnergy() const;

    /// @brief FNV-1a fold of every chunk, for determinism checks.
    [[nodiscard]] core::u32 fold() const;

private:
    lpl::pmr::vector<BodyChunk> _chunks;
    lpl::pmr::vector<BodyChunkConnection> _links;
};

/**
 * @struct TwoBoneSolution
 * @brief Where the middle joint goes so the end reaches the target.
 */
struct TwoBoneSolution {
    math::Fixed32 jointX{};
    math::Fixed32 jointZ{};
    bool reachable{false}; ///< False when the target is beyond the limb's extent.
};

/**
 * @brief Solves a two-bone limb: hip fixed, foot at the target, find the knee.
 *
 * Analytic rather than iterative, because there is a closed form and iteration
 * would spend a variable number of steps for a worse answer. The construction is
 * the intersection of two circles; the usual spelling uses @c acos, which is
 * forbidden in an authoritative path, so it is done with the projection identity
 * instead and one hardware square root.
 *
 * An unreachable target is REPORTED, not clamped silently. A limb quietly
 * snapping to full extension is how a creature ends up standing on nothing.
 *
 * @param rootX   Hip position.
 * @param rootZ   Hip position.
 * @param targetX Where the foot should land.
 * @param targetZ Where the foot should land.
 * @param upper   Length of the first bone.
 * @param lower   Length of the second bone.
 * @param flip    Which of the two mirror solutions to take (knee left or right).
 * @return The joint position, and whether the target was reachable.
 */
[[nodiscard]] TwoBoneSolution solveTwoBone(math::Fixed32 rootX, math::Fixed32 rootZ, math::Fixed32 targetX,
                                           math::Fixed32 targetZ, math::Fixed32 upper, math::Fixed32 lower,
                                           bool flip);

} // namespace lpl::ai

#endif // LPL_AI_SPRINGBODY_HPP
