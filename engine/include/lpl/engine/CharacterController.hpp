/**
 * @file CharacterController.hpp
 * @brief A body that walks the world: gravity, ground, slopes, and a jump.
 *
 * The thing a free camera is not. Flying the view around a heightfield proves the
 * terrain is there; it does not put you IN it — you pass through hills, you do not
 * fall, and nothing about the shape of the ground is something you can feel. What
 * every game with a walker needs on its first day is a body: it is pulled down, it
 * stands on what is under it, it cannot climb a cliff, and it can leave the ground
 * on purpose.
 *
 * Two decisions define this file.
 *
 * AUTHORITATIVE, therefore Fixed32 throughout — position, velocity AND heading. The
 * heading is the one that is easy to get wrong: turning is a mouse gesture, mouse
 * deltas are naturally float, and letting a float yaw pick the walk direction would
 * make the player's POSITION float-derived. The whole determinism contract dies
 * quietly there, because nothing looks wrong on one machine. So the controller owns
 * the yaw as Fixed32 and turns it through CORDIC; the camera READS that yaw to
 * build its matrices, which is the allowed direction — authoritative state feeds
 * presentation, never the reverse.
 *
 * IT KNOWS NOTHING ABOUT TERRAIN. The world arrives as one callback answering, for a
 * world cell and a height, the GAP a body would occupy there — its floor and its
 * ceiling. A streamed heightfield, a bounded grid, a flat plane and a unit test all
 * satisfy it, which is why this is engine and not sample.
 *
 * That callback used to return a single height, and the change is the one thing in
 * this file worth arguing about. A height per column is a complete description of a
 * world only while a column has one surface; under a hill there is ground beneath a
 * body and rock above it, and the body can be between them. Adding a second callback
 * for the ceiling would have been the smaller diff and the wrong shape — a floor and
 * the ceiling over it are ONE fact about ONE gap, and two calls could return halves of
 * two different gaps in a stack of galleries. Callers with no caves in them wrap their
 * height function in @ref procgen::surfaceSpan and nothing else changes.
 *
 * What it deliberately does NOT do: collide against other entities (that is the
 * broad-phase's job, and a character that resolved its own would resolve them
 * twice), or move the camera. It produces a position; who looks from it is the
 * caller's business.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-31
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_CHARACTER_CONTROLLER_HPP
#    define LPL_ENGINE_CHARACTER_CONTROLLER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedMath.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/VerticalSpan.hpp>

namespace lpl::engine {

/**
 * @struct CharacterParams
 * @brief How the body moves. Every value is authoritative, so every one is Fixed32.
 *
 * The defaults are a person, in world cells where one cell is one metre: a walk of
 * six metres a second is a jog, gravity is a little stronger than Earth's because
 * game gravity always is (real gravity makes a jump feel like a balloon), and a
 * jump that clears about a metre and a quarter.
 */
struct CharacterParams {
    math::Fixed32 walkSpeed{math::Fixed32::fromFloat(6.0f)};       ///< Cells per second.
    math::Fixed32 sprintScale{math::Fixed32::fromFloat(1.9f)};     ///< Multiplier while sprinting.
    math::Fixed32 acceleration{math::Fixed32::fromFloat(38.0f)};   ///< Cells per second squared.
    math::Fixed32 groundFriction{math::Fixed32::fromFloat(30.0f)}; ///< Deceleration with no input.
    /**
     * @brief Share of ground acceleration available in mid-air.
     *
     * Not zero, and not one. At zero a jump is committed the instant it starts,
     * which reads as ice; at one the air is as good as the ground and the jump
     * stops meaning anything. A third is the usual answer.
     */
    math::Fixed32 airControl{math::Fixed32::fromFloat(0.34f)};

    math::Fixed32 gravity{math::Fixed32::fromFloat(22.0f)};      ///< Cells per second squared, downward.
    math::Fixed32 terminalFall{math::Fixed32::fromFloat(55.0f)}; ///< Fastest downward speed.
    math::Fixed32 jumpSpeed{math::Fixed32::fromFloat(7.4f)};     ///< Upward speed at take-off.

    math::Fixed32 eyeHeight{math::Fixed32::fromFloat(1.7f)}; ///< Eye above the feet.
    /**
     * @brief How tall the body is, for deciding what it fits under.
     *
     * Its own number rather than the eye height, and a little above it: a person's eyes
     * are not at the top of their head, and a doorway sized to the eye is a doorway
     * every walker scrapes through. Only read where there is a ceiling, so a surface
     * world is unaffected by it.
     */
    math::Fixed32 standHeight{math::Fixed32::fromFloat(1.8f)};
    /**
     * @brief Rise a single step may absorb without being a wall.
     *
     * A heightfield is a staircase: a cell boundary where the terrain rises by half
     * a metre is not a cliff, it is a kerb, and a body that refused it would be
     * stopped by scenery it should walk over. Anything above this is a wall.
     */
    math::Fixed32 stepHeight{math::Fixed32::fromFloat(0.75f)};
    /**
     * @brief Steepest ground the body can stand on, as a rise per cell walked.
     *
     * Above it, gravity wins and the body slides. Expressed as a slope rather than
     * an angle because computing an angle needs an arctangent, and comparing rise
     * against run needs a multiply.
     */
    math::Fixed32 maxSlope{math::Fixed32::fromFloat(1.30f)};
    math::Fixed32 slideSpeed{math::Fixed32::fromFloat(4.5f)}; ///< Downhill drift on ground too steep.

    /**
     * @brief Ticks after walking off a ledge during which a jump still works.
     *
     * "Coyote time". Without it a player who presses jump one tick after leaving the
     * edge gets nothing, and reports it as the jump not registering — they are not
     * wrong, they pressed it when they were visibly at the edge. Six ticks at 60 Hz
     * is a tenth of a second: unnoticeable as a cheat, decisive as a fix.
     */
    core::u32 coyoteTicks{6u};
    /**
     * @brief Ticks a jump pressed in mid-air is remembered for.
     *
     * The mirror of coyote time, for the other end of the same mistake: pressing
     * jump just BEFORE landing. Without it the input is dropped and the player has
     * to press again on exactly the right frame.
     */
    core::u32 jumpBufferTicks{6u};
};

/**
 * @struct CharacterIntent
 * @brief What the player is asking for this tick, in the body's own frame.
 *
 * Scalars in [-1, 1] rather than a world direction, because the body owns the
 * heading: handing it a world vector would mean the caller had already resolved the
 * yaw, and the caller's yaw is the float one belonging to a camera.
 */
struct CharacterIntent {
    math::Fixed32 forward{}; ///< +1 walks toward the heading, -1 backs away.
    math::Fixed32 strafe{};  ///< +1 sidesteps right.
    math::Fixed32 turn{};    ///< Radians added to the heading this tick.
    bool jump{false};        ///< Jump pressed THIS tick (an edge, not a hold).
    bool sprint{false};
};

/**
 * @class CharacterController
 * @brief An authoritative walking body on a height callback.
 */
class CharacterController {
public:
    /**
     * @brief Places the body, snapped onto whatever it comes to rest on.
     *
     * @param startY  Where to look from. It matters the moment a column has more than
     *                one floor: dropped in at the surface a body lands on the hill,
     *                dropped in inside a gallery it lands on the gallery. A caller that
     *                has no caves passes the ground height and the argument does nothing.
     * @param spaceAt (worldX, worldZ, y) -> procgen::VerticalSpan.
     */
    template <typename SpaceAt>
    void placeAt(math::Fixed32 worldX, math::Fixed32 worldZ, math::Fixed32 startY, SpaceAt &&spaceAt) noexcept;

    /**
     * @brief One authoritative tick: intent, gravity, ground, slope, jump.
     *
     * @param dt      Fixed timestep. FIXED is not a detail: the whole point of a
     *                deterministic step is that it is the same length every time,
     *                and a controller integrated with a frame-dependent dt puts the
     *                player somewhere else on a faster machine.
     * @param spaceAt (worldX, worldZ, y) -> procgen::VerticalSpan. Integer cells: the
     *                heightfield IS defined per cell, and pretending to interpolate
     *                would invent a surface the renderer does not draw.
     */
    template <typename SpaceAt>
    void step(const CharacterParams &params, const CharacterIntent &intent, math::Fixed32 dt, SpaceAt &&spaceAt);

    [[nodiscard]] math::Fixed32 x() const noexcept { return _x; }
    [[nodiscard]] math::Fixed32 y() const noexcept { return _y; } ///< Feet, not eye.
    [[nodiscard]] math::Fixed32 z() const noexcept { return _z; }
    [[nodiscard]] math::Fixed32 yaw() const noexcept { return _yaw; }
    [[nodiscard]] math::Fixed32 verticalSpeed() const noexcept { return _vy; }

    /** @brief Horizontal speed, for a readout or a walk cycle. */
    [[nodiscard]] math::Fixed32 groundSpeed() const noexcept { return math::fixedSqrt(_vx * _vx + _vz * _vz); }

    [[nodiscard]] bool isGrounded() const noexcept { return _grounded; }
    [[nodiscard]] bool isSliding() const noexcept { return _sliding; }
    /**
     * @brief Whether there was rock over the body on the last tick.
     *
     * Authoritative, and therefore folded: it is what stops the body sliding, so two
     * targets that disagreed about it would put the walker in two places. A renderer
     * reads it to decide there is no sky, which is the allowed direction.
     */
    [[nodiscard]] bool isEnclosed() const noexcept { return _enclosed; }
    /** @brief Underside of the rock over the body; procgen::openSky under the open sky. */
    [[nodiscard]] math::Fixed32 ceilingHeight() const noexcept { return _ceiling; }
    /** @brief Times a move was refused because the gap ahead was too low to enter. */
    [[nodiscard]] core::u32 duckedCount() const noexcept { return _ducked; }
    /** @brief Times the body's head met a ceiling. */
    [[nodiscard]] core::u32 headBumpCount() const noexcept { return _headBumps; }
    /** @brief Ticks since the body last touched ground; 0 while standing. */
    [[nodiscard]] core::u32 airborneTicks() const noexcept { return _airborneTicks; }
    /** @brief Jumps taken since the body was placed — a cheap liveness readout. */
    [[nodiscard]] core::u32 jumpCount() const noexcept { return _jumps; }
    /** @brief Times a move was refused because the rise ahead was a wall. */
    [[nodiscard]] core::u32 blockedCount() const noexcept { return _blocked; }

    void setYaw(math::Fixed32 radians) noexcept { _yaw = radians; }

    /**
     * @brief FNV-1a over the authoritative state.
     *
     * The body is simulation, so it folds. Position, velocity, heading and the two
     * booleans that decide what the next tick does — nothing presentational, or the
     * signature would move when the camera did.
     */
    [[nodiscard]] core::u32 fold(core::u32 seed = 0x811C9DC5u) const noexcept;

private:
    /** @brief Advances one axis, refusing the move if it walks into a wall or under a low roof. */
    template <typename SpaceAt>
    void moveAxis(const CharacterParams &params, math::Fixed32 &coordinate, math::Fixed32 delta, SpaceAt &&spaceAt);

    math::Fixed32 _x{};
    math::Fixed32 _y{};
    math::Fixed32 _z{};
    math::Fixed32 _vx{};
    math::Fixed32 _vy{};
    math::Fixed32 _vz{};
    math::Fixed32 _yaw{};
    math::Fixed32 _groundHeight{};
    math::Fixed32 _ceiling{procgen::openSky()};
    bool _grounded{false};
    bool _sliding{false};
    bool _enclosed{false};
    core::u32 _coyote{0u};
    core::u32 _jumpBuffer{0u};
    core::u32 _airborneTicks{0u};
    core::u32 _jumps{0u};
    core::u32 _blocked{0u};
    core::u32 _ducked{0u};
    core::u32 _headBumps{0u};
};

} // namespace lpl::engine

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/engine/CharacterController.inl>

#endif // LPL_ENGINE_CHARACTER_CONTROLLER_HPP
