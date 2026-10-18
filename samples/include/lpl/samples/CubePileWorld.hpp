/**
 * @file CubePileWorld.hpp
 * @brief The cube-pile sample as an injectable engine World.
 *
 * A concrete engine::World: its 1024 cubes live in the World's own registry
 * (the game state IS the World's state), and it carries everything the sample
 * needs on top — the orbit camera, entity possession, the scale-to-surface blit
 * and the HUD. The engine hosts it with no knowledge of the game; a server would
 * host many Worlds like it.
 *
 * Authority split: Position/Velocity/AABB/Mass are advanced by the engine's own
 * built-in @c systems::PhysicsSystem (selected by @c Config::enablePhysics,
 * which CubePileWorld's hosts turn on) — the same generic system legacy's
 * client registered via @c core.registerSystem(Systems::PhysicsSystem()), and
 * the one the server profile already used by default. CubePileWorld does not
 * step its own copy: it only seeds the entities (@ref CubePile::init) and reads
 * them back for rendering. onRender is float/pixel work and is never folded, so
 * nothing here perturbs the parity signature (folded separately, off the
 * engine, by the engine test `cube_pile`).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-17
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_CUBEPILEWORLD_HPP
#    define LPL_SAMPLES_CUBEPILEWORLD_HPP

#    include <lpl/core/Log.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/engine/Engine.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/image/Font8x16.hpp>
#    include <lpl/platform/IPlatform.hpp>
#    include <lpl/samples/CubePile.hpp>

namespace lpl::samples {

/**
 * @class CubePileWorld
 * @brief engine::World running the CubePile simulation.
 */
class CubePileWorld final : public engine::World {
public:
    /**
     * @brief The cubes run on the World's own registry — that is the whole point of
     * the World seam. registry() refers to the base subobject, constructed
     * before this member, so the reference is valid.
     */
    CubePileWorld() : _cube{registry()} {}

    /**
     * @brief Seeds the entities, and finds the surface to draw on.
     *
     * Seeds the entities only. Config::enablePhysics (set by CubePileWorld's
     * hosts) has already registered the engine's built-in PhysicsSystem on
     * this same scheduler/registry before onInit runs — that is what
     * advances Position/Velocity from here on. Registering a second,
     * game-owned physics system here would step every entity twice per
     * tick, on the same chunk buffers, silently doubling gravity/damping/
     * collision impulses.
     *
     * Headless is legitimate (the server profile has no display): the
     * simulation still steps, only the render path is skipped.
     */
    [[nodiscard]] inline core::Expected<void> onInit(engine::WorldContext &context) override;

    /**
     * @brief Non-authoritative: input, camera, rasterize, scale, HUD, present.
     */
    inline void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override;

    void onShutdown() override { core::Log::info("CubePileWorld: exited"); }

    [[nodiscard]] const char *name() const noexcept override { return "CubePile"; }

private:
    /**
     * @name Internal render resolution; the frame is scaled onto the display surface
     * @{
     */
    static constexpr core::u32 kRenderWidth = 480u;
    static constexpr core::u32 kRenderHeight = 300u;
    /** @} */

    void drainInput(engine::WorldContext &context) noexcept;

    [[nodiscard]] static core::f32 clampF(core::f32 value, core::f32 low, core::f32 high) noexcept
    {
        return value < low ? low : (value > high ? high : value);
    }

    /**
     * @brief Nearest-neighbour scale of the engine frame onto the display surface.
     */
    void blitScaled() const noexcept;

    void drawHud() const noexcept;

    /**
     * @brief FPS over a one-second wall-clock window, read from the platform clock.
     *
     * The elapsed ticks are a wrap-safe modular delta.
     */
    void updateFps(engine::WorldContext &context) noexcept;

    static void appendU32(char *buffer, core::u32 &pos, core::u32 value) noexcept;

    static void appendString(char *buffer, core::u32 &pos, const char *text) noexcept;

    /**
     * @name Colour/depth targets in BSS, not in the object
     *
     * ~1.1 MiB at 480x300, far past a kernel heap allocation or stack frame. Only one
     * CubePileWorld is ever live, so static storage is safe.
     * @{
     */
    static inline core::u32 _color[kRenderWidth * kRenderHeight]{};
    static inline core::f32 _depth[kRenderWidth * kRenderHeight]{};
    /** @} */

    CubePile _cube; /**< The sim, on this World's registry. */
    platform::SurfaceDescriptor _surface{};
    bool _hasSurface{false};
    CubePile::Camera _camera{};

    core::u32 _fps{0};
    core::u32 _framesInWindow{0};
    core::u32 _windowStart{0};
};

} // namespace lpl::samples

#    include "CubePileWorld.inl"

#endif // LPL_SAMPLES_CUBEPILEWORLD_HPP
