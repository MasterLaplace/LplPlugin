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
 * engine, by @ref runCubePileAndFold for the oracle/kernel-smoke path).
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
    /// The cubes run on the World's own registry — that is the whole point of
    /// the World seam. registry() refers to the base subobject, constructed
    /// before this member, so the reference is valid.
    CubePileWorld() : _cube{registry()} {}

    [[nodiscard]] core::Expected<void> onInit(engine::WorldContext &context) override
    {
        // Seeds the entities only. Config::enablePhysics (set by CubePileWorld's
        // hosts) has already registered the engine's built-in PhysicsSystem on
        // this same scheduler/registry before onInit runs — that is what
        // advances Position/Velocity from here on. Registering a second,
        // game-owned physics system here would step every entity twice per
        // tick, on the same chunk buffers, silently doubling gravity/damping/
        // collision impulses.
        _cube.init();

        if (!context.platform.display().querySurface(_surface) || _surface.buffer == nullptr ||
            _surface.bitsPerPixel != 32u || _surface.width < 64u || _surface.height < 64u)
        {
            // Headless is legitimate (the server profile has no display): the
            // simulation still steps, only the render path is skipped.
            _hasSurface = false;
            core::Log::info("CubePileWorld: no usable 32bpp surface, running headless");
            return {};
        }

        _hasSurface = true;
        core::Log::info("CubePileWorld: WASD=cam Q/E=zoom P=possess N=next X=exit");
        return {};
    }

    /// Non-authoritative: input, camera, rasterize, scale, HUD, present.
    void onRender(engine::WorldContext &context, core::f64 /*alpha*/) override
    {
        drainInput(context);
        if (!_hasSurface)
            return;

        render::RenderTarget target{_color, _depth, kRenderWidth, kRenderHeight};
        _cube.render(target, _camera);

        blitScaled();
        drawHud();

        context.platform.display().present();
        ++_framesInWindow;
        updateFps(context);
    }

    void onShutdown() override { core::Log::info("CubePileWorld: exited"); }

    [[nodiscard]] const char *name() const noexcept override { return "CubePile"; }

private:
    /// Internal render resolution; the frame is scaled onto the display surface.
    static constexpr core::u32 kRenderWidth = 480u;
    static constexpr core::u32 kRenderHeight = 300u;

    void drainInput(engine::WorldContext &context) noexcept
    {
        char key;
        while (context.platform.input().tryPopCharacter(key))
        {
            switch (key)
            {
            case 'a': _camera.yaw -= 0.08f; break;
            case 'd': _camera.yaw += 0.08f; break;
            case 'w': _camera.pitch = clampF(_camera.pitch + 0.06f, -1.40f, 1.40f); break;
            case 's': _camera.pitch = clampF(_camera.pitch - 0.06f, -1.40f, 1.40f); break;
            case 'q': _camera.dist = clampF(_camera.dist - 0.6f, 2.0f, 40.0f); break;
            case 'e': _camera.dist = clampF(_camera.dist + 0.6f, 2.0f, 40.0f); break;
            case 'p': _camera.possess = (_camera.possess < 0) ? 0 : -1; break;
            case 'n':
                if (_camera.possess >= 0)
                    _camera.possess =
                        static_cast<core::i32>((static_cast<core::u32>(_camera.possess) + 1u) % CubePile::count());
                break;
            case 'x':
            case 27:
                if (context.engine != nullptr)
                    context.engine->requestShutdown();
                break;
            default: break;
            }
        }
    }

    [[nodiscard]] static core::f32 clampF(core::f32 value, core::f32 low, core::f32 high) noexcept
    {
        return value < low ? low : (value > high ? high : value);
    }

    /// Nearest-neighbour scale of the engine frame onto the display surface.
    void blitScaled() const noexcept
    {
        const core::u32 pitchPixels = _surface.pitch / 4u;
        for (core::u32 dy = 0u; dy < _surface.height; ++dy)
        {
            const core::u32 *sourceRow = &_color[((dy * kRenderHeight) / _surface.height) * kRenderWidth];
            core::u32 *destinationRow = &_surface.buffer[dy * pitchPixels];
            for (core::u32 dx = 0u; dx < _surface.width; ++dx)
                destinationRow[dx] = sourceRow[(dx * kRenderWidth) / _surface.width];
        }
    }

    void drawHud() const noexcept
    {
        const core::u32 pitchPixels = _surface.pitch / 4u;
        char line[64];
        core::u32 pos = 0u;

        pos = 0u;
        appendString(line, pos, "FPS: ");
        appendU32(line, pos, _fps);
        line[pos] = '\0';
        image::drawText8x16(_surface.buffer, pitchPixels, 8u, 8u, line, 0x00FFFF66u);

        pos = 0u;
        appendString(line, pos, "ENTITIES: ");
        appendU32(line, pos, CubePile::count());
        line[pos] = '\0';
        image::drawText8x16(_surface.buffer, pitchPixels, 8u, 26u, line, 0x00A0C0FFu);

        pos = 0u;
        if (_camera.possess < 0)
        {
            appendString(line, pos, "POSSESS: none");
        }
        else
        {
            appendString(line, pos, "POSSESS: #");
            appendU32(line, pos, static_cast<core::u32>(_camera.possess));
        }
        line[pos] = '\0';
        image::drawText8x16(_surface.buffer, pitchPixels, 8u, 44u, line, 0x0060FF80u);

        image::drawText8x16(_surface.buffer, pitchPixels, 8u, _surface.height - 20u,
                            "WASD=cam Q/E=zoom P=possess N=next X=exit", 0x00808890u);
    }

    /// FPS over a one-second wall-clock window, read from the platform clock.
    void updateFps(engine::WorldContext &context) noexcept
    {
        platform::IClockBackend &clock = context.platform.clock();
        const core::u32 tickHertz = clock.tickHertz();
        const core::u32 now = clock.tickCount();
        const core::u32 elapsed = now - _windowStart; // wrap-safe modular delta

        if (tickHertz != 0u && elapsed >= tickHertz)
        {
            _fps = (_framesInWindow * tickHertz) / elapsed;
            _framesInWindow = 0u;
            _windowStart = now;
        }
    }

    static void appendU32(char *buffer, core::u32 &pos, core::u32 value) noexcept
    {
        char reversed[10];
        core::u32 length = 0u;
        if (value == 0u)
            reversed[length++] = '0';
        while (value != 0u)
        {
            reversed[length++] = static_cast<char>('0' + (value % 10u));
            value /= 10u;
        }
        while (length != 0u)
            buffer[pos++] = reversed[--length];
    }

    static void appendString(char *buffer, core::u32 &pos, const char *text) noexcept
    {
        for (core::u32 i = 0u; text[i] != '\0'; ++i)
            buffer[pos++] = text[i];
    }

    /// Colour/depth targets in BSS, not in the object: ~1.1 MiB at 480x300, far
    /// past a kernel heap allocation or stack frame. Only one CubePileWorld is
    /// ever live, so static storage is safe.
    static inline core::u32 _color[kRenderWidth * kRenderHeight]{};
    static inline core::f32 _depth[kRenderWidth * kRenderHeight]{};

    CubePile _cube; ///< The sim, on this World's registry.
    platform::SurfaceDescriptor _surface{};
    bool _hasSurface{false};
    CubePile::Camera _camera{};

    core::u32 _fps{0};
    core::u32 _framesInWindow{0};
    core::u32 _windowStart{0};
};

} // namespace lpl::samples

#endif // LPL_SAMPLES_CUBEPILEWORLD_HPP
