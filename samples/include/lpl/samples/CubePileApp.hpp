/**
 * @file CubePileApp.hpp
 * @brief The cube-pile sample packaged as an injectable engine application.
 *
 * Wraps lpl::samples::CubePile in the engine::IApplication seam so the engine
 * can drive it on any host. Everything the kernel's C client_app.c used to do
 * by hand lives here instead, engine-side and cross-platform: the orbit camera,
 * entity possession, the scale-to-surface blit and the HUD. The kernel keeps no
 * game logic — it only injects this payload.
 *
 * Authority split: CubePile::step is Fixed32 and authoritative (folded against
 * the Linux oracle); the camera, the blit and the HUD are float/pixel work and
 * are never folded, so nothing here can perturb the parity signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-17
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_CUBEPILEAPP_HPP
#    define LPL_SAMPLES_CUBEPILEAPP_HPP

#    include <lpl/core/Log.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/engine/Engine.hpp>
#    include <lpl/engine/IApplication.hpp>
#    include <lpl/image/Font8x16.hpp>
#    include <lpl/platform/IPlatform.hpp>
#    include <lpl/samples/CubePile.hpp>

#    include <new>

namespace lpl::samples {

/**
 * @class CubePileApp
 * @brief IApplication payload driving the CubePile simulation.
 */
class CubePileApp final : public engine::IApplication {
public:
    [[nodiscard]] core::Expected<void> init(engine::AppContext &context) override
    {
        _context = &context;
        sim().init();

        if (!context.platform.display().querySurface(_surface) || _surface.buffer == nullptr ||
            _surface.bitsPerPixel != 32u || _surface.width < 64u || _surface.height < 64u)
        {
            // Headless is legitimate (the server profile has no display): the
            // simulation still steps, only the render path is skipped.
            _hasSurface = false;
            core::Log::info("CubePileApp: no usable 32bpp surface, running headless");
            return {};
        }

        _hasSurface = true;
        core::Log::info("CubePileApp: WASD=cam Q/E=zoom P=possess N=next X=exit");
        return {};
    }

    /// Authoritative: fixed Fixed32 step. The engine calls this at a fixed rate.
    void fixedStep(core::f32 /*dt*/) override { sim().step(); }

    /// Non-authoritative: input, camera, rasterize, scale, HUD, present.
    void render(core::f64 /*alpha*/) override
    {
        drainInput();
        if (!_hasSurface)
            return;

        render::RenderTarget target{_color, _depth, kRenderWidth, kRenderHeight};
        sim().render(target, _camera);

        blitScaled();
        drawHud();

        _context->platform.display().present();
        ++_framesInWindow;
        updateFps();
    }

    void shutdown() override { core::Log::info("CubePileApp: exited"); }

    [[nodiscard]] const char *name() const noexcept override { return "CubePile"; }

private:
    /// Internal render resolution; the frame is scaled onto the display surface.
    static constexpr core::u32 kRenderWidth = 480u;
    static constexpr core::u32 kRenderHeight = 300u;

    /// The sim owns an ecs::Registry (heap-backed). Constructing it at static-init
    /// time would touch the kernel heap before kernel_main brings it up, so the
    /// storage sits in BSS and the object is placement-constructed on first use,
    /// which only happens once the engine is running.
    [[nodiscard]] static CubePile &sim() noexcept
    {
        static bool constructed = false;
        alignas(CubePile) static unsigned char storage[sizeof(CubePile)];
        static CubePile *instance = nullptr;
        if (!constructed)
        {
            instance = new (static_cast<void *>(storage)) CubePile();
            constructed = true;
        }
        return *instance;
    }

    void drainInput() noexcept
    {
        char key;
        while (_context->platform.input().tryPopCharacter(key))
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
            case 27: _context->engine.requestShutdown(); break;
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
    void updateFps() noexcept
    {
        platform::IClockBackend &clock = _context->platform.clock();
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

    /// Colour/depth targets live in BSS, not in the object: at 480x300 they are
    /// ~1.1 MiB together, far too much to put on a kernel heap allocation (or a
    /// stack frame). Only one CubePileApp is ever live, so static storage is safe.
    static inline core::u32 _color[kRenderWidth * kRenderHeight]{};
    static inline core::f32 _depth[kRenderWidth * kRenderHeight]{};

    engine::AppContext *_context{nullptr};
    platform::SurfaceDescriptor _surface{};
    bool _hasSurface{false};
    CubePile::Camera _camera{};

    core::u32 _fps{0};
    core::u32 _framesInWindow{0};
    core::u32 _windowStart{0};
};

} // namespace lpl::samples

#endif // LPL_SAMPLES_CUBEPILEAPP_HPP
