#ifndef LPL_SAMPLES_CUBEPILEWORLD_INL
#define LPL_SAMPLES_CUBEPILEWORLD_INL

namespace lpl::samples {

inline core::Expected<void> CubePileWorld::onInit(engine::WorldContext &context)
{
    _cube.init();

    if (!context.platform.display().querySurface(_surface) || _surface.buffer == nullptr ||
        _surface.bitsPerPixel != 32u || _surface.width < 64u || _surface.height < 64u)
    {
        _hasSurface = false;
        core::Log::info("CubePileWorld: no usable 32bpp surface, running headless");
        return {};
    }

    _hasSurface = true;
    core::Log::info("CubePileWorld: WASD=cam Q/E=zoom P=possess N=next X=exit");
    return {};
}

inline void CubePileWorld::onRender(engine::WorldContext &context, core::f64 /*alpha*/)
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

inline void CubePileWorld::drainInput(engine::WorldContext &context) noexcept
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

inline void CubePileWorld::blitScaled() const noexcept
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

inline void CubePileWorld::drawHud() const noexcept
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

inline void CubePileWorld::updateFps(engine::WorldContext &context) noexcept
{
    platform::IClockBackend &clock = context.platform.clock();
    const core::u32 tickHertz = clock.tickHertz();
    const core::u32 now = clock.tickCount();
    const core::u32 elapsed = now - _windowStart;

    if (tickHertz != 0u && elapsed >= tickHertz)
    {
        _fps = (_framesInWindow * tickHertz) / elapsed;
        _framesInWindow = 0u;
        _windowStart = now;
    }
}

inline void CubePileWorld::appendU32(char *buffer, core::u32 &pos, core::u32 value) noexcept
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

inline void CubePileWorld::appendString(char *buffer, core::u32 &pos, const char *text) noexcept
{
    for (core::u32 i = 0u; text[i] != '\0'; ++i)
        buffer[pos++] = text[i];
}

} // namespace lpl::samples

#endif // LPL_SAMPLES_CUBEPILEWORLD_INL
