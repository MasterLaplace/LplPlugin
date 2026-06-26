/**
 * @file KernelDisplayRenderer.cpp
 * @brief Software renderer implementation for the kernel linear framebuffer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#if LPL_TARGET_KERNEL

#    include <lpl/render/kernel/KernelDisplayRenderer.hpp>

#    include <lpl/math/Cordic.hpp>

namespace lpl::render::kernel {

// ---------------------------------------------------------------------------
// Triangle geometry constants (Fixed32 authority).
// ---------------------------------------------------------------------------

// Rotation increment per tick: 2° per tick (π / 90), i.e. one full revolution per 180 ticks.
// Fixed32::fromInt(n) creates the integer n in Q16.16 (raw = n << 16), so this equals π/90.
static constexpr math::Fixed32 kDeltaAngle = math::Fixed32::pi() / math::Fixed32::fromInt(90);

// Base equilateral triangle vertices in Fixed32 normalised space [-1, 1].
// Inscribed in radius 0.45:  V_k = (r·cos(90° + 120°·k), r·sin(90° + 120°·k))
// In Q16.16: 0.45 ≈ 29491, cos(30°) ≈ 56756, sin(30°) = 32768
//   V0 =  (0,        29491)         top
//   V1 =  (29491·cos210°, 29491·sin210°) = (-25527, -14746)
//   V2 =  (29491·cos330°, 29491·sin330°) = ( 25527, -14746)
static const math::Fixed32 kBaseX[3] = {
    math::Fixed32{0},
    math::Fixed32::fromRaw(-25527),
    math::Fixed32::fromRaw(25527),
};
static const math::Fixed32 kBaseY[3] = {
    math::Fixed32::fromRaw(29491),
    math::Fixed32::fromRaw(-14746),
    math::Fixed32::fromRaw(-14746),
};

// Per-vertex colours (0x00RRGGBB)
static constexpr core::u32 kVertexColor[3] = {0x00FF4040u, 0x0040FF40u, 0x004040FFu};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline core::i32 edgeFunction(core::i32 ax, core::i32 ay, core::i32 bx, core::i32 by, core::i32 px,
                                     core::i32 py) noexcept
{
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// ---------------------------------------------------------------------------
// KernelDisplayRenderer
// ---------------------------------------------------------------------------

KernelDisplayRenderer::KernelDisplayRenderer(platform::IDisplayBackend &display) noexcept : _display{display} {}

void KernelDisplayRenderer::tick() noexcept
{
    _angle = _angle + kDeltaAngle;
    // Wrap at 2π using subtraction to avoid accumulated drift.
    const math::Fixed32 twoPi = math::Fixed32::pi() + math::Fixed32::pi();
    while (_angle >= twoPi)
        _angle = _angle - twoPi;
}

core::Expected<void> KernelDisplayRenderer::init(core::u32 /*width*/, core::u32 /*height*/)
{
    _display.querySurface(_surface);
    _initialized = (_surface.buffer != nullptr);
    return {};
}

void KernelDisplayRenderer::beginFrame()
{
    if (!_initialized)
        return;
    _display.querySurface(_surface);
    // Clear to dark navy background.
    _display.clear(0x00001040u);
}

void KernelDisplayRenderer::endFrame()
{
    if (!_initialized)
        return;
    drawTriangle();
    _display.present();
}

void KernelDisplayRenderer::resize(core::u32 /*width*/, core::u32 /*height*/) { _display.querySurface(_surface); }

void KernelDisplayRenderer::shutdown() { _initialized = false; }

const char *KernelDisplayRenderer::name() const noexcept { return "KernelDisplayRenderer"; }

void KernelDisplayRenderer::drawTriangle() noexcept
{
    if (!_surface.buffer || _surface.width == 0u || _surface.height == 0u)
        return;

    // --- Rotate base vertices (Fixed32 authority) --------------------------
    math::Fixed32 sinA{0}, cosA{0};
    math::Cordic::sincos(_angle, sinA, cosA);

    math::Fixed32 rx[3], ry[3];
    for (int i = 0; i < 3; ++i)
    {
        rx[i] = cosA * kBaseX[i] - sinA * kBaseY[i];
        ry[i] = sinA * kBaseX[i] + cosA * kBaseY[i];
    }

    // --- Project to screen coordinates (float rasterisation) ---------------
    const float hw = static_cast<float>(_surface.width) * 0.5f;
    const float hh = static_cast<float>(_surface.height) * 0.5f;

    core::i32 sx[3], sy[3];
    for (int i = 0; i < 3; ++i)
    {
        sx[i] = static_cast<core::i32>(hw + rx[i].toFloat() * hw);
        sy[i] = static_cast<core::i32>(hh - ry[i].toFloat() * hh); // Y flipped
    }

    // --- Bounding box (clamped to surface) ----------------------------------
    core::i32 minX = sx[0], maxX = sx[0];
    core::i32 minY = sy[0], maxY = sy[0];
    for (int i = 1; i < 3; ++i)
    {
        if (sx[i] < minX)
            minX = sx[i];
        if (sx[i] > maxX)
            maxX = sx[i];
        if (sy[i] < minY)
            minY = sy[i];
        if (sy[i] > maxY)
            maxY = sy[i];
    }
    const core::i32 W = static_cast<core::i32>(_surface.width);
    const core::i32 H = static_cast<core::i32>(_surface.height);
    if (minX < 0)
        minX = 0;
    if (minY < 0)
        minY = 0;
    if (maxX >= W)
        maxX = W - 1;
    if (maxY >= H)
        maxY = H - 1;

    // --- Rasterise (barycentric edge-function fill) -------------------------
    const core::u32 pitchPixels = _surface.pitch / 4u;

    // Total triangle area (for barycentric weight normalisation).
    const core::i32 area = edgeFunction(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
    if (area == 0)
        return;
    const float rcpArea = 1.0f / static_cast<float>(area);

    for (core::i32 py = minY; py <= maxY; ++py)
    {
        for (core::i32 px = minX; px <= maxX; ++px)
        {
            const core::i32 w0 = edgeFunction(sx[1], sy[1], sx[2], sy[2], px, py);
            const core::i32 w1 = edgeFunction(sx[2], sy[2], sx[0], sy[0], px, py);
            const core::i32 w2 = edgeFunction(sx[0], sy[0], sx[1], sy[1], px, py);

            // Accept CW and CCW: require all weights same sign as the area.
            if (area > 0 && (w0 < 0 || w1 < 0 || w2 < 0))
                continue;
            if (area < 0 && (w0 > 0 || w1 > 0 || w2 > 0))
                continue;

            // Barycentric blend of the three vertex colours.
            const float b0 = static_cast<float>(w0) * rcpArea;
            const float b1 = static_cast<float>(w1) * rcpArea;
            const float b2 = static_cast<float>(w2) * rcpArea;

            const auto r0 = static_cast<float>((kVertexColor[0] >> 16) & 0xFF);
            const auto g0 = static_cast<float>((kVertexColor[0] >> 8) & 0xFF);
            const auto b_0 = static_cast<float>(kVertexColor[0] & 0xFF);

            const auto r1 = static_cast<float>((kVertexColor[1] >> 16) & 0xFF);
            const auto g1 = static_cast<float>((kVertexColor[1] >> 8) & 0xFF);
            const auto b_1 = static_cast<float>(kVertexColor[1] & 0xFF);

            const auto r2 = static_cast<float>((kVertexColor[2] >> 16) & 0xFF);
            const auto g2 = static_cast<float>((kVertexColor[2] >> 8) & 0xFF);
            const auto b_2 = static_cast<float>(kVertexColor[2] & 0xFF);

            const auto R = static_cast<core::u32>(b0 * r0 + b1 * r1 + b2 * r2);
            const auto G = static_cast<core::u32>(b0 * g0 + b1 * g1 + b2 * g2);
            const auto B = static_cast<core::u32>(b0 * b_0 + b1 * b_1 + b2 * b_2);

            _surface.buffer[static_cast<core::u32>(py) * pitchPixels + static_cast<core::u32>(px)] =
                (R << 16) | (G << 8) | B;
        }
    }
}

} // namespace lpl::render::kernel

#endif // LPL_TARGET_KERNEL
