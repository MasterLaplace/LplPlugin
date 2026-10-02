/**
 * @file Minimap.cpp
 * @brief Cutting a plane out of the coarsest level, and putting the eye on it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Minimap.hpp>

namespace lpl::voxel {

namespace {

/// The two axes a slice spans, in (u, v) order.
void spanAxes(SliceAxis axis, core::u32 &u, core::u32 &v, core::u32 &normal) noexcept
{
    switch (axis)
    {
    case SliceAxis::Z:
        // A rolled scroll's axis is the volume's slowest one, so this is the spiral seen end-on.
        normal = 0u;
        u = 2u; // x across
        v = 1u; // y down
        break;
    case SliceAxis::Y:
        normal = 1u;
        u = 2u;
        v = 0u;
        break;
    case SliceAxis::X:
    default:
        normal = 2u;
        u = 1u;
        v = 0u;
        break;
    }
}

[[nodiscard]] core::u8 sampleAt(const BrickMosaic &mosaic, core::i64 z, core::i64 y, core::i64 x, bool &found) noexcept
{
    const BrickView *brick = mosaic.find(z, y, x);
    if (brick == nullptr)
    {
        found = false;
        return 0u;
    }
    found = true;
    const core::u32 level = brick->key.level;
    const core::i64 span = brickSpanInBaseSamples(level);
    const core::i64 lz = (z - static_cast<core::i64>(brick->key.z) * span) >> level;
    const core::i64 ly = (y - static_cast<core::i64>(brick->key.y) * span) >> level;
    const core::i64 lx = (x - static_cast<core::i64>(brick->key.x) * span) >> level;
    const core::i64 edge = static_cast<core::i64>(kBrickEdge);
    if (lz < 0 || ly < 0 || lx < 0 || lz >= edge || ly >= edge || lx >= edge)
    {
        found = false;
        return 0u;
    }
    return brick->at(static_cast<core::u32>(lz), static_cast<core::u32>(ly), static_cast<core::u32>(lx));
}

} // namespace

core::u32 extractSlice(const BrickMosaic &mosaic, const VolumeGeometry &geometry, VolumeSlice &slice) noexcept
{
    if (!slice.valid() || !geometry.valid())
        return 0u;

    core::u32 u = 0u;
    core::u32 v = 0u;
    core::u32 normal = 0u;
    spanAxes(slice.axis, u, v, normal);

    // The slice always spans the WHOLE subject: a map that showed only part of it would be a
    // second view of where you already are.
    const core::i64 extentU = geometry.samples[u];
    const core::i64 extentV = geometry.samples[v];
    slice.lowU = 0;
    slice.lowV = 0;
    slice.stepU = extentU / static_cast<core::i64>(slice.width);
    slice.stepV = extentV / static_cast<core::i64>(slice.height);
    if (slice.stepU < 1)
        slice.stepU = 1;
    if (slice.stepV < 1)
        slice.stepV = 1;

    core::i64 at = slice.at;
    if (at < 0)
        at = 0;
    if (at >= geometry.samples[normal])
        at = geometry.samples[normal] - 1;

    core::u32 hits = 0u;
    for (core::u32 row = 0u; row < slice.height; ++row)
    {
        for (core::u32 col = 0u; col < slice.width; ++col)
        {
            core::i64 position[3]{0, 0, 0};
            position[normal] = at;
            position[u] = slice.lowU + static_cast<core::i64>(col) * slice.stepU;
            position[v] = slice.lowV + static_cast<core::i64>(row) * slice.stepV;

            bool found = false;
            const core::u8 value = sampleAt(mosaic, position[0], position[1], position[2], found);
            slice.samples[static_cast<core::usize>(row) * slice.width + col] = found ? value : 0u;
            if (found)
                ++hits;
        }
    }
    return hits;
}

void drawMinimap(core::u32 *pixels, core::u32 width, core::u32 height, const MinimapStyle &style,
                 const VolumeSlice &slice, const VolumeGeometry &geometry, const Eye &eye) noexcept
{
    if (pixels == nullptr || !slice.valid() || !geometry.valid())
        return;

    const core::u32 right = style.left + style.width;
    const core::u32 bottom = style.top + style.height;
    if (right > width || bottom > height)
        return;

    const auto put = [&](core::u32 x, core::u32 y, core::u32 colour) noexcept {
        if (x < width && y < height)
            pixels[static_cast<core::usize>(y) * width + x] = colour;
    };

    core::u32 u = 0u;
    core::u32 v = 0u;
    core::u32 normal = 0u;
    spanAxes(slice.axis, u, v, normal);

    // The slice's own contrast, stretched over what is actually in it. A fixed window would show
    // a flat grey on one volume and pure white on the next -- the same failure the main renderer's
    // density window exists to avoid, one panel smaller.
    core::u8 lowest = 255u;
    core::u8 highest = 0u;
    for (core::usize i = 0u; i < static_cast<core::usize>(slice.width) * slice.height; ++i)
    {
        const core::u8 s = slice.samples[i];
        if (s == 0u)
            continue; // Outside the subject; the mask, not a density.
        if (s < lowest)
            lowest = s;
        if (s > highest)
            highest = s;
    }
    const core::f32 span = highest > lowest ? static_cast<core::f32>(highest - lowest) : 1.0f;

    for (core::u32 py = 0u; py < style.height; ++py)
    {
        const core::u32 sy = slice.height * py / style.height;
        for (core::u32 px = 0u; px < style.width; ++px)
        {
            const core::u32 sx = slice.width * px / style.width;
            const core::u8 s = slice.samples[static_cast<core::usize>(sy) * slice.width + sx];
            core::u32 colour = style.background;
            if (s != 0u)
            {
                core::f32 t = (static_cast<core::f32>(s) - static_cast<core::f32>(lowest)) / span;
                if (t < 0.0f)
                    t = 0.0f;
                if (t > 1.0f)
                    t = 1.0f;
                t *= style.dim;
                const core::u32 grey = static_cast<core::u32>(t * 255.0f + 0.5f);
                colour = 0xFF000000u | (grey << 16) | (grey << 8) | grey;
            }
            put(style.left + px, style.top + py, colour);
        }
    }

    for (core::u32 px = 0u; px < style.width; ++px)
    {
        put(style.left + px, style.top, style.border);
        put(style.left + px, bottom - 1u, style.border);
    }
    for (core::u32 py = 0u; py < style.height; ++py)
    {
        put(style.left, style.top + py, style.border);
        put(right - 1u, style.top + py, style.border);
    }

    // Where the eye is, derived from the SAME geometry the march uses. Two answers to that is how
    // a map ends up confidently pointing at the wrong turn of a spiral and looking plausible.
    const core::f32 mps = geometry.metresPerSample();
    if (mps <= 0.0f)
        return;
    const core::f32 world[3]{eye.position.z / mps, eye.position.y / mps, eye.position.x / mps};
    const core::f32 forward[3]{eye.forward.z, eye.forward.y, eye.forward.x};

    const core::f32 spanU = static_cast<core::f32>(geometry.samples[u]);
    const core::f32 spanV = static_cast<core::f32>(geometry.samples[v]);
    if (spanU <= 0.0f || spanV <= 0.0f)
        return;
    const core::f32 fx = world[u] / spanU;
    const core::f32 fy = world[v] / spanV;
    // Clamped rather than dropped: an eye outside the subject is a real place to be -- you can fly
    // out of a scroll -- and a marker that vanished there would read as the map being broken.
    const core::f32 cx = static_cast<core::f32>(style.left) +
                         (fx < 0.0f ? 0.0f : (fx > 1.0f ? 1.0f : fx)) * static_cast<core::f32>(style.width - 1u);
    const core::f32 cy = static_cast<core::f32>(style.top) +
                         (fy < 0.0f ? 0.0f : (fy > 1.0f ? 1.0f : fy)) * static_cast<core::f32>(style.height - 1u);

    // The heading, projected into the slice. Without it the marker says where you are and not
    // which way you are facing, and in a spiral those are the same question.
    core::f32 du = forward[u];
    core::f32 dv = forward[v];
    const core::f32 len2 = du * du + dv * dv;

    // ⚠ **Looking ALONG the slice normal is the common case, not an edge case, and a heading that
    // projects to nothing must say so rather than disappear.** On a scroll the interesting
    // direction is down the axis -- which is exactly the direction this slice is cut across -- so
    // the arrow would vanish precisely when somebody is doing the normal thing. A ring means "into
    // or out of the page", and it is drawn instead of the line rather than beside it.
    if (len2 <= 0.05f)
    {
        for (core::i32 dy = -7; dy <= 7; ++dy)
        {
            for (core::i32 dx = -7; dx <= 7; ++dx)
            {
                const core::i32 r2 = dx * dx + dy * dy;
                if (r2 < 30 || r2 > 49)
                    continue;
                const core::f32 x = cx + static_cast<core::f32>(dx);
                const core::f32 y = cy + static_cast<core::f32>(dy);
                if (x < static_cast<core::f32>(style.left) || y < static_cast<core::f32>(style.top) ||
                    x >= static_cast<core::f32>(right) || y >= static_cast<core::f32>(bottom))
                    continue;
                put(static_cast<core::u32>(x), static_cast<core::u32>(y), style.marker);
            }
        }
    }
    else
    {
        core::f32 g = len2 > 1.0f ? len2 : 1.0f;
        for (int i = 0; i < 12; ++i)
            g = 0.5f * (g + len2 / g);
        du /= g;
        dv /= g;
        for (core::f32 step = 0.0f; step < style.coneLength; step += 0.5f)
        {
            const core::f32 x = cx + du * step;
            const core::f32 y = cy + dv * step;
            if (x < static_cast<core::f32>(style.left) || y < static_cast<core::f32>(style.top) ||
                x >= static_cast<core::f32>(right) || y >= static_cast<core::f32>(bottom))
                break;
            put(static_cast<core::u32>(x), static_cast<core::u32>(y), style.marker);
        }
    }

    for (core::i32 dy = -2; dy <= 2; ++dy)
    {
        for (core::i32 dx = -2; dx <= 2; ++dx)
        {
            if (dx * dx + dy * dy > 5)
                continue;
            const core::f32 x = cx + static_cast<core::f32>(dx);
            const core::f32 y = cy + static_cast<core::f32>(dy);
            if (x < static_cast<core::f32>(style.left) || y < static_cast<core::f32>(style.top) ||
                x >= static_cast<core::f32>(right) || y >= static_cast<core::f32>(bottom))
                continue;
            put(static_cast<core::u32>(x), static_cast<core::u32>(y), style.marker);
        }
    }
}

} // namespace lpl::voxel
