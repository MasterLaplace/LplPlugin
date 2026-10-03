/**
 * @file Minimap.cpp
 * @brief Cutting a plane out of the mosaic the caller passes, and putting the eye on it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/std/cmath.hpp>
#include <lpl/voxel/Minimap.hpp>

#include <algorithm>
#include <array>
#include <optional>

namespace lpl::voxel {

namespace {

/// The heading is drawn as a ring when the squared length of its projection into the slice is at most this.
constexpr core::f32 kRingInPlaneLengthSquared = 0.05f;

/// The ring around the dot, from 5.5 to 7 pixels out.
constexpr core::i32 kRingInnerRadiusSquared = 30;
constexpr core::i32 kRingOuterRadiusSquared = 49;

/// The dot that marks the eye, about 2.2 pixels in radius.
constexpr core::i32 kDotRadiusSquared = 5;

/// Pixels between two points of the heading line.
constexpr core::f32 kHeadingStepPixels = 0.5f;

/// Volume axes, in (z, y, x) order: along the slice's columns, along its rows, and across it.
struct SliceAxes final {
    core::u32 u;
    core::u32 v;
    core::u32 normal;
};

constexpr SliceAxes kAcrossZ{.u = 2u, .v = 1u, .normal = 0u};
constexpr SliceAxes kAcrossY{.u = 2u, .v = 0u, .normal = 1u};
constexpr SliceAxes kAcrossX{.u = 1u, .v = 0u, .normal = 2u};

[[nodiscard]] constexpr SliceAxes spanAxes(SliceAxis axis) noexcept
{
    switch (axis)
    {
    case SliceAxis::Z: return kAcrossZ;
    case SliceAxis::Y: return kAcrossY;
    case SliceAxis::X: return kAcrossX;
    }
    return kAcrossX;
}

[[nodiscard]] constexpr std::array<core::f32, 3> inVolumeOrder(const math::Vec3<core::f32> &vector) noexcept
{
    return {vector.z, vector.y, vector.x};
}

[[nodiscard]] constexpr core::i64 smallestStepCovering(core::i64 extent, core::u32 cells) noexcept
{
    return (extent + cells - 1) / static_cast<core::i64>(cells);
}

[[nodiscard]] constexpr bool panelFits(const MinimapStyle &style, core::u32 frameWidth, core::u32 frameHeight) noexcept
{
    return style.width != 0u && style.height != 0u && style.width <= frameWidth &&
           style.left <= frameWidth - style.width && style.height <= frameHeight &&
           style.top <= frameHeight - style.height;
}

/// The panel's rectangle in the caller's frame. Coordinates are panel pixels from its top-left corner.
struct Panel final {
    core::u32 *pixels;
    core::u32 frameWidth;
    core::u32 left;
    core::u32 top;
    core::u32 width;
    core::u32 height;

    [[nodiscard]] bool contains(core::f32 x, core::f32 y) const noexcept
    {
        return x >= 0.0f && y >= 0.0f && x < static_cast<core::f32>(width) && y < static_cast<core::f32>(height);
    }

    /// @pre @p x < @ref width and @p y < @ref height.
    void put(core::u32 x, core::u32 y, core::u32 colour) const noexcept
    {
        pixels[static_cast<core::usize>(top + y) * frameWidth + left + x] = colour;
    }

    void putIfInside(core::f32 x, core::f32 y, core::u32 colour) const noexcept
    {
        if (contains(x, y))
            put(static_cast<core::u32>(x), static_cast<core::u32>(y), colour);
    }
};

struct PanelPoint final {
    core::f32 x;
    core::f32 y;
};

/// The darkest non-zero sample of a slice and the distance to its brightest. A fixed window would show a flat
/// grey on one volume and pure white on the next -- the failure the main renderer's density window exists to
/// avoid, one panel smaller. A sample of 0 is the mask, not a density, so it takes no part.
struct ContrastWindow final {
    core::u8 lowest;
    core::f32 span;
};

[[nodiscard]] ContrastWindow contrastWindow(const VolumeSlice &slice) noexcept
{
    core::u8 lowest = 255u;
    core::u8 highest = 0u;
    const core::usize count = static_cast<core::usize>(slice.width) * slice.height;
    for (core::usize i = 0u; i < count; ++i)
    {
        const core::u8 sample = slice.samples[i];
        if (sample == 0u)
            continue;
        lowest = std::min(lowest, sample);
        highest = std::max(highest, sample);
    }
    return {.lowest = lowest, .span = highest > lowest ? static_cast<core::f32>(highest - lowest) : 1.0f};
}

[[nodiscard]] core::u32 greyOf(core::u8 sample, ContrastWindow window, core::f32 brightness) noexcept
{
    const core::f32 level = (static_cast<core::f32>(sample) - static_cast<core::f32>(window.lowest)) / window.span;
    const core::u32 grey = static_cast<core::u32>(level * brightness * 255.0f + 0.5f);
    return 0xFF000000u | (grey << 16) | (grey << 8) | grey;
}

void fillPanel(const Panel &panel, const VolumeSlice &slice, const MinimapStyle &style) noexcept
{
    const ContrastWindow window = contrastWindow(slice);
    const core::f32 brightness = style.sliceBrightness > 0.0f ? std::min(style.sliceBrightness, 1.0f) : 0.0f;
    for (core::u32 y = 0u; y < panel.height; ++y)
    {
        const core::u64 row = static_cast<core::u64>(slice.height) * y / panel.height;
        for (core::u32 x = 0u; x < panel.width; ++x)
        {
            const core::u64 column = static_cast<core::u64>(slice.width) * x / panel.width;
            const core::u8 sample = slice.samples[row * slice.width + column];
            panel.put(x, y, sample == 0u ? style.background : greyOf(sample, window, brightness));
        }
    }
}

void drawFrame(const Panel &panel, core::u32 colour) noexcept
{
    for (core::u32 x = 0u; x < panel.width; ++x)
    {
        panel.put(x, 0u, colour);
        panel.put(x, panel.height - 1u, colour);
    }
    for (core::u32 y = 0u; y < panel.height; ++y)
    {
        panel.put(0u, y, colour);
        panel.put(panel.width - 1u, y, colour);
    }
}

[[nodiscard]] PanelPoint eyeOnPanel(const Panel &panel, const VolumeSlice &slice, const VolumeGeometry &geometry,
                                    const Eye &eye, SliceAxes axes) noexcept
{
    const core::f32 metresPerSample = geometry.metresPerSample();
    const std::array<core::f32, 3> eyeInMetres = inVolumeOrder(eye.position);
    const core::f32 fractionAlongColumns =
        (eyeInMetres[axes.u] / metresPerSample - static_cast<core::f32>(slice.lowU)) /
        static_cast<core::f32>(slice.stepU * static_cast<core::i64>(slice.width));
    const core::f32 fractionAlongRows = (eyeInMetres[axes.v] / metresPerSample - static_cast<core::f32>(slice.lowV)) /
                                        static_cast<core::f32>(slice.stepV * static_cast<core::i64>(slice.height));
    return {.x = std::clamp(fractionAlongColumns * static_cast<core::f32>(panel.width), 0.0f,
                            static_cast<core::f32>(panel.width - 1u)),
            .y = std::clamp(fractionAlongRows * static_cast<core::f32>(panel.height), 0.0f,
                            static_cast<core::f32>(panel.height - 1u))};
}

/// Every pixel whose offset (dx, dy) from @p centre has dx * dx + dy * dy between the two bounds, inclusive.
void stampAnnulus(const Panel &panel, PanelPoint centre, core::i32 innerRadiusSquared, core::i32 outerRadiusSquared,
                  core::u32 colour) noexcept
{
    core::i32 reach = 0;
    while ((reach + 1) * (reach + 1) <= outerRadiusSquared)
        ++reach;
    for (core::i32 dy = -reach; dy <= reach; ++dy)
    {
        for (core::i32 dx = -reach; dx <= reach; ++dx)
        {
            const core::i32 distanceSquared = dx * dx + dy * dy;
            if (distanceSquared >= innerRadiusSquared && distanceSquared <= outerRadiusSquared)
                panel.putIfInside(centre.x + static_cast<core::f32>(dx), centre.y + static_cast<core::f32>(dy), colour);
        }
    }
}

void drawHeading(const Panel &panel, PanelPoint eyePoint, const Eye &eye, SliceAxes axes,
                 const MinimapStyle &style) noexcept
{
    const std::array<core::f32, 3> forward = inVolumeOrder(eye.forward);
    const core::f32 alongColumns = forward[axes.u];
    const core::f32 alongRows = forward[axes.v];
    const core::f32 inPlaneSquared = alongColumns * alongColumns + alongRows * alongRows;
    if (inPlaneSquared <= kRingInPlaneLengthSquared)
    {
        stampAnnulus(panel, eyePoint, kRingInnerRadiusSquared, kRingOuterRadiusSquared, style.marker);
        return;
    }

    const core::f32 inPlaneLength = pmr::sqrt(inPlaneSquared);
    const core::f32 directionX = alongColumns / inPlaneLength;
    const core::f32 directionY = alongRows / inPlaneLength;
    for (core::f32 distance = 0.0f; distance < style.headingLength; distance += kHeadingStepPixels)
    {
        const core::f32 x = eyePoint.x + directionX * distance;
        const core::f32 y = eyePoint.y + directionY * distance;
        if (!panel.contains(x, y))
            break;
        panel.put(static_cast<core::u32>(x), static_cast<core::u32>(y), style.marker);
    }
}

} // namespace

core::u32 extractSlice(const BrickMosaic &mosaic, const VolumeGeometry &geometry, VolumeSlice &slice) noexcept
{
    if (!slice.valid() || !geometry.valid())
        return 0u;

    const SliceAxes axes = spanAxes(slice.axis);
    const core::i64 extentU = geometry.samples[axes.u];
    const core::i64 extentV = geometry.samples[axes.v];
    slice.lowU = 0;
    slice.lowV = 0;
    slice.stepU = smallestStepCovering(extentU, slice.width);
    slice.stepV = smallestStepCovering(extentV, slice.height);
    slice.at = std::clamp<core::i64>(slice.at, 0, geometry.samples[axes.normal] - 1);

    core::u32 hits = 0u;
    for (core::u32 row = 0u; row < slice.height; ++row)
    {
        for (core::u32 column = 0u; column < slice.width; ++column)
        {
            core::i64 position[3]{};
            position[axes.normal] = slice.at;
            position[axes.u] = slice.lowU + static_cast<core::i64>(column) * slice.stepU;
            position[axes.v] = slice.lowV + static_cast<core::i64>(row) * slice.stepV;
            const bool insideTheVolume = position[axes.u] < extentU && position[axes.v] < extentV;
            const std::optional<core::u8> sample =
                insideTheVolume ? mosaic.sampleAt(position[0], position[1], position[2]) : std::nullopt;
            slice.samples[static_cast<core::usize>(row) * slice.width + column] = sample.value_or(0u);
            if (sample.has_value())
                ++hits;
        }
    }
    return hits;
}

bool drawMinimap(core::u32 *pixels, core::u32 frameWidth, core::u32 frameHeight, const MinimapStyle &style,
                 const VolumeSlice &slice, const VolumeGeometry &geometry, const Eye &eye) noexcept
{
    if (pixels == nullptr || !slice.valid() || slice.stepU < 1 || slice.stepV < 1 || !geometry.valid() ||
        geometry.metresPerSample() <= 0.0f || !panelFits(style, frameWidth, frameHeight))
        return false;

    const Panel panel{.pixels = pixels,
                      .frameWidth = frameWidth,
                      .left = style.left,
                      .top = style.top,
                      .width = style.width,
                      .height = style.height};
    fillPanel(panel, slice, style);
    drawFrame(panel, style.border);

    const SliceAxes axes = spanAxes(slice.axis);
    const PanelPoint eyePoint = eyeOnPanel(panel, slice, geometry, eye, axes);
    drawHeading(panel, eyePoint, eye, axes, style);
    stampAnnulus(panel, eyePoint, 0, kDotRadiusSquared, style.marker);
    return true;
}

} // namespace lpl::voxel
