/**
 * @file test_minimap.cpp
 * @brief Where the panel says you are, against where you actually are.
 *
 * @warning **A map is the one thing whose being wrong is invisible.** Every other picture this
 * renderer makes can be checked against the samples behind it; a marker on a slice is checked
 * against nothing, so a map that pointed at the wrong turn of a spiral would look exactly as
 * convincing as one that pointed at the right one. So where the panel puts the eye is asserted here
 * rather than trusted.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Minimap.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool ok, const char *what)
{
    ++gChecks;
    if (!ok)
    {
        ++gFailures;
        std::printf("  FAIL  %s\n", what);
    }
}

void checkEq(long long got, long long want, const char *what)
{
    ++gChecks;
    if (got != want)
    {
        ++gFailures;
        std::printf("  FAIL  %s: got %lld, want %lld\n", what, got, want);
    }
}

using lpl::core::f32;
using lpl::core::i32;
using lpl::core::i64;
using lpl::core::u32;
using lpl::core::u8;
namespace voxel = lpl::voxel;

constexpr u32 kMarker = 0xFF00FF00u;

/// A brick whose value rises with X, so a mirrored or transposed slice shows as a gradient running the wrong way.
std::vector<u8> xGradientBrick(i64 brickX)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, 0u);
    for (u32 lz = 0u; lz < voxel::kBrickEdge; ++lz)
        for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
            for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
            {
                const i64 x = brickX * voxel::kBrickEdge + static_cast<i64>(lx);
                bytes[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] =
                    static_cast<u8>(40 + (x % 200));
            }
    return bytes;
}

/// Level-0 bricks of @ref xGradientBrick, @p bricksZ by @p bricksY by @p bricksX of them from the origin.
struct Subject final {
    std::vector<std::vector<u8>> storage;
    voxel::BrickMosaic mosaic;

    Subject(i32 bricksZ, i32 bricksY, i32 bricksX)
    {
        storage.reserve(static_cast<std::size_t>(bricksZ * bricksY * bricksX));
        for (i32 z = 0; z < bricksZ; ++z)
            for (i32 y = 0; y < bricksY; ++y)
                for (i32 x = 0; x < bricksX; ++x)
                {
                    storage.push_back(xGradientBrick(x));
                    voxel::BrickView view{};
                    view.voxels = storage.back().data();
                    view.key = voxel::BrickKey{0u, z, y, x};
                    voxel::summarise(view);
                    mosaic.insert(view);
                }
    }
};

voxel::VolumeGeometry geometryOf(i64 samplesZ, i64 samplesY, i64 samplesX)
{
    voxel::VolumeGeometry geometry{};
    geometry.samples[0] = samplesZ;
    geometry.samples[1] = samplesY;
    geometry.samples[2] = samplesX;
    geometry.levels = 1u;
    geometry.voxelMicrometres = 7.91f;
    geometry.metresPerMicrometre = 11538.0f;
    return geometry;
}

voxel::VolumeSlice sliceInto(std::vector<u8> &samples, u32 width, u32 height, voxel::SliceAxis axis, i64 at)
{
    samples.assign(static_cast<std::size_t>(width) * height, 0u);
    voxel::VolumeSlice slice{};
    slice.samples = samples.data();
    slice.width = width;
    slice.height = height;
    slice.axis = axis;
    slice.at = at;
    return slice;
}

/// An eye at the level-0 sample (x, y, z), facing @p forward.
voxel::Eye eyeAtSample(const voxel::VolumeGeometry &geometry, f32 x, f32 y, f32 z, lpl::math::Vec3<f32> forward)
{
    const f32 metresPerSample = geometry.metresPerSample();
    voxel::Eye eye{};
    eye.position = {x * metresPerSample, y * metresPerSample, z * metresPerSample};
    eye.forward = forward;
    return eye;
}

voxel::MinimapStyle panelAtOrigin(u32 size)
{
    voxel::MinimapStyle style{};
    style.width = size;
    style.height = size;
    style.marker = kMarker;
    return style;
}

struct MarkerPixels final {
    double x{-1.0};
    double y{-1.0};
    std::size_t count{0};
};

/// Centroid and count of the marker pixels inside the panel of @p style: where the panel says the eye is.
MarkerPixels markerOn(const std::vector<u32> &pixels, u32 frameWidth, const voxel::MinimapStyle &style)
{
    double sumX = 0.0;
    double sumY = 0.0;
    MarkerPixels marker{};
    for (u32 y = style.top; y < style.top + style.height; ++y)
        for (u32 x = style.left; x < style.left + style.width; ++x)
            if (pixels[static_cast<std::size_t>(y) * frameWidth + x] == style.marker)
            {
                sumX += x;
                sumY += y;
                ++marker.count;
            }
    if (marker.count > 0)
    {
        marker.x = sumX / static_cast<double>(marker.count);
        marker.y = sumY / static_cast<double>(marker.count);
    }
    return marker;
}

bool isMarkerAt(const std::vector<u32> &pixels, u32 frameWidth, long long x, long long y)
{
    return pixels[static_cast<std::size_t>(y) * frameWidth + static_cast<std::size_t>(x)] == kMarker;
}

/// Draws into a square frame of @p frameSize pixels, failing a check when the panel is refused.
void drawInto(std::vector<u32> &pixels, u32 frameSize, const voxel::MinimapStyle &style,
              const voxel::VolumeSlice &slice, const voxel::VolumeGeometry &geometry, const voxel::Eye &eye)
{
    check(voxel::drawMinimap(pixels.data(), frameSize, frameSize, style, slice, geometry, eye),
          "a panel that fits is drawn");
}

} // namespace

int main()
{
    std::printf("minimap\n");

    const voxel::VolumeGeometry geometry = geometryOf(256, 128, 256);
    const Subject subject(2, 1, 2);
    const lpl::math::Vec3<f32> intoThePage{0.0f, 0.0f, 1.0f};

    // ── the slice spans the WHOLE subject ──────────────────────────────────
    {
        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, voxel::SliceAxis::Z, 100);
        const u32 hits = voxel::extractSlice(subject.mosaic, geometry, slice);
        checkEq(hits, 64 * 64, "every sample of the slice found the subject");
        checkEq(slice.stepU * static_cast<i64>(slice.width), geometry.samples[2], "the slice spans X entirely");
        checkEq(slice.stepV * static_cast<i64>(slice.height), geometry.samples[1], "and Y entirely");

        check(samples[10u * 64u + 60u] != samples[10u * 64u + 4u], "the slice varies along its own X");
        check(samples[0] != samples[63], "and the two ends of a row differ");

        slice.at = 999999;
        checkEq(voxel::extractSlice(subject.mosaic, geometry, slice), 64 * 64, "a position past the end is clamped");
        checkEq(slice.at, geometry.samples[0] - 1, "and the slice records where it was actually cut");
    }

    // ── an extent the slice width does not divide ─────────────────────────
    // Review of PR #208 (2026-10-03): the step was rounded down, so the slice stopped short of the
    // far edge while the marker was placed over the whole extent; at 300 samples across, the marker
    // sat fifty samples from the eye.
    {
        const voxel::VolumeGeometry wide = geometryOf(256, 128, 300);
        const Subject reachingPastTheEdge(2, 1, 3);
        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, voxel::SliceAxis::Z, 100);
        const u32 hits = voxel::extractSlice(reachingPastTheEdge.mosaic, wide, slice);

        checkEq(slice.stepU, 5, "the step is the smallest one whose columns reach across 300 samples");
        bool pastTheEdgeIsEmpty = true;
        for (u32 row = 0u; row < slice.height; ++row)
            for (u32 column = 60u; column < slice.width; ++column)
                pastTheEdgeIsEmpty = pastTheEdgeIsEmpty && samples[static_cast<std::size_t>(row) * 64u + column] == 0u;
        check(pastTheEdgeIsEmpty,
              "columns 60 to 63 lie past the extent and read nothing, a resident brick there or not");
        checkEq(hits, 60 * 64, "and are not counted as found");
    }
    for (const i64 extentX : {300, 383, 40})
    {
        const voxel::VolumeGeometry wide = geometryOf(256, 128, extentX);
        const Subject coveringIt(2, 1, 3);
        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, voxel::SliceAxis::Z, 100);
        (void) voxel::extractSlice(coveringIt.mosaic, wide, slice);

        voxel::MinimapStyle style = panelAtOrigin(100u);
        style.headingLength = 0.0f;
        std::vector<u32> pixels(200u * 200u, 0u);
        const i64 eyeX = extentX - 10;
        drawInto(pixels, 200u, style, slice, wide,
                 eyeAtSample(wide, static_cast<f32>(eyeX), 64.0f, 128.0f, {1.0f, 0.0f, 0.0f}));
        const MarkerPixels marker = markerOn(pixels, 200u, style);
        const long long columnUnderMarker = static_cast<long long>(slice.width) *
                                            static_cast<long long>(marker.x + 0.5) /
                                            static_cast<long long>(style.width);
        const long long columnOfEye = (eyeX - slice.lowU) / slice.stepU;
        std::printf("  extent %lld: the eye is in slice column %lld, the dot sits on column %lld\n",
                    static_cast<long long>(extentX), columnOfEye, columnUnderMarker);
        check(std::llabs(columnUnderMarker - columnOfEye) <= 1,
              "the dot sits on the slice column that holds the eye, within the one-column rounding of the panel");
    }

    // ── the marker lands where the eye is ──────────────────────────────────
    {
        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, voxel::SliceAxis::Z, 0);
        (void) voxel::extractSlice(subject.mosaic, geometry, slice);

        const voxel::MinimapStyle style = panelAtOrigin(100u);
        std::vector<u32> pixels(200u * 200u, 0u);
        // A quarter along X, three quarters along Y: distinguishable from any transposition.
        drawInto(pixels, 200u, style, slice, geometry, eyeAtSample(geometry, 64.0f, 96.0f, 128.0f, intoThePage));
        const MarkerPixels marker = markerOn(pixels, 200u, style);
        std::printf("  marker at (%.1f, %.1f) of 100x100; expected about (25, 75)\n", marker.x, marker.y);
        check(marker.x > 20.0 && marker.x < 30.0, "the marker sits a quarter across, where the eye is in X");
        check(marker.y > 70.0 && marker.y < 80.0, "and three quarters down, where it is in Y");

        const long long centreX = static_cast<long long>(marker.x + 0.5);
        const long long centreY = static_cast<long long>(marker.y + 0.5);
        check(isMarkerAt(pixels, 200u, centreX + 6, centreY) && isMarkerAt(pixels, 200u, centreX - 6, centreY) &&
                  isMarkerAt(pixels, 200u, centreX, centreY + 6) && isMarkerAt(pixels, 200u, centreX, centreY - 6),
              "a heading into the page draws a ring six pixels out on every side");
        check(!isMarkerAt(pixels, 200u, centreX + 4, centreY), "a ring, with a gap between it and the dot");

        std::vector<u32> inPlane(pixels.size(), 0u);
        drawInto(inPlane, 200u, style, slice, geometry,
                 eyeAtSample(geometry, 64.0f, 96.0f, 128.0f, {1.0f, 0.0f, 0.0f}));
        std::size_t reach = 0;
        for (u32 x = 0u; x < style.width; ++x)
            if (isMarkerAt(inPlane, 200u, x, centreY))
                reach = x > reach ? x : reach;
        std::printf("  in-plane heading reaches x = %zu from a marker at %lld\n", reach, centreX);
        check(static_cast<long long>(reach) > centreX + 8, "a heading in the plane draws a line pointing that way");
        check(!isMarkerAt(inPlane, 200u, centreX - 6, centreY), "and no ring");
    }

    // ── every axis puts the marker where the eye is ───────────────────────
    {
        struct AxisCase final {
            voxel::SliceAxis axis;
            lpl::math::Vec3<f32> alongNormal;
            double wantDown;
            const char *name;
        };
        // The eye at x = 1/4, y = 1/4, z = 3/4 of the volume, read through the table of SliceAxis.
        const AxisCase cases[]{
            {voxel::SliceAxis::Z, {0.0f, 0.0f, 1.0f}, 25.0, "Z: a quarter across (x), a quarter down (y)"     },
            {voxel::SliceAxis::Y, {0.0f, 1.0f, 0.0f}, 75.0, "Y: a quarter across (x), three quarters down (z)"},
            {voxel::SliceAxis::X, {1.0f, 0.0f, 0.0f}, 75.0, "X: a quarter across (y), three quarters down (z)"},
        };
        for (const AxisCase &axisCase : cases)
        {
            std::vector<u8> samples;
            voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, axisCase.axis, 60);
            checkEq(voxel::extractSlice(subject.mosaic, geometry, slice), 64 * 64,
                    "every sample of a slice along each axis found the subject");
            const voxel::MinimapStyle style = panelAtOrigin(100u);
            std::vector<u32> pixels(200u * 200u, 0u);
            drawInto(pixels, 200u, style, slice, geometry,
                     eyeAtSample(geometry, 64.0f, 32.0f, 192.0f, axisCase.alongNormal));
            const MarkerPixels marker = markerOn(pixels, 200u, style);
            check(std::fabs(marker.x - 25.0) < 1.0 && std::fabs(marker.y - axisCase.wantDown) < 1.0, axisCase.name);
        }
    }

    // ── the panel: the slice in its own contrast, framed ──────────────────
    {
        const voxel::VolumeGeometry wide = geometryOf(256, 128, 300);
        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, voxel::SliceAxis::Z, 0);
        (void) voxel::extractSlice(subject.mosaic, wide, slice);

        voxel::MinimapStyle style = panelAtOrigin(100u);
        style.left = 10u;
        style.top = 20u;
        std::vector<u32> pixels(200u * 200u, 0u);
        drawInto(pixels, 200u, style, slice, wide, eyeAtSample(wide, 10.0f, 10.0f, 0.0f, intoThePage));

        const auto pixelAt = [&](u32 x, u32 y) { return pixels[static_cast<std::size_t>(y) * 200u + x]; };
        check(pixelAt(10u, 20u) == style.border && pixelAt(109u, 20u) == style.border &&
                  pixelAt(10u, 119u) == style.border && pixelAt(109u, 119u) == style.border,
              "the frame is drawn on the panel's outermost pixels");
        check(pixelAt(9u, 20u) == 0u && pixelAt(110u, 119u) == 0u && pixelAt(10u, 19u) == 0u &&
                  pixelAt(10u, 120u) == 0u,
              "and nothing outside the panel is touched");

        u32 brightestGrey = 0u;
        bool sawBlack = false;
        bool sawBackground = false;
        for (u32 y = 21u; y < 119u; ++y)
            for (u32 x = 11u; x < 109u; ++x)
            {
                const u32 pixel = pixelAt(x, y);
                if (pixel == style.marker)
                    continue;
                sawBackground = sawBackground || pixel == style.background;
                sawBlack = sawBlack || pixel == 0xFF000000u;
                if (pixel != style.background && (pixel & 0xFFu) > brightestGrey)
                    brightestGrey = pixel & 0xFFu;
            }
        checkEq(brightestGrey, 191, "the slice's brightest sample is drawn at three quarters of white");
        check(sawBlack, "and its dimmest at black: the contrast is stretched over what the slice holds");
        check(sawBackground, "a cell with no sample is the background");
    }

    // ── a finer resident brick answers where it covers ────────────────────
    {
        std::vector<u8> fineBytes(voxel::kBrickVoxels, 7u);
        std::vector<u8> coarseBytes(voxel::kBrickVoxels, 99u);
        voxel::BrickMosaic mixed;
        voxel::BrickView coarse{};
        coarse.voxels = coarseBytes.data();
        coarse.key = voxel::BrickKey{1u, 0, 0, 0};
        voxel::summarise(coarse);
        mixed.insert(coarse);
        voxel::BrickView fine{};
        fine.voxels = fineBytes.data();
        fine.key = voxel::BrickKey{0u, 0, 0, 0};
        voxel::summarise(fine);
        mixed.insert(fine);

        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, voxel::SliceAxis::Z, 0);
        checkEq(voxel::extractSlice(mixed, geometry, slice), 64 * 64, "a coarse brick covers the whole slice");
        checkEq(samples[0], 7, "but where a finer brick is resident, it is the finer one that answers");
        checkEq(samples[63], 99, "and the coarse one everywhere else");
    }

    // ── an eye outside the subject is clamped, never dropped ───────────────
    {
        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 64u, 64u, voxel::SliceAxis::Z, 0);
        (void) voxel::extractSlice(subject.mosaic, geometry, slice);

        const voxel::MinimapStyle style = panelAtOrigin(100u);
        std::vector<u32> pixels(120u * 120u, 0u);
        drawInto(pixels, 120u, style, slice, geometry, eyeAtSample(geometry, 9000.0f, 9000.0f, 9000.0f, intoThePage));
        check(markerOn(pixels, 120u, style).count > 0,
              "an eye outside the subject is still shown, clamped to the edge");
    }

    // ── a panel that would not fit is refused rather than trampling memory ──
    {
        std::vector<u8> samples;
        voxel::VolumeSlice slice = sliceInto(samples, 16u, 16u, voxel::SliceAxis::Z, 0);
        (void) voxel::extractSlice(subject.mosaic, geometry, slice);
        const voxel::Eye eye = eyeAtSample(geometry, 0.0f, 0.0f, 0.0f, intoThePage);

        std::vector<u32> pixels(64u * 64u, 0u);
        const std::vector<u32> before = pixels;
        voxel::MinimapStyle style = panelAtOrigin(100u);
        style.left = 40u;
        style.top = 40u;
        check(!voxel::drawMinimap(pixels.data(), 64u, 64u, style, slice, geometry, eye),
              "a panel past the right edge of the frame is refused");
        check(pixels == before, "and draws nothing at all");

        style.left = 0xFFFFFFF0u;
        style.top = 0u;
        style.width = 32u;
        style.height = 32u;
        check(!voxel::drawMinimap(pixels.data(), 64u, 64u, style, slice, geometry, eye),
              "so is one whose right edge only fits because left + width wrapped around");
        check(pixels == before, "and it draws nothing either");

        style.left = 0u;
        style.top = 10u;
        style.height = 0u;
        check(!voxel::drawMinimap(pixels.data(), 64u, 64u, style, slice, geometry, eye), "so is a panel with no rows");
        check(pixels == before, "which draws no frame on the rows around it");
    }

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
