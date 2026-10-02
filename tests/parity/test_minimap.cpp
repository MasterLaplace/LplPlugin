/**
 * @file test_minimap.cpp
 * @brief Where the panel says you are, against where you actually are.
 *
 * @warning **A map is the one thing whose being wrong is invisible.** Every other picture this
 * renderer makes can be checked against the samples behind it; a marker on a slice is checked
 * against nothing, so a map that pointed at the wrong turn of a spiral would look exactly as
 * convincing as one that pointed at the right one. That is why the marker is derived from the same
 * geometry the march uses, and why that derivation is asserted here rather than trusted.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Minimap.hpp>

#include <cstdio>
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
using lpl::core::i64;
using lpl::core::u32;
using lpl::core::u8;
namespace voxel = lpl::voxel;

/// A brick whose value encodes its own position, so a slice that sampled the wrong place says so.
std::vector<u8> positionalBrick(i64 brickZ, i64 brickY, i64 brickX)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, 0u);
    for (u32 lz = 0u; lz < voxel::kBrickEdge; ++lz)
        for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
            for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
            {
                // A value that varies smoothly with X so a mirrored or transposed slice is visible
                // as a gradient running the wrong way.
                const i64 x = brickX * voxel::kBrickEdge + static_cast<i64>(lx);
                bytes[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] =
                    static_cast<u8>(40 + (x % 200));
            }
    (void) brickZ;
    (void) brickY;
    return bytes;
}

} // namespace

int main()
{
    std::printf("minimap\n");

    voxel::VolumeGeometry geometry{};
    geometry.samples[0] = 256; // z
    geometry.samples[1] = 128; // y
    geometry.samples[2] = 256; // x
    geometry.levels = 1u;
    geometry.voxelMicrometres = 7.91f;
    geometry.metresPerMicrometre = 11538.0f;

    std::vector<std::vector<u8>> storage;
    voxel::BrickMosaic mosaic;
    for (i64 z = 0; z < 2; ++z)
        for (i64 y = 0; y < 1; ++y)
            for (i64 x = 0; x < 2; ++x)
            {
                storage.push_back(positionalBrick(z, y, x));
                voxel::BrickView view{};
                view.voxels = storage.back().data();
                view.key = voxel::BrickKey{0u, static_cast<lpl::core::i32>(z), static_cast<lpl::core::i32>(y),
                                           static_cast<lpl::core::i32>(x)};
                voxel::summarise(view);
                mosaic.insert(view);
            }

    // ── the slice spans the WHOLE subject ──────────────────────────────────
    {
        std::vector<u8> samples(64u * 64u, 0u);
        voxel::VolumeSlice slice{};
        slice.samples = samples.data();
        slice.width = 64u;
        slice.height = 64u;
        slice.axis = voxel::SliceAxis::Z;
        slice.at = 100;
        const u32 hits = voxel::extractSlice(mosaic, geometry, slice);
        checkEq(hits, 64 * 64, "every sample of the slice found the subject");

        // A map that showed only part of the subject would be a second view of where you already
        // are, so the step has to cover the whole extent.
        checkEq(slice.stepU * static_cast<i64>(slice.width), geometry.samples[2], "the slice spans X entirely");
        checkEq(slice.stepV * static_cast<i64>(slice.height), geometry.samples[1], "and Y entirely");

        // The gradient runs the way X runs: a transposed or mirrored slice is a map that points
        // confidently at the wrong place, and nothing about it would look wrong.
        check(samples[10u * 64u + 60u] != samples[10u * 64u + 4u], "the slice varies along its own X");
        const u8 leftValue = samples[0];
        const u8 rightValue = samples[63];
        check(leftValue != rightValue, "and the two ends of a row differ");

        // A position past the end is clamped, not read out of bounds.
        slice.at = 999999;
        checkEq(voxel::extractSlice(mosaic, geometry, slice), 64 * 64, "a position past the end is clamped");
    }

    // ── the marker lands where the eye is ──────────────────────────────────
    {
        std::vector<u8> samples(64u * 64u, 128u);
        voxel::VolumeSlice slice{};
        slice.samples = samples.data();
        slice.width = 64u;
        slice.height = 64u;
        slice.axis = voxel::SliceAxis::Z;
        (void) voxel::extractSlice(mosaic, geometry, slice);

        constexpr u32 kW = 200u;
        constexpr u32 kH = 200u;
        std::vector<u32> pixels(static_cast<std::size_t>(kW) * kH, 0u);

        voxel::MinimapStyle style{};
        style.left = 0u;
        style.top = 0u;
        style.width = 100u;
        style.height = 100u;
        style.marker = 0xFF00FF00u;

        const f32 mps = geometry.metresPerSample();
        // A quarter along X, three quarters along Y: distinguishable from any transposition.
        voxel::Eye eye{};
        eye.position = {64.0f * mps, 96.0f * mps, 128.0f * mps};
        eye.forward = {0.0f, 0.0f, 1.0f};
        eye.right = {1.0f, 0.0f, 0.0f};
        eye.up = {0.0f, 1.0f, 0.0f};
        voxel::drawMinimap(pixels.data(), kW, kH, style, slice, geometry, eye);

        // Centroid of the marker pixels: where the panel says the eye is.
        double sx = 0.0;
        double sy = 0.0;
        std::size_t n = 0;
        for (u32 y = 0u; y < style.height; ++y)
            for (u32 x = 0u; x < style.width; ++x)
                if (pixels[static_cast<std::size_t>(y) * kW + x] == style.marker)
                {
                    sx += x;
                    sy += y;
                    ++n;
                }
        check(n > 0, "the marker is drawn");
        const double mx = n > 0 ? sx / static_cast<double>(n) : -1.0;
        const double my = n > 0 ? sy / static_cast<double>(n) : -1.0;
        // x = 64 of 256 is a quarter across; y = 96 of 128 is three quarters down.
        std::printf("  marker at (%.1f, %.1f) of 100x100; expected about (25, 75)\n", mx, my);
        check(mx > 20.0 && mx < 30.0, "the marker sits a quarter across, where the eye is in X");
        check(my > 70.0 && my < 80.0, "and three quarters down, where it is in Y");

        // ⚠ Looking along the slice normal must still SAY something. On a scroll that is the
        // common case -- the interesting direction is the axis, which is what the slice is cut
        // across -- so an arrow that projected to nothing would vanish exactly when somebody is
        // doing the normal thing.
        check(n > 20, "a heading into the page draws a ring, not a vanished arrow");

        // And a heading in the plane draws a line instead: fewer pixels than a ring, extending
        // away from the centre.
        std::vector<u32> inPlane(pixels.size(), 0u);
        voxel::Eye sideways = eye;
        sideways.forward = {1.0f, 0.0f, 0.0f};
        voxel::drawMinimap(inPlane.data(), kW, kH, style, slice, geometry, sideways);
        std::size_t reach = 0;
        for (u32 x = 0u; x < style.width; ++x)
            if (inPlane[static_cast<std::size_t>(static_cast<u32>(my)) * kW + x] == style.marker)
                reach = x > reach ? x : reach;
        std::printf("  in-plane heading reaches x = %zu from a marker at %.0f\n", reach, mx);
        check(static_cast<double>(reach) > mx + 8.0, "a heading in the plane draws a line pointing that way");
    }

    // ── an eye outside the subject is clamped, never dropped ───────────────
    {
        std::vector<u8> samples(64u * 64u, 128u);
        voxel::VolumeSlice slice{};
        slice.samples = samples.data();
        slice.width = 64u;
        slice.height = 64u;
        (void) voxel::extractSlice(mosaic, geometry, slice);

        constexpr u32 kW = 120u;
        constexpr u32 kH = 120u;
        std::vector<u32> pixels(static_cast<std::size_t>(kW) * kH, 0u);
        voxel::MinimapStyle style{};
        style.width = 100u;
        style.height = 100u;
        style.marker = 0xFF00FF00u;

        const f32 mps = geometry.metresPerSample();
        voxel::Eye outside{};
        // Well past the far corner: you can fly out of a scroll, and that is a real place to be.
        outside.position = {9000.0f * mps, 9000.0f * mps, 9000.0f * mps};
        outside.forward = {0.0f, 0.0f, 1.0f};
        outside.right = {1.0f, 0.0f, 0.0f};
        outside.up = {0.0f, 1.0f, 0.0f};
        voxel::drawMinimap(pixels.data(), kW, kH, style, slice, geometry, outside);

        std::size_t n = 0;
        for (u32 y = 0u; y < style.height; ++y)
            for (u32 x = 0u; x < style.width; ++x)
                if (pixels[static_cast<std::size_t>(y) * kW + x] == style.marker)
                    ++n;
        // A marker that vanished there would read as the map being broken rather than as the
        // operator being outside.
        check(n > 0, "an eye outside the subject is still shown, clamped to the edge");
    }

    // ── a panel that would not fit is refused rather than trampling memory ──
    {
        std::vector<u8> samples(16u * 16u, 128u);
        voxel::VolumeSlice slice{};
        slice.samples = samples.data();
        slice.width = 16u;
        slice.height = 16u;
        (void) voxel::extractSlice(mosaic, geometry, slice);

        std::vector<u32> pixels(64u * 64u, 0u);
        const std::vector<u32> before = pixels;
        voxel::MinimapStyle style{};
        style.left = 40u;
        style.top = 40u;
        style.width = 100u; // Past the right edge of a 64-wide frame.
        style.height = 100u;
        voxel::Eye eye{};
        eye.forward = {0.0f, 0.0f, 1.0f};
        voxel::drawMinimap(pixels.data(), 64u, 64u, style, slice, geometry, eye);
        check(pixels == before, "a panel that does not fit draws nothing at all");
    }

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
