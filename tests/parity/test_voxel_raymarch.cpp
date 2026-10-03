/**
 * @file test_voxel_raymarch.cpp
 * @brief What the volume renderer must do, checked against a field whose answer is known.
 *
 * @warning **The fixture is synthetic, and that is the point.** A real brick is a download, and a
 * check that needs the network goes red when the network does. The sheets here come from an
 * integer formula, so the expected answer is derivable by hand -- a slab of high density every
 * `pitch` samples inside a medium that is dark but never empty, which is the shape real
 * carbonised papyrus turned out to have when it was actually measured.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Raymarch.hpp>
#include <lpl/voxel/Residency.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
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

constexpr u8 kMedium = 130u;
constexpr u8 kSheet = 200u;

/// Sheets perpendicular to volume axis X, one slab of `thick` every `pitch` samples.
std::vector<u8> makeSheetBrick(u32 level, i64 brickX, u32 pitch, u32 thick)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
    const i64 span = voxel::brickSpanInBaseSamples(level);
    const i64 originX = brickX * span;
    const i64 step = static_cast<i64>(1) << level;
    for (u32 lz = 0u; lz < voxel::kBrickEdge; ++lz)
    {
        for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
        {
            for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
            {
                const i64 baseX = originX + static_cast<i64>(lx) * step;
                const i64 phase = ((baseX % static_cast<i64>(pitch)) + pitch) % static_cast<i64>(pitch);
                if (phase < static_cast<i64>(thick))
                    bytes[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] = kSheet;
            }
        }
    }
    return bytes;
}

std::vector<u8> makeFlatBrick(u8 value) { return std::vector<u8>(voxel::kBrickVoxels, value); }

/// A brick that is medium everywhere except a slab of `thick` samples at its low Z face.
std::vector<u8> makeEntrySlabBrick(u32 thick)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
    for (u32 lz = 0u; lz < thick && lz < voxel::kBrickEdge; ++lz)
    {
        for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
        {
            for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
                bytes[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] = kSheet;
        }
    }
    return bytes;
}

/// A dense ball in a medium: the only fixture here whose normals actually vary.
std::vector<u8> makeBallBrick(u32 radius)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
    const long long c = voxel::kBrickEdge / 2;
    const long long r2 = static_cast<long long>(radius) * radius;
    for (u32 lz = 0u; lz < voxel::kBrickEdge; ++lz)
    {
        for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
        {
            for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
            {
                const long long dz = static_cast<long long>(lz) - c;
                const long long dy = static_cast<long long>(ly) - c;
                const long long dx = static_cast<long long>(lx) - c;
                if (dz * dz + dy * dy + dx * dx <= r2)
                    bytes[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] = kSheet;
            }
        }
    }
    return bytes;
}

voxel::BrickView viewOf(const std::vector<u8> &bytes, voxel::BrickKey key)
{
    voxel::BrickView v{};
    v.voxels = bytes.data();
    v.key = key;
    voxel::summarise(v);
    return v;
}

/// Field of view of the one-pixel frames below. Their single ray goes straight ahead whatever it is.
constexpr f32 kPencil = 0.01f;

/// A camera looking down volume Z from (x, y, z), given in level-0 samples.
voxel::Eye eyeAlongZ(f32 metresPerSample, f32 x, f32 y, f32 z, f32 fieldOfView)
{
    voxel::Eye eye{};
    eye.position = {x * metresPerSample, y * metresPerSample, z * metresPerSample};
    eye.forward = {0.0f, 0.0f, 1.0f};
    eye.right = {1.0f, 0.0f, 0.0f};
    eye.up = {0.0f, 1.0f, 0.0f};
    eye.horizontalFieldOfView = fieldOfView;
    return eye;
}

/// Mean luminance of the non-background part of a frame, in [0,1].
f32 meanLuminance(const std::vector<u32> &pixels, u32 background)
{
    double acc = 0.0;
    std::size_t n = 0;
    for (u32 p : pixels)
    {
        if (p == background)
            continue;
        acc += static_cast<double>(((p >> 16) & 0xFFu) + ((p >> 8) & 0xFFu) + (p & 0xFFu)) / (3.0 * 255.0);
        ++n;
    }
    return n == 0 ? 0.0f : static_cast<f32>(acc / static_cast<double>(n));
}

/// Red channel of a 0xAARRGGBB pixel, in [0,1]. The ramp's red rises fastest with density.
f32 redLevel(u32 pixel) { return static_cast<f32>((pixel >> 16) & 0xFFu) / 255.0f; }

/// Largest change between two consecutive values of @p readingAt(0 .. count-1).
template <typename Reading> f32 worstAdjacentChange(int count, Reading readingAt)
{
    f32 worst = 0.0f;
    f32 previous = readingAt(0);
    for (int i = 1; i < count; ++i)
    {
        const f32 current = readingAt(i);
        const f32 change = current > previous ? current - previous : previous - current;
        if (change > worst)
            worst = change;
        previous = current;
    }
    return worst;
}

/// Whether @p key is the brick, at its own level, that holds the level-0 sample @p eye.
bool isInEyeColumn(const voxel::BrickKey &key, const i64 eye[3])
{
    return voxel::brickIndexOfBase(eye[0], key.level) == key.z && voxel::brickIndexOfBase(eye[1], key.level) == key.y &&
           voxel::brickIndexOfBase(eye[2], key.level) == key.x;
}

} // namespace

int main()
{
    std::printf("voxel raymarch\n");

    // ── bricks and indexing ────────────────────────────────────────────────
    {
        const auto flat = makeFlatBrick(77u);
        voxel::BrickView v = viewOf(flat, voxel::BrickKey{0u, 0, 0, 0});
        checkEq(v.lowest, 77, "a flat brick reports its value as both bounds (low)");
        checkEq(v.highest, 77, "a flat brick reports its value as both bounds (high)");

        const auto sheets = makeSheetBrick(0u, 0, 32u, 4u);
        v = viewOf(sheets, voxel::BrickKey{0u, 0, 0, 0});
        checkEq(v.lowest, kMedium, "a sheet brick's floor is the medium");
        checkEq(v.highest, kSheet, "a sheet brick's ceiling is the sheet");

        // Floor division, not truncation: the brick below the origin is -1, never 0.
        checkEq(voxel::brickIndexOfBase(0, 0u), 0, "sample 0 is in brick 0");
        checkEq(voxel::brickIndexOfBase(127, 0u), 0, "the last sample of brick 0 is in brick 0");
        checkEq(voxel::brickIndexOfBase(128, 0u), 1, "sample 128 opens brick 1");
        checkEq(voxel::brickIndexOfBase(-1, 0u), -1, "the sample below the origin is in brick -1");
        checkEq(voxel::brickIndexOfBase(-128, 0u), -1, "brick -1 starts at -128");
        checkEq(voxel::brickIndexOfBase(-129, 0u), -2, "and brick -2 below it");
        checkEq(voxel::brickIndexOfBase(255, 1u), 0, "a level-1 brick spans 256 base samples");
        checkEq(voxel::brickIndexOfBase(256, 1u), 1, "and the next one starts at 256");
    }

    // ── the mosaic: finest wins ────────────────────────────────────────────
    {
        const auto fine = makeFlatBrick(10u);
        const auto coarse = makeFlatBrick(20u);
        voxel::BrickMosaic m;
        // Coarse inserted FIRST on purpose: a mosaic that returned the first match would answer
        // with it, and the picture would depend on which download landed first.
        check(m.insert(viewOf(coarse, voxel::BrickKey{3u, 0, 0, 0})), "the coarse brick is listed");
        check(m.insert(viewOf(fine, voxel::BrickKey{0u, 0, 0, 0})), "the fine brick is listed");
        checkEq(m.count(), 2, "both are resident");
        checkEq(m.coarsestLevel(), 3, "the mosaic knows its coarsest level");
        checkEq(m.finestLevel(), 0, "and its finest, which bounds every hop a ray takes");

        const voxel::BrickView *hit = m.find(4, 4, 4);
        check(hit != nullptr, "a covered point finds a brick");
        checkEq(hit == nullptr ? -1 : static_cast<long long>(hit->key.level), 0,
                "the FINEST brick answers, not the first");

        // Outside the fine brick but inside the coarse one, the coarse one answers: that overlap
        // is what lets a fine brick be evicted without leaving a hole.
        hit = m.find(500, 4, 4);
        check(hit != nullptr && hit->key.level == 3u, "outside the fine brick, the coarse one covers");

        check(m.remove(voxel::BrickKey{0u, 0, 0, 0}), "a listed brick can be unlisted");
        check(!m.contains(voxel::BrickKey{0u, 0, 0, 0}), "and is then absent");
        checkEq(m.finestLevel(), 3, "evicting the finest brick moves the finest level up");
        hit = m.find(4, 4, 4);
        check(hit != nullptr && hit->key.level == 3u, "eviction of the fine brick leaves the coarse one, not a hole");
        checkEq(m.count(), 1, "the count follows the removal");
    }

    // ── the nearest sample, read from the brick find returns ──────────────
    {
        std::vector<u8> risingAlongX(voxel::kBrickVoxels, 0u);
        for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
            risingAlongX[lx] = static_cast<u8>(100u + lx);
        voxel::BrickMosaic m;
        check(m.insert(viewOf(risingAlongX, voxel::BrickKey{1u, 0, 0, 1})),
              "a level-1 brick at x 256 to 511 is listed");

        checkEq(m.sampleAt(0, 0, 256).value_or(0u), 100, "its first cell answers at its origin");
        checkEq(m.sampleAt(1, 1, 261).value_or(0u), 102, "a level-1 cell spans two level-0 samples on every axis");
        check(!m.sampleAt(0, 0, 255).has_value(), "nothing answers where nothing is resident");
        // 2^40 + 256 lands in brick index 2^32 + 1, which an i32 index holds as 1: this brick.
        check(!m.sampleAt(0, 0, (static_cast<i64>(1) << 40) + 256).has_value(),
              "nor where a coordinate is past what a brick index holds, rather than the brick it wraps onto");
    }

    // ── residency: nearest first ───────────────────────────────────────────
    {
        voxel::VolumeGeometry geometry{};
        geometry.samples[0] = 2048;
        geometry.samples[1] = 2048;
        geometry.samples[2] = 2048;
        geometry.levels = 4u;
        geometry.voxelMicrometres = 7.91f;
        geometry.metresPerMicrometre = 11538.0f;
        check(geometry.valid(), "the fixture geometry is valid");

        voxel::ResidencyParams params{};
        params.finestLevel = 0u;
        params.coarsestLevel = 3u;
        params.ringRadius = 2;
        params.budget = 512u;

        const i64 eye[3]{1024, 1024, 1024};
        std::vector<voxel::BrickKey> plan(512);
        u32 n = voxel::planResidency(geometry, params, eye, plan.data(), static_cast<u32>(plan.size()));
        check(n > 0u, "a plan is produced");

        // The first four entries are the eye's own column at each level: radius zero, every level.
        // That is what makes a truncated plan still cover everywhere.
        bool firstShellIsRadiusZero = true;
        for (u32 i = 0u; i < 4u && i < n; ++i)
            firstShellIsRadiusZero = firstShellIsRadiusZero && isInEyeColumn(plan[i], eye);
        check(firstShellIsRadiusZero, "the plan opens with the eye's own brick at every level");

        // A budget smaller than the plan must lose the far shell, never the near one.
        params.budget = 4u;
        const u32 tight = voxel::planResidency(geometry, params, eye, plan.data(), static_cast<u32>(plan.size()));
        checkEq(tight, 4, "a tight budget is respected exactly");
        bool tightIsAllNear = true;
        for (u32 i = 0u; i < tight; ++i)
            tightIsAllNear = tightIsAllNear && isInEyeColumn(plan[i], eye);
        check(tightIsAllNear, "a truncated plan keeps the ground under the eye and drops the horizon");

        // Nothing outside the subject is ever requested.
        const i64 corner[3]{0, 0, 0};
        params.budget = 512u;
        n = voxel::planResidency(geometry, params, corner, plan.data(), static_cast<u32>(plan.size()));
        bool allInside = true;
        for (u32 i = 0u; i < n; ++i)
        {
            const voxel::BrickKey &k = plan[i];
            if (k.z < 0 || k.y < 0 || k.x < 0 || k.z >= geometry.bricksAtLevel(0, k.level) ||
                k.y >= geometry.bricksAtLevel(1, k.level) || k.x >= geometry.bricksAtLevel(2, k.level))
                allInside = false;
        }
        check(allInside, "no request ever leaves the subject");

        // No key is asked for twice: a duplicate would push a real brick past the budget.
        bool unique = true;
        for (u32 i = 0u; i < n && unique; ++i)
        {
            for (u32 j = i + 1u; j < n; ++j)
            {
                if (plan[i] == plan[j])
                {
                    unique = false;
                    break;
                }
            }
        }
        check(unique, "the plan never asks for the same brick twice");
    }

    // ── the transfer curve ─────────────────────────────────────────────────
    voxel::DensityProfile profile{};
    profile.floorSample = 140u;
    profile.sheetSample = 200u;
    profile.mean = 147.0f;
    profile.deviation = 23.5f;
    // The ratios measured on one paired cubic millimetre of PHerc0172.
    const f32 measured[6]{1.0f, 0.940f, 0.821f, 0.651f, 0.485f, 0.383f};
    for (u32 lv = 0u; lv < 6u; ++lv)
        profile.spreadRatio[lv] = measured[lv];

    // A sheet is four samples thick in this fixture, and a sheet you can see through is not a
    // sheet: the peak is derived from that intent rather than picked.
    const f32 peakAlpha = voxel::alphaForOpaqueAfter(4.0f, 0.85f);
    check(peakAlpha > 0.3f && peakAlpha < 0.6f, "the derived peak opacity is in a sane range");
    const voxel::TransferFunction base = voxel::rampTransfer(profile.floorSample, profile.sheetSample, peakAlpha);
    {
        check(base.alpha[100] == 0.0f, "below the floor, nothing is painted");
        check(base.alpha[255] > base.alpha[170], "the ramp rises toward the sheet");
        check(base.alpha[170] > base.alpha[150], "and rises through the middle of the band");
        check(base.firstVisible > profile.floorSample && base.firstVisible < profile.sheetSample,
              "the first visible density sits inside the band");
        check(!base.reachesVisible(100u), "a brick that never reaches the band is skippable");
        check(base.reachesVisible(255u), "a brick that reaches it is not");

        const voxel::TransferFunction coarse = base.forLevel(profile, 4u);
        // Squeezed toward the mean: the samples moved there, so the window has to follow. Without
        // this the level-4 curve would paint the entire histogram.
        check(coarse.firstVisible > base.firstVisible,
              "a coarser level's window starts higher, following the collapsing spread");
        check(coarse.firstVisible < static_cast<u8>(profile.mean + 30.0f),
              "and stays near the mean rather than running away");
        // The level-0 window is [140, 200]. Squeezed toward 147 by 0.485 at level 4 it opens at
        // 143.6, and by 0.383 at level 5 it closes at 167.3. Recovered from the table rather than
        // stored, it had shrunk to [141, 199] first: it opened at 144.1 and closed at 166.9, one
        // density off at each end.
        check(coarse.alpha[143] == 0.0f && coarse.alpha[144] > 0.0f, "the rescaled window opens where it should");
        const voxel::TransferFunction coarser = base.forLevel(profile, 5u);
        check(coarser.alpha[167] < coarser.alpha[168], "and closes where it should");
    }

    // ── the march ──────────────────────────────────────────────────────────
    voxel::VolumeGeometry geometry{};
    geometry.samples[0] = 128;
    geometry.samples[1] = 128;
    geometry.samples[2] = 128;
    geometry.levels = 1u;
    geometry.voxelMicrometres = 7.91f;
    geometry.metresPerMicrometre = 11538.0f;
    const f32 metresPerSample = geometry.metresPerSample();

    const auto sheets = makeSheetBrick(0u, 0, 32u, 4u);
    voxel::BrickMosaic mosaic;
    mosaic.insert(viewOf(sheets, voxel::BrickKey{0u, 0, 0, 0}));

    constexpr u32 kW = 96u;
    constexpr u32 kH = 64u;
    std::vector<u32> frame(static_cast<std::size_t>(kW) * kH, 0u);

    voxel::MarchParams params{};
    params.stepSamples = 1.0f;
    params.maxDistanceMetres = 1.0e6f;

    voxel::Eye eye{};
    eye.position = {-40.0f * metresPerSample, 64.0f * metresPerSample, 64.0f * metresPerSample};
    eye.forward = {1.0f, 0.0f, 0.0f};
    eye.right = {0.0f, 0.0f, 1.0f};
    eye.up = {0.0f, 1.0f, 0.0f};

    {
        const voxel::MarchReport r =
            voxel::march(mosaic, geometry, profile, base, eye, params, frame.data(), kW, kH, 0u, kH);
        checkEq(static_cast<long long>(r.rays), static_cast<long long>(kW) * kH, "every pixel is traced");
        check(r.steps > 0, "the ray actually entered the volume");
        // Not saturation: whether a four-sample sheet goes fully opaque is a fact about the peak
        // opacity somebody chose, and boundary modulation deliberately lets a ray through the flat
        // middle of one. What the fixture can actually claim is that a stack of sheets is bright.
        check(meanLuminance(frame, params.background) > 0.4f, "a stack of sheets comes out bright");

        std::size_t painted = 0;
        for (u32 p : frame)
        {
            if (p != params.background)
                ++painted;
        }
        check(painted > frame.size() / 4, "most of the frame is the subject, not the background");
    }

    // Looking away must produce nothing. Without this, every check above is also satisfied by a
    // renderer that paints the same picture regardless of where the camera points.
    {
        voxel::Eye away = eye;
        away.forward = {-1.0f, 0.0f, 0.0f};
        away.right = {0.0f, 0.0f, -1.0f};
        std::vector<u32> blank(frame.size(), 0u);
        const voxel::MarchReport r =
            voxel::march(mosaic, geometry, profile, base, away, params, blank.data(), kW, kH, 0u, kH);
        checkEq(static_cast<long long>(r.escaped), static_cast<long long>(r.rays),
                "a camera pointed away hits nothing");
        checkEq(static_cast<long long>(r.steps), 0, "and takes no samples at all");
        bool allBackground = true;
        for (u32 p : blank)
        {
            if (p != params.background)
                allBackground = false;
        }
        check(allBackground, "the frame is entirely background");
    }

    // Empty-space skipping: a brick whose whole range is below the band must cost one comparison,
    // not two million reads. The step count is the only thing that can tell the difference.
    {
        const auto dark = makeFlatBrick(90u);
        voxel::BrickMosaic dim;
        dim.insert(viewOf(dark, voxel::BrickKey{0u, 0, 0, 0}));
        std::vector<u32> blank(frame.size(), 0u);
        const voxel::MarchReport r =
            voxel::march(dim, geometry, profile, base, eye, params, blank.data(), kW, kH, 0u, kH);
        check(r.skippedBricks > 0, "an all-medium brick is skipped whole");
        check(r.steps < r.rays * 4, "and skipping means a handful of samples per ray, not hundreds");
    }

    // ── the occupancy grid: leaving the medium in one comparison ───────────
    // Ninety-eight per cent of the samples a real frame takes paint nothing. The whole-brick
    // summary cannot skip them because the brick DOES contain sheets somewhere; a cell can.
    {
        // Medium everywhere except a slab at the far end: the near half of every ray is nothing.
        std::vector<u8> half(voxel::kBrickVoxels, kMedium);
        for (u32 lz = 96u; lz < voxel::kBrickEdge; ++lz)
            for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
                for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
                    half[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] = kSheet;

        voxel::BrickView plain{};
        plain.voxels = half.data();
        plain.key = voxel::BrickKey{0u, 0, 0, 0};
        voxel::summarise(plain);

        std::vector<u8> cells(voxel::kOccupancyBytes);
        voxel::BrickView celled{};
        celled.voxels = half.data();
        celled.key = voxel::BrickKey{0u, 0, 0, 0};
        voxel::summariseCells(celled, cells.data());
        check(celled.occupancy != nullptr, "the occupancy grid is filled");
        checkEq(celled.cellHighest(0u, 0u, 0u), kMedium, "a cell in the medium reports the medium");
        checkEq(celled.cellHighest(120u, 0u, 0u), kSheet, "and a cell in the slab reports the slab");

        voxel::VolumeGeometry one = geometry;
        one.samples[0] = 128;
        const voxel::Eye wide =
            eyeAlongZ(one.metresPerSample(), 64.0f, 64.0f, -4.0f, voxel::Eye{}.horizontalFieldOfView);

        std::vector<u32> a(frame.size(), 0u);
        voxel::BrickMosaic without;
        without.insert(plain);
        const voxel::MarchReport slow =
            voxel::march(without, one, profile, base, wide, params, a.data(), kW, kH, 0u, kH);

        std::vector<u32> b(frame.size(), 0u);
        voxel::BrickMosaic with;
        with.insert(celled);
        const voxel::MarchReport fast = voxel::march(with, one, profile, base, wide, params, b.data(), kW, kH, 0u, kH);

        std::printf("  occupancy: %llu samples without, %llu with, %llu cells skipped\n",
                    static_cast<unsigned long long>(slow.steps), static_cast<unsigned long long>(fast.steps),
                    static_cast<unsigned long long>(fast.skippedCells));
        check(fast.skippedCells > 0, "cells in the medium are skipped");
        check(fast.steps < slow.steps * 3u / 4u, "and skipping them costs materially fewer samples");
        // With nearest samples the skip changes nothing at all.
        voxel::MarchParams nearest = params;
        nearest.trilinear = false;
        std::vector<u32> nearestWithout(frame.size(), 0u);
        std::vector<u32> nearestWith(frame.size(), 0u);
        voxel::march(without, one, profile, base, wide, nearest, nearestWithout.data(), kW, kH, 0u, kH);
        voxel::march(with, one, profile, base, wide, nearest, nearestWith.data(), kW, kH, 0u, kH);
        checkEq(voxel::foldFrame(nearestWith.data(), static_cast<u32>(nearestWith.size())),
                voxel::foldFrame(nearestWithout.data(), static_cast<u32>(nearestWithout.size())),
                "with nearest samples, skipping cells leaves the frame identical");

        // Trilinear points lean across a cell's face (MarchParams::skipEmptyCells), so what is
        // checked there is the size of the difference.
        u32 worst = 0u;
        std::size_t differing = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            for (u32 shift = 0u; shift < 24u; shift += 8u)
            {
                const int x = static_cast<int>((a[i] >> shift) & 0xFFu);
                const int y = static_cast<int>((b[i] >> shift) & 0xFFu);
                const u32 d = static_cast<u32>(x > y ? x - y : y - x);
                if (d > worst)
                    worst = d;
                if (d != 0u)
                    ++differing;
            }
        }
        std::printf("  occupancy: worst channel difference %u of 255, %zu channels of %zu differ\n", worst, differing,
                    a.size() * 3u);
        // Ten of 255 is what was measured, and twelve leaves room for a change of rounding alone. It
        // buys a 3.4x cut in samples; whether that trade is acceptable is the caller's call, which
        // is why the skip can be switched off and why the count is printed beside every frame.
        check(worst <= 12u, "and the picture differs by at most a few per cent of a channel");
        check(fast.steps * 3u < slow.steps, "for a threefold cut in samples");
    }

    // Missing bricks are counted rather than silently painted.
    {
        voxel::BrickMosaic empty;
        std::vector<u32> blank(frame.size(), 0u);
        const voxel::MarchReport r =
            voxel::march(empty, geometry, profile, base, eye, params, blank.data(), kW, kH, 0u, kH);
        check(r.missingBricks > 0, "a mosaic with nothing in it reports the holes");
        check(r.saturated == 0, "and paints nothing");
    }

    // The step-length correction. Opacity is declared per level-0 sample of path, so a longer
    // step has to absorb what it stepped over; without that, the same matter looks thinner purely
    // because the camera moved back, which reads as the object changing at every level boundary.
    //
    // @warning The fixture matters more than the assertion here. A first version marched the sheet
    // field, where four-sample sheets at high opacity saturate either way -- so removing the
    // correction entirely left the drift at 0.2 %, and the check could not fail. It takes a FAINT,
    // UNIFORM medium, far from saturation, for the difference to be visible at all: there,
    // accumulated opacity is nearly proportional to the number of samples taken, and quartering
    // the samples quarters the picture.
    {
        const auto haze = makeFlatBrick(static_cast<u8>((profile.floorSample + profile.sheetSample) / 2u));
        voxel::BrickMosaic fog;
        fog.insert(viewOf(haze, voxel::BrickKey{0u, 0, 0, 0}));
        const voxel::TransferFunction faint =
            voxel::rampTransfer(profile.floorSample, profile.sheetSample, voxel::alphaForOpaqueAfter(2000.0f, 0.9f));

        std::vector<u32> fine(frame.size(), 0u);
        std::vector<u32> coarse(frame.size(), 0u);
        // Shading and boundary opacity are off HERE and nowhere else: they exist to make a
        // uniform medium nearly invisible, which is exactly the property this fixture needs to
        // keep in order to measure the step-length correction at all.
        voxel::MarchParams p1 = params;
        p1.stepSamples = 1.0f;
        p1.shading = 0.0f;
        p1.boundaryOpacity = 0.0f;
        voxel::MarchParams p4 = p1;
        p4.stepSamples = 4.0f;
        voxel::march(fog, geometry, profile, faint, eye, p1, fine.data(), kW, kH, 0u, kH);
        voxel::march(fog, geometry, profile, faint, eye, p4, coarse.data(), kW, kH, 0u, kH);
        const f32 a = meanLuminance(fine, params.background);
        const f32 b = meanLuminance(coarse, params.background);
        const f32 drift = (a > b ? a - b : b - a) / (a > 0.0f ? a : 1.0f);
        std::printf("  faint haze, step 1 -> %.4f, step 4 -> %.4f, drift %.1f %%\n", static_cast<double>(a),
                    static_cast<double>(b), static_cast<double>(drift * 100.0f));
        check(a > 0.05f, "the faint fixture actually accumulates something to compare");
        check(drift < 0.05f, "quadrupling the step barely moves a faint medium");
    }

    // And the same comparison on the sheet field, which is where a viewer actually lives.
    {
        std::vector<u32> fine(frame.size(), 0u);
        std::vector<u32> coarse(frame.size(), 0u);
        voxel::MarchParams p1 = params;
        p1.stepSamples = 1.0f;
        voxel::MarchParams p2 = params;
        p2.stepSamples = 2.0f;
        voxel::march(mosaic, geometry, profile, base, eye, p1, fine.data(), kW, kH, 0u, kH);
        voxel::march(mosaic, geometry, profile, base, eye, p2, coarse.data(), kW, kH, 0u, kH);
        const f32 a = meanLuminance(fine, params.background);
        const f32 b = meanLuminance(coarse, params.background);
        const f32 drift = (a > b ? a - b : b - a) / (a > 0.0f ? a : 1.0f);
        std::printf("  sheets, step 1 -> %.4f, step 2 -> %.4f, drift %.1f %%\n", static_cast<double>(a),
                    static_cast<double>(b), static_cast<double>(drift * 100.0f));
        check(drift < 0.08f, "doubling the step barely moves the picture");
    }

    // A rendered frame is stable across two identical calls. Not a parity gate -- it is float, and
    // this project reserves that word for folds that survive a change of target.
    {
        std::vector<u32> a(frame.size(), 0u);
        std::vector<u32> b(frame.size(), 0u);
        voxel::march(mosaic, geometry, profile, base, eye, params, a.data(), kW, kH, 0u, kH);
        voxel::march(mosaic, geometry, profile, base, eye, params, b.data(), kW, kH, 0u, kH);
        const u32 fa = voxel::foldFrame(a.data(), static_cast<u32>(a.size()));
        const u32 fb = voxel::foldFrame(b.data(), static_cast<u32>(b.size()));
        checkEq(fa, fb, "the same pose renders the same frame");
        std::printf("  frame signature 0x%08X\n", fa);
    }

    // Rendering by bands must equal rendering in one pass, or a thread pool changes the picture.
    {
        std::vector<u32> whole(frame.size(), 0u);
        std::vector<u32> banded(frame.size(), 0u);
        voxel::march(mosaic, geometry, profile, base, eye, params, whole.data(), kW, kH, 0u, kH);
        for (u32 row = 0u; row < kH; row += 7u)
            voxel::march(mosaic, geometry, profile, base, eye, params, banded.data(), kW, kH, row, 7u);
        checkEq(voxel::foldFrame(banded.data(), static_cast<u32>(banded.size())),
                voxel::foldFrame(whole.data(), static_cast<u32>(whole.size())),
                "splitting a frame into bands renders the same frame");
    }

    // ── shading: a lit surface, not a bright silhouette ────────────────────
    // What shading does is put a gradient ACROSS an object: the part facing the eye is bright and
    // the limb falls away. So the measurement is centre-versus-limb inside the silhouette, not
    // contrast over the frame.
    //
    // @warning Two earlier versions of this check measured the wrong thing and both reported the
    // wrong SIGN. Mean luminance points down, because shading darkens oblique surfaces. Variance
    // points down too, because an unshaded object is a flat disc against a dark ground -- maximum
    // contrast, and no structure whatsoever. A metric that moves in the wrong direction is worse
    // than no metric: it argues confidently for removing the feature.
    {
        const auto ball = makeBallBrick(40u);
        voxel::BrickMosaic round;
        round.insert(viewOf(ball, voxel::BrickKey{0u, 0, 0, 0}));

        voxel::MarchParams lit = params;
        voxel::MarchParams plain = params;
        plain.shading = 0.0f; // Only shading varies; boundary opacity stays on in both.

        auto centreOverLimb = [&](const voxel::MarchParams &p) {
            std::vector<u32> px(frame.size(), 0u);
            voxel::march(round, geometry, profile, base, eye, p, px.data(), kW, kH, 0u, kH);
            double centre = 0.0;
            double limb = 0.0;
            std::size_t nc = 0;
            std::size_t nl = 0;
            for (u32 y = 0u; y < kH; ++y)
            {
                for (u32 x = 0u; x < kW; ++x)
                {
                    const u32 v = px[static_cast<std::size_t>(y) * kW + x];
                    if (v == p.background)
                        continue;
                    const double l =
                        static_cast<double>(((v >> 16) & 0xFFu) + ((v >> 8) & 0xFFu) + (v & 0xFFu)) / (3.0 * 255.0);
                    const double dx = (static_cast<double>(x) - kW * 0.5) / (kW * 0.5);
                    const double dy = (static_cast<double>(y) - kH * 0.5) / (kH * 0.5);
                    const double r = dx * dx + dy * dy;
                    if (r < 0.09)
                    {
                        centre += l;
                        ++nc;
                    }
                    else if (r > 0.36)
                    {
                        limb += l;
                        ++nl;
                    }
                }
            }
            if (nc == 0 || nl == 0)
                return 0.0;
            return (centre / static_cast<double>(nc)) / ((limb / static_cast<double>(nl)) + 1e-9);
        };

        const double litRatio = centreOverLimb(lit);
        const double plainRatio = centreOverLimb(plain);
        std::printf("  ball, centre over limb: shaded %.3f, unshaded %.3f\n", litRatio, plainRatio);
        check(litRatio > 1.15, "a shaded ball is brighter where it faces the eye");
        check(litRatio > plainRatio * 1.10, "and flatter without shading");
    }

    // Boundary opacity buys DEPTH, and depth is measured in steps rather than in brightness: a ray
    // crossing homogeneous matter has to reach the first real surface instead of drowning in the
    // medium on the way there.
    {
        const auto uniform = makeFlatBrick(kSheet);
        voxel::BrickMosaic solid;
        solid.insert(viewOf(uniform, voxel::BrickKey{0u, 0, 0, 0}));

        std::vector<u32> a(frame.size(), 0u);
        const voxel::MarchReport withIt =
            voxel::march(solid, geometry, profile, base, eye, params, a.data(), kW, kH, 0u, kH);

        voxel::MarchParams none = params;
        none.boundaryOpacity = 0.0f;
        const voxel::MarchReport without =
            voxel::march(solid, geometry, profile, base, eye, none, a.data(), kW, kH, 0u, kH);

        std::printf("  uniform matter: %llu steps with boundary opacity, %llu without\n",
                    static_cast<unsigned long long>(withIt.steps), static_cast<unsigned long long>(without.steps));
        check(withIt.steps > without.steps * 2, "homogeneous matter lets a ray through instead of stopping it");
    }

    // ── skipping a brick must leave the ray AT ITS EXIT, not a span further on ──
    // samplesToFinestBoxExit in Raymarch.cpp says what a blind span jump looked like.
    {
        const auto empty = makeFlatBrick(kMedium); // Skippable: nothing in it can paint.
        const auto slab = makeEntrySlabBrick(8u);  // Visible, but only in its first 8 samples.
        voxel::BrickMosaic corridor;
        corridor.insert(viewOf(empty, voxel::BrickKey{0u, 0, 0, 0}));
        corridor.insert(viewOf(slab, voxel::BrickKey{0u, 1, 0, 0}));

        voxel::VolumeGeometry two = geometry;
        two.samples[0] = 256; // Two bricks deep along Z.

        // Enter the empty brick well past its face: a blind span jump from here overshoots the
        // slab entirely.
        const voxel::Eye pencil = eyeAlongZ(two.metresPerSample(), 64.0f, 64.0f, 50.0f, kPencil);
        u32 pixel = 0u;
        const voxel::MarchReport r = voxel::march(corridor, two, profile, base, pencil, params, &pixel, 1u, 1u, 0u, 1u);
        std::printf("  skip-then-hit: %llu bricks skipped, red %.3f\n",
                    static_cast<unsigned long long>(r.skippedBricks), static_cast<double>(redLevel(pixel)));
        check(r.skippedBricks >= 1u, "the empty brick really is skipped");
        check(redLevel(pixel) > 0.4f, "and the slab just past it is still hit");
    }

    // ── no skip ever leaps over a finer resident brick ─────────────────────
    // Levels overlap, so the brick answering at a point can be coarser than one a few samples
    // further on. A hop sized by the answering brick, or by the coarsest resident level, lands past
    // that finer brick and it vanishes from the picture: here, a bright brick that should stop the
    // ray outright.
    {
        voxel::VolumeGeometry deep = geometry;
        deep.samples[0] = 2048;
        deep.samples[1] = 2048;
        deep.samples[2] = 2048;
        deep.levels = 5u;
        const voxel::Eye pencil = eyeAlongZ(deep.metresPerSample(), 64.0f, 64.0f, -10.0f, kPencil);
        const auto saturatedRays = [&](const voxel::BrickMosaic &resident) {
            u32 pixel = 0u;
            return static_cast<long long>(
                voxel::march(resident, deep, profile, base, pencil, params, &pixel, 1u, 1u, 0u, 1u).saturated);
        };

        const auto bright = makeFlatBrick(kSheet);
        const auto dark = makeFlatBrick(100u);

        voxel::BrickMosaic insideDarkBrick;
        insideDarkBrick.insert(viewOf(dark, voxel::BrickKey{2u, 0, 0, 0}));
        insideDarkBrick.insert(viewOf(bright, voxel::BrickKey{0u, 2, 0, 0}));
        checkEq(saturatedRays(insideDarkBrick), 1, "skipping a dark coarse brick stops at the fine brick inside it");

        voxel::BrickMosaic pastAHole;
        pastAHole.insert(viewOf(bright, voxel::BrickKey{0u, 1, 0, 0}));
        pastAHole.insert(viewOf(bright, voxel::BrickKey{3u, 1, 1, 1}));
        checkEq(saturatedRays(pastAHole), 1, "crossing a hole stops at the fine brick beyond it");

        std::vector<u8> darkButVisibleElsewhere = dark;
        for (u32 lz = 0u; lz < voxel::kBrickEdge; ++lz)
            for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
                for (u32 lx = 120u; lx < voxel::kBrickEdge; ++lx)
                    darkButVisibleElsewhere[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) *
                                                voxel::kBrickEdge +
                                            lx] = kSheet;
        std::vector<u8> cells(voxel::kOccupancyBytes);
        voxel::BrickView celled = viewOf(darkButVisibleElsewhere, voxel::BrickKey{4u, 0, 0, 0});
        voxel::summariseCells(celled, cells.data());
        voxel::BrickMosaic insideDarkCell;
        insideDarkCell.insert(celled);
        insideDarkCell.insert(viewOf(bright, voxel::BrickKey{0u, 1, 0, 0}));
        checkEq(saturatedRays(insideDarkCell), 1, "skipping a dark coarse cell stops at the fine brick inside it");

        // The same hops, from a ray that sits exactly on the low face of its box and leaves through
        // it: the exit is right there, so the hop is one step, never a whole box.
        voxel::VolumeGeometry exact = geometry;
        exact.samples[0] = 512;
        exact.samples[1] = 512;
        exact.samples[2] = 512;
        exact.levels = 3u;
        exact.voxelMicrometres = 1.0f;
        exact.metresPerMicrometre = 1.0e6f;
        check(exact.metresPerSample() == 1.0f, "one sample is exactly one metre, so a brick face is an exact position");
        voxel::Eye down = eyeAlongZ(1.0f, 64.0f, 64.0f, 256.0f, kPencil);
        down.forward = {0.0f, 0.0f, -1.0f};
        down.right = {-1.0f, 0.0f, 0.0f};
        const auto saturatedGoingDown = [&](const voxel::BrickMosaic &resident) {
            u32 pixel = 0u;
            return static_cast<long long>(
                voxel::march(resident, exact, profile, base, down, params, &pixel, 1u, 1u, 0u, 1u).saturated);
        };

        voxel::BrickMosaic belowAHole;
        belowAHole.insert(viewOf(bright, voxel::BrickKey{0u, 1, 0, 0}));
        checkEq(saturatedGoingDown(belowAHole), 1, "leaving a hole through the face it sits on stops right below it");

        voxel::BrickMosaic belowADarkBrick;
        belowADarkBrick.insert(viewOf(dark, voxel::BrickKey{2u, 0, 0, 0}));
        belowADarkBrick.insert(viewOf(bright, voxel::BrickKey{0u, 1, 0, 0}));
        checkEq(saturatedGoingDown(belowADarkBrick), 1, "and so does leaving a dark coarse brick through that face");
    }

    // ── the brick face must not be visible ─────────────────────────────────
    // crossBrickSample in Raymarch.cpp says what the one-sample seam of a clamped cell looked like.
    {
        // Two bricks side by side along Z with a smooth ramp running through both, so a correct
        // sample crossing the face sees a straight line and a clamped one sees a step.
        auto ramp = [](i64 brickZ) {
            std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
            for (u32 lz = 0u; lz < voxel::kBrickEdge; ++lz)
            {
                const i64 z = brickZ * static_cast<i64>(voxel::kBrickEdge) + static_cast<i64>(lz);
                // Monotone across BOTH bricks. A first version used a sawtooth, which put a real
                // discontinuity at exactly the face being tested -- the fixture would have failed
                // a perfect sampler.
                const u8 v = static_cast<u8>(140 + z / 4);
                for (u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
                    for (u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
                        bytes[(static_cast<std::size_t>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] = v;
            }
            return bytes;
        };
        const auto low = ramp(0);
        const auto high = ramp(1);
        voxel::BrickMosaic pair;
        pair.insert(viewOf(low, voxel::BrickKey{0u, 0, 0, 0}));
        pair.insert(viewOf(high, voxel::BrickKey{0u, 1, 0, 0}));

        voxel::VolumeGeometry two = geometry;
        two.samples[0] = 256;

        voxel::MarchParams once = params;
        once.shading = 0.0f;
        once.boundaryOpacity = 0.0f;
        once.stepSamples = 0.5f;
        once.maxSteps = 1u; // The FIRST sample only: this is a probe of the field, not a walk.

        // Sample the field either side of the face by putting the eye at successive Z offsets, from
        // 120 to 136 across the face at 128, and reading the very first thing each ray paints.
        const f32 worst = worstAdjacentChange(40, [&](int i) {
            const f32 z = 120.0f + static_cast<f32>(i) * 0.4f;
            u32 pixel = 0u;
            voxel::march(pair, two, profile, base, eyeAlongZ(two.metresPerSample(), 64.0f, 64.0f, z, kPencil), once,
                         &pixel, 1u, 1u, 0u, 1u);
            return redLevel(pixel);
        });
        std::printf("  brick face: worst single-step change across it %.4f\n", static_cast<double>(worst));
        check(worst < 0.05f, "crossing a brick face leaves no step in a field that has none");
    }

    // ── the level-of-detail seam ───────────────────────────────────────────
    // MarchParams::levelBlendSamples says what a hard switch between levels looked like.
    {
        // A fine brick over the first 128 samples and a coarse one covering four times that, at a
        // different density: crossing out of the fine one is exactly the seam.
        const auto near = makeFlatBrick(200u);
        const auto far = makeFlatBrick(150u);
        voxel::BrickMosaic layered;
        layered.insert(viewOf(far, voxel::BrickKey{2u, 0, 0, 0}));
        layered.insert(viewOf(near, voxel::BrickKey{0u, 0, 0, 0}));

        // Pencil rays fired at successive points from x = 100 to 160, across the fine brick's
        // outer face at 128, and the biggest jump between neighbours.
        auto biggestStep = [&](f32 band) {
            voxel::MarchParams p = params;
            p.levelBlendSamples = band;
            p.shading = 0.0f;
            p.boundaryOpacity = 0.0f;
            return worstAdjacentChange(60, [&](int i) {
                const f32 x = 100.0f + static_cast<f32>(i);
                u32 pixel = 0u;
                voxel::march(layered, geometry, profile, base, eyeAlongZ(metresPerSample, x, 64.0f, -20.0f, kPencil), p,
                             &pixel, 1u, 1u, 0u, 1u);
                return redLevel(pixel);
            });
        };

        const f32 hard = biggestStep(0.0f);
        const f32 soft = biggestStep(24.0f);
        std::printf("  level seam: hard switch jumps %.4f, blended jumps %.4f\n", static_cast<double>(hard),
                    static_cast<double>(soft));
        check(hard > 0.02f, "the fixture really does have a seam to hide");
        // Measured, not hoped for: 0.62 down to 0.35, a 44 % cut. It is not zero and cannot be,
        // because a ray INTEGRATES -- a density that ramps smoothly still produces a pixel that
        // does not, since opacity accumulates nonlinearly along the path. The band softens the
        // step; only a much wider band would flatten it, and a wide band throws away the detail it
        // was supposed to be showing. This fixture is also deliberately extreme: 200 against 150,
        // with the coarse level's window at 141 to 191, so the two levels are opaque against
        // nearly transparent.
        check(soft < hard * 0.7f, "blending across the band cuts the worst jump by a third or more");
    }

    // ── the corridor: looking ALONG the sheets versus ACROSS them ──────────
    // This is the whole point of walking a scroll, and it is also the only check here that can
    // catch a transposed axis mapping. The sheets are perpendicular to volume X, so a ray running
    // along volume Z stays in the gap and comes out the far side; the same ray turned to cross
    // them piles up four sheets and goes opaque. Swap the world-to-volume mapping and the two
    // assertions trade places -- while every other check in this file still passes, because a
    // transposed subject is still a plausible subject.
    {
        const voxel::Eye down = eyeAlongZ(metresPerSample, 16.5f, 64.0f, 2.0f, kPencil); // x = 16.5: inside a gap.
        voxel::Eye across = down;
        across.forward = {1.0f, 0.0f, 0.0f};
        across.right = {0.0f, 0.0f, -1.0f};

        u32 onePixel = 0u;
        const voxel::MarchReport alongRay =
            voxel::march(mosaic, geometry, profile, base, down, params, &onePixel, 1u, 1u, 0u, 1u);
        const f32 alongRed = redLevel(onePixel);

        const voxel::MarchReport acrossRay =
            voxel::march(mosaic, geometry, profile, base, across, params, &onePixel, 1u, 1u, 0u, 1u);
        const f32 acrossRed = redLevel(onePixel);

        std::printf("  along the corridor: %llu steps, red %.3f | across the sheets: %llu steps, red %.3f\n",
                    static_cast<unsigned long long>(alongRay.steps), static_cast<double>(alongRed),
                    static_cast<unsigned long long>(acrossRay.steps), static_cast<double>(acrossRed));

        checkEq(static_cast<long long>(alongRay.escaped), 1, "a ray down the corridor comes out the far side");
        check(alongRed < 0.15f, "and comes out having picked up almost nothing: the gap is a gap");
        check(acrossRed > 0.6f, "a ray across the sheets is stopped by them");
        // The ratio is the claim, not either number: from inside a corridor you see ALONG it and
        // not THROUGH its walls. A threshold on saturation would instead be a claim about the
        // peak opacity somebody picked, which is a setting rather than a property of the field.
        check(acrossRed > alongRed * 10.0f, "the wall is an order of magnitude brighter than the corridor");
    }

    // ── the profile, measured rather than assumed ──────────────────────────
    {
        const auto sheetBrick = makeSheetBrick(0u, 0, 32u, 4u);
        voxel::BrickView v = viewOf(sheetBrick, voxel::BrickKey{0u, 0, 0, 0});
        // Sheets are 4 samples of every 32, so an eighth of the matter is sheet.
        const voxel::DensityProfile measuredProfile = voxel::measureProfile(&v, 1u, 0.125f);
        check(measuredProfile.valid(), "a profile comes out usable");
        check(measuredProfile.sheetSample > kMedium, "the sheet end of the window is above the medium");
        check(measuredProfile.sheetSample <= kSheet, "and does not exceed the brightest matter present");
        check(measuredProfile.mean > kMedium && measuredProfile.mean < kSheet, "the mean sits between the two");
    }

    // ── the free camera, and the bug it exists to not repeat ──────────────
    {
        // FreeCamera's header names the bug: a sine that does not reduce its argument freezes the
        // camera, and the first sixty degrees of every turn still work perfectly. The reference is
        // the library sine: the same reduction on both sides would only prove the code agrees with
        // itself -- a mistake this repository has already recorded twice.
        const auto worstSineError = [](int first, int last, f32 radiansPerStep) {
            f32 worst = 0.0f;
            for (int i = first; i <= last; ++i)
            {
                const f32 a = static_cast<f32>(i) * radiansPerStep;
                const f32 got = voxel::wrappedSine(a);
                const f32 want = static_cast<f32>(std::sin(static_cast<double>(a)));
                const f32 e = got > want ? got - want : want - got;
                if (e > worst)
                    worst = e;
            }
            return worst;
        };
        const f32 nearTurns = worstSineError(-1440, 1440, 0.0174532925f); // Four full turns, both ways.
        const f32 farTurns = worstSineError(-7300, 7300, 1.37f);          // Out to ten thousand radians.
        std::printf("  wrapped sine: worst error %.2e over four turns, %.2e out to 1e4 radians\n",
                    static_cast<double>(nearTurns), static_cast<double>(farTurns));
        check(nearTurns < 2e-4f, "the sine holds over the whole circle, not just near zero");
        check(farTurns < 2e-4f, "and holds out to ten thousand radians, well past 64 turns");
        check(voxel::wrapAngle(std::numeric_limits<f32>::infinity()) == 0.0f &&
                  voxel::wrapAngle(std::numeric_limits<f32>::quiet_NaN()) == 0.0f,
              "an angle that is not a number wraps to zero instead of poisoning a direction");
        bool hugeStayOnTheCircle = true;
        for (f32 huge = 1.0e8f; huge < 3.0e38f; huge *= 3.0f)
        {
            for (const f32 a : {huge, -huge, 1.0e17f})
            {
                const f32 wrapped = voxel::wrapAngle(a);
                const f32 sine = voxel::wrappedSine(a);
                if (!(wrapped >= -3.1415927f && wrapped <= 3.1415927f) || !(sine >= -1.0f && sine <= 1.0f))
                    hugeStayOnTheCircle = false;
            }
        }
        check(hugeStayOnTheCircle, "an angle too large to hold a fraction of a turn still lands on the circle");

        const auto facesYaw = [](const voxel::Eye &e, double yaw) {
            return std::fabs(static_cast<double>(e.forward.x) - std::sin(yaw)) < 1e-3 &&
                   std::fabs(static_cast<double>(e.forward.z) - std::cos(yaw)) < 1e-3;
        };
        voxel::FreeCamera camera;
        camera.turn(3.14159265f, 0.0f); // A half turn.
        check(camera.eye().forward.z < -0.99f, "a half turn actually faces the other way");
        camera.turn(100.0f, 0.0f); // Sixteen turns more, the case that froze.
        check(facesYaw(camera.eye(), 3.14159265 + 100.0), "sixteen more turns face exactly where they should");
        camera.turn(4096.0f, 0.0f); // Six hundred and fifty more.
        check(facesYaw(camera.eye(), 3.14159265 + 100.0 + 4096.0), "and so do six hundred and fifty more");

        // Pitch is clamped rather than wrapped: rolling over the vertical flips the horizon and no
        // reading of the controls recovers from it.
        camera.turn(0.0f, 100.0f);
        check(camera.pitch() < 1.5708f, "pitch stops just short of straight up");
        camera.turn(0.0f, -200.0f);
        check(camera.pitch() > -1.5708f, "and just short of straight down");

        // Moving with no input must not move: a normalize of a zero vector is how a camera drifts.
        voxel::FreeCamera still;
        const lpl::math::Vec3<f32> before = still.position;
        still.move(0.0f, 0.0f, 0.0f, 10.0f);
        check(still.position.x == before.x && still.position.y == before.y && still.position.z == before.z,
              "no input moves nothing");
        still.move(1.0f, 0.0f, 0.0f, 5.0f);
        check(still.position.z > 4.9f && still.position.z < 5.1f, "forward moves forward by the distance asked");
    }

    // ── the overlay, and the three rules that keep it honest ───────────────
    // MarchParams::overlay says why a prediction must never look like the scan.
    {
        const auto ink = makeFlatBrick(255u); // "Ink everywhere", the most demanding overlay there is.
        voxel::BrickMosaic predicted;
        predicted.insert(viewOf(ink, voxel::BrickKey{0u, 0, 0, 0}));

        std::vector<u32> plainFrame(frame.size(), 0u);
        std::vector<u32> tinted(frame.size(), 0u);

        // Rule one: off by default. A caller that does not ask gets the scan.
        voxel::MarchParams bare = params;
        const voxel::MarchReport noOverlay =
            voxel::march(mosaic, geometry, profile, base, eye, bare, plainFrame.data(), kW, kH, 0u, kH);
        checkEq(static_cast<long long>(noOverlay.overlaid), 0, "no overlay is asked for, so none is painted");

        voxel::MarchParams shown = params;
        shown.overlay = &predicted;
        shown.overlayConfidence = 1.0f;
        const voxel::MarchReport withOverlay =
            voxel::march(mosaic, geometry, profile, base, eye, shown, tinted.data(), kW, kH, 0u, kH);
        check(withOverlay.overlaid > 0, "asked for, it paints");

        auto redness = [](const std::vector<u32> &px, u32 background) {
            double r = 0.0;
            double other = 0.0;
            for (u32 p : px)
            {
                if (p == background)
                    continue;
                r += static_cast<double>((p >> 16) & 0xFFu);
                other += static_cast<double>(((p >> 8) & 0xFFu) + (p & 0xFFu)) * 0.5;
            }
            return other > 0.0 ? r / other : 0.0;
        };
        const double plainRed = redness(plainFrame, params.background);
        const double fullRed = redness(tinted, params.background);
        check(fullRed > plainRed * 1.2, "and at full confidence it is clearly visible");

        // Rule two: confidence multiplies the tint. A map worth half must not look as solid as one
        // worth nearly all -- putting the number in a caption instead is how uncertainty gets lost.
        std::vector<u32> half(frame.size(), 0u);
        voxel::MarchParams unsure = shown;
        unsure.overlayConfidence = 0.3f;
        voxel::march(mosaic, geometry, profile, base, eye, unsure, half.data(), kW, kH, 0u, kH);
        const double halfRed = redness(half, params.background);
        std::printf("  overlay redness: none %.3f, confidence 0.3 -> %.3f, confidence 1.0 -> %.3f\n", plainRed, halfRed,
                    fullRed);
        check(halfRed > plainRed && halfRed < fullRed, "a less confident map is painted less strongly");

        // Rule three: the overlay only tints matter the scan already put there. A prediction
        // floating in a void would be the model drawing rather than reading.
        voxel::BrickMosaic emptyScan;
        std::vector<u32> nothing(frame.size(), 0u);
        const voxel::MarchReport onNothing =
            voxel::march(emptyScan, geometry, profile, base, eye, shown, nothing.data(), kW, kH, 0u, kH);
        checkEq(static_cast<long long>(onNothing.overlaid), 0, "with no scan under it, an overlay paints nothing");
    }

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
