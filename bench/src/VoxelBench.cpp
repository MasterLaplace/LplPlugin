/**
 * @file VoxelBench.cpp
 * @brief Where the volume renderer's time actually goes.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/bench/Harness.hpp>
#include <lpl/bench/VoxelBench.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/NonCopyable.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/voxel/Brick.hpp>
#include <lpl/voxel/Mosaic.hpp>
#include <lpl/voxel/Raymarch.hpp>
#include <lpl/voxel/Sheet.hpp>
#include <lpl/voxel/Transfer.hpp>
#include <lpl/voxel/Volume.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

namespace lpl::bench {

namespace {

constexpr core::u8 kMedium = 130u;
constexpr core::u8 kSheet = 205u;

/// The measured spire pitch of a Herculaneum roll, in level-0 samples: a ridge at every multiple of it along X.
constexpr core::i64 kPitch = 38;
constexpr core::i64 kRidgeHalfWidth = 3;

constexpr core::u32 kAxisZ = 0u;
constexpr core::u32 kAxisY = 1u;
constexpr core::u32 kAxisX = 2u;

/// More than one, so that the level blend has coarser bricks to fade into: with one, it changes no count.
constexpr core::u32 kSceneLevels = 3u;
constexpr core::u32 kFrameWidth = 320u;
constexpr core::u32 kFrameHeight = 200u;

/**
 * A resident set of synthetic bricks, plus everything a march needs.
 *
 * Move-only: the mosaic points into the brick buffers, which a move hands over and a copy would not.
 */
struct VoxelScene final : private core::NonCopyable<VoxelScene> {
    std::vector<std::vector<core::u8>> brickVoxels;
    std::vector<std::vector<core::u8>> brickOccupancy;
    voxel::BrickMosaic mosaic;
    voxel::VolumeGeometry geometry{};
    voxel::DensityProfile profile{};
    voxel::TransferFunction transfer{};
    voxel::Eye eye{};
};

[[nodiscard]] constexpr core::i64 distanceToRidge(core::i64 x) noexcept
{
    const core::i64 phase = ((x % kPitch) + kPitch) % kPitch;
    return phase < kPitch / 2 ? phase : kPitch - phase;
}

[[nodiscard]] core::u8 sheetSampleAt(core::i64 x) noexcept
{
    // A ridge, not a slab: the marcher's gradient has nothing to read off a flat-topped sheet,
    // and such a fixture would report a speed the real field never gives.
    const core::i64 distance = distanceToRidge(x);
    if (distance >= kRidgeHalfWidth)
        return kMedium;
    const core::f32 ridgeWeight = 1.0f - static_cast<core::f32>(distance) / static_cast<core::f32>(kRidgeHalfWidth);
    return static_cast<core::u8>(static_cast<core::f32>(kMedium) +
                                 static_cast<core::f32>(kSheet - kMedium) * ridgeWeight * ridgeWeight);
}

void fillWithSheetsAcrossX(std::vector<core::u8> &voxels, const voxel::BrickKey &key)
{
    const core::i64 origin = voxel::brickOriginInBaseSamples(key, kAxisX);
    const core::i64 step = static_cast<core::i64>(1) << key.level;
    std::array<core::u8, voxel::kBrickEdge> row{};
    for (core::u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
        row[lx] = sheetSampleAt(origin + static_cast<core::i64>(lx) * step);

    voxels.resize(voxel::kBrickVoxels);
    for (auto rowStart = voxels.begin(); rowStart != voxels.end(); rowStart += voxel::kBrickEdge)
        std::ranges::copy(row, rowStart);
}

void addSheetBrick(VoxelScene &scene, const voxel::BrickKey &key)
{
    std::vector<core::u8> &voxels = scene.brickVoxels.emplace_back();
    fillWithSheetsAcrossX(voxels, key);
    std::vector<core::u8> &occupancy = scene.brickOccupancy.emplace_back(voxel::kOccupancyBytes);

    voxel::BrickView view{};
    view.voxels = voxels.data();
    view.key = key;
    voxel::summariseCells(view, occupancy.data());
    const bool listed = scene.mosaic.insert(view);
    LPL_VERIFY(listed);
}

void addSheetBricksOfLevel(VoxelScene &scene, core::u32 level)
{
    const core::i32 bricksAlongZ = scene.geometry.bricksAtLevel(kAxisZ, level);
    const core::i32 bricksAlongY = scene.geometry.bricksAtLevel(kAxisY, level);
    const core::i32 bricksAlongX = scene.geometry.bricksAtLevel(kAxisX, level);
    for (core::i32 z = 0; z < bricksAlongZ; ++z)
    {
        for (core::i32 y = 0; y < bricksAlongY; ++y)
        {
            for (core::i32 x = 0; x < bricksAlongX; ++x)
                addSheetBrick(scene, voxel::BrickKey{level, z, y, x});
        }
    }
}

[[nodiscard]] voxel::DensityProfile measureLevelZeroProfile(const voxel::BrickMosaic &mosaic)
{
    constexpr core::f32 kSheetQuantile = 0.10f;
    constexpr core::f32 kWindowDeviations = 1.0f;
    std::vector<voxel::BrickView> levelZeroBricks;
    for (core::u32 i = 0u; i < mosaic.count(); ++i)
        if (mosaic.at(i).key.level == 0u)
            levelZeroBricks.push_back(mosaic.at(i));
    return voxel::measureProfile(levelZeroBricks.data(), static_cast<core::u32>(levelZeroBricks.size()), kSheetQuantile,
                                 kWindowDeviations);
}

[[nodiscard]] voxel::Eye eyeInsideTheFrontFace(const voxel::VolumeGeometry &geometry)
{
    constexpr core::f32 kDepthSamples = 4.0f;
    const core::f32 metresPerSample = geometry.metresPerSample();
    const auto middleOf = [&](core::u32 axis) {
        return static_cast<core::f32>(geometry.samples[axis] / 2) * metresPerSample;
    };
    voxel::Eye eye{};
    eye.position = {middleOf(kAxisX), middleOf(kAxisY), kDepthSamples * metresPerSample};
    eye.forward = {0.0f, 0.0f, 1.0f};
    eye.right = {1.0f, 0.0f, 0.0f};
    eye.up = {0.0f, 1.0f, 0.0f};
    return eye;
}

/**
 * A cube of @p bricksPerAxis bricks of sheets along each axis at level 0, plus the @p levels - 1 coarser levels.
 * Every level covers the whole cube, so a lookup is answered by its level-0 brick.
 */
[[nodiscard]] VoxelScene makeSheetScene(core::u32 bricksPerAxis, core::u32 levels)
{
    constexpr core::f32 kScrollVoxelMicrometres = 7.91f;
    constexpr core::f32 kScrollMetresPerMicrometre = 11538.0f;
    constexpr core::f32 kSheetThicknessSamples = static_cast<core::f32>(2 * kRidgeHalfWidth - 1);
    constexpr core::f32 kOpacityAcrossASheet = 0.85f;

    VoxelScene scene;
    const core::i64 edge = static_cast<core::i64>(bricksPerAxis) * voxel::kBrickEdge;
    scene.geometry.samples[kAxisZ] = edge;
    scene.geometry.samples[kAxisY] = edge;
    scene.geometry.samples[kAxisX] = edge;
    scene.geometry.levels = levels;
    scene.geometry.voxelMicrometres = kScrollVoxelMicrometres;
    scene.geometry.metresPerMicrometre = kScrollMetresPerMicrometre;
    LPL_VERIFY(scene.geometry.valid());

    for (core::u32 level = 0u; level < levels; ++level)
        addSheetBricksOfLevel(scene, level);

    scene.profile = measureLevelZeroProfile(scene.mosaic);
    scene.transfer = voxel::rampTransfer(scene.profile.floorSample, scene.profile.sheetSample,
                                         voxel::alphaForOpaqueAfter(kSheetThicknessSamples, kOpacityAcrossASheet));
    scene.eye = eyeInsideTheFrontFace(scene.geometry);
    return scene;
}

void timeMarch(const char *label, const VoxelScene &scene, const voxel::MarchParams &params,
               std::vector<core::u32> &frame)
{
    voxel::MarchReport last{};
    const Result timing = run(label, [&]() {
        last = voxel::march(scene.mosaic, scene.geometry, scene.profile, scene.transfer, scene.eye, params,
                            frame.data(), kFrameWidth, kFrameHeight, 0u, kFrameHeight);
        doNotOptimize(frame[0]);
    });
    if (last.steps == 0u)
    {
        std::printf("        !! NO SAMPLE: the frame never entered the volume, and the time above measures nothing\n");
        return;
    }
    std::printf("        %llu samples, %llu gradients, %llu saturated, %llu escaped -> %.1f ns a sample\n",
                static_cast<unsigned long long>(last.steps), static_cast<unsigned long long>(last.gradients),
                static_cast<unsigned long long>(last.saturated), static_cast<unsigned long long>(last.escaped),
                timing.medianNs / static_cast<core::f64>(last.steps));
}

// Two working-set sizes, because the difference between them is the memory behaviour. The label prints each
// one's size: which cache it fits is a fact about the machine, not about the fixture.
void benchmarkMarch()
{
    constexpr core::f32 kNoDistanceLimitMetres = 1.0e6f;
    std::vector<core::u32> frame(static_cast<core::usize>(kFrameWidth) * kFrameHeight, 0u);
    for (const core::u32 bricksPerAxis : {2u, 3u})
    {
        const VoxelScene scene = makeSheetScene(bricksPerAxis, kSceneLevels);
        voxel::MarchParams params{};
        params.maxDistanceMetres = kNoDistanceLimitMetres;

        const core::u64 residentMebibytes = (static_cast<core::u64>(scene.mosaic.count()) * voxel::kBrickVoxels) >> 20u;
        char label[64];
        std::snprintf(label, sizeof(label), "march %ux%u, %u bricks, %llu MiB", kFrameWidth, kFrameHeight,
                      scene.mosaic.count(), static_cast<unsigned long long>(residentMebibytes));
        timeMarch(label, scene, params, frame);

        // Each summary line below times the same frame with one thing changed. That can move the sample count too,
        // because rays stop at another depth or a longer step crosses the same depth in fewer samples, so a frame
        // time alone mixes the thing's cost with the work it saved or added: the time a sample is the number to
        // compare.
        voxel::MarchParams gradientOff = params;
        gradientOff.shading = 0.0f;
        gradientOff.boundaryOpacity = 0.0f;
        timeMarch("  ... gradient off (shading, boundary)", scene, gradientOff, frame);

        voxel::MarchParams nearest = params;
        nearest.trilinear = false;
        timeMarch("  ... nearest instead of trilinear", scene, nearest, frame);

        voxel::MarchParams coarse = params;
        coarse.stepSamples = 2.0f;
        timeMarch("  ... two samples a step", scene, coarse, frame);

        voxel::MarchParams noBlend = params;
        noBlend.levelBlendSamples = 0.0f;
        timeMarch("  ... without the level blend", scene, noBlend, frame);
    }
}

/// @pre 0 <= @p position < @p edge and 0 < @p stride < @p edge.
[[nodiscard]] constexpr core::i64 advanceWrapping(core::i64 position, core::i64 stride, core::i64 edge) noexcept
{
    const core::i64 next = position + stride;
    return next < edge ? next : next - edge;
}

// The lookup on its own, because it is what the whole frame is made of. Every point of the walk is answered by a
// level-0 brick, so this times the hit path at level 0, not the fallback to coarser levels.
void benchmarkMosaicLookup()
{
    constexpr core::u32 kLookupBricksPerAxis = 3u;
    constexpr core::u32 kLookups = 1'000'000u;
    constexpr core::i64 kStrideZ = 29;
    constexpr core::i64 kStrideY = 13;
    constexpr core::i64 kStrideX = 7;
    const VoxelScene scene = makeSheetScene(kLookupBricksPerAxis, kSceneLevels);
    const core::i64 *extent = scene.geometry.samples;

    char label[64];
    std::snprintf(label, sizeof(label), "mosaic lookup %u, level-0 hits", kLookups);
    core::u32 hits = 0u;
    (void) run(label, [&]() {
        hits = 0u;
        core::i64 z = 0;
        core::i64 y = 0;
        core::i64 x = 0;
        for (core::u32 i = 0u; i < kLookups; ++i)
        {
            hits += scene.mosaic.find(z, y, x) != nullptr ? 1u : 0u;
            z = advanceWrapping(z, kStrideZ, extent[kAxisZ]);
            y = advanceWrapping(y, kStrideY, extent[kAxisY]);
            x = advanceWrapping(x, kStrideX, extent[kAxisX]);
        }
        doNotOptimize(hits);
    });
    if (hits != kLookups)
        std::printf("        !! MISSES: %u of %u lookups found a brick, so the time above is not the hit path's\n",
                    hits, kLookups);
}

void printPatchOutcome(const voxel::SheetPatch &patch, core::u32 requestedRows, core::u32 requestedColumns)
{
    static_assert(voxel::kSheetStopCount == 5u, "a new SheetStop needs its name in the summary line below");
    const auto stops = [&patch](voxel::SheetStop stop) { return patch.stops[static_cast<core::u32>(stop)]; };
    std::printf("        %u of %u rows, %u short; stops: %u budget, %u left matter, %u left resident, %u would jump, "
                "%u degenerate\n",
                patch.rows, requestedRows, patch.shortRows, stops(voxel::SheetStop::Budget),
                stops(voxel::SheetStop::LeftMatter), stops(voxel::SheetStop::LeftResident),
                stops(voxel::SheetStop::WouldJump), stops(voxel::SheetStop::Degenerate));

    const core::u32 walksOfAWholePatch = 2u * requestedRows + 2u;
    if (stops(voxel::SheetStop::Budget) != walksOfAWholePatch || patch.shortRows != 0u)
        std::printf("        !! INCOMPLETE PATCH: %u of %u walks ran their full length, so the time above is not the "
                    "time of a %ux%u patch\n",
                    stops(voxel::SheetStop::Budget), walksOfAWholePatch, requestedRows, requestedColumns);
}

// And the tracer, because it is the other thing somebody waits for. The seed sits on a ridge: between two, the field
// is flat, the seed has no normal, and the summary line would time an empty patch.
void benchmarkSheetTrace()
{
    constexpr core::u32 kTraceBricksPerAxis = 2u;
    constexpr core::u32 kTraceLevels = 1u;
    constexpr core::u32 kRows = 32u;
    constexpr core::u32 kColumns = 64u;
    const VoxelScene scene = makeSheetScene(kTraceBricksPerAxis, kTraceLevels);
    voxel::SheetTraceParams trace{};
    trace.floorSample = static_cast<core::u8>(scene.profile.mean);
    std::vector<math::Vec3<core::f32>> points(static_cast<core::usize>(kRows) * kColumns);

    const core::i64 middleX = scene.geometry.samples[kAxisX] / 2;
    const core::i64 ridgeBeforeTheMiddle = middleX - middleX % kPitch;
    const math::Vec3<core::f32> seed{static_cast<core::f32>(ridgeBeforeTheMiddle),
                                     static_cast<core::f32>(scene.geometry.samples[kAxisY] / 2),
                                     static_cast<core::f32>(scene.geometry.samples[kAxisZ] / 2)};

    char label[64];
    std::snprintf(label, sizeof(label), "traceSheetPatch %ux%u", kRows, kColumns);
    voxel::SheetPatch patch{};
    (void) run(label, [&]() {
        patch = voxel::traceSheetPatch(scene.mosaic, seed, trace, kRows, kColumns, points.data());
        doNotOptimize(patch.rows);
    });
    printPatchOutcome(patch, kRows, kColumns);
}

} // namespace

void runVoxelBenchmarks()
{
    section("voxel: direct volume rendering");
    benchmarkMarch();
    benchmarkMosaicLookup();
    benchmarkSheetTrace();
}

} // namespace lpl::bench
