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
#include <lpl/voxel/Sheet.hpp>

#include <cstdio>

namespace lpl::bench {

namespace {

constexpr core::u8 kMedium = 130u;
constexpr core::u8 kSheet = 205u;

/// Sheets perpendicular to X: a ridge every `pitch` samples, in a medium that is not empty.
void fillBrick(std::vector<core::u8> &bytes, core::u32 level, core::i64 brickX)
{
    bytes.assign(voxel::kBrickVoxels, kMedium);
    const core::i64 span = voxel::brickSpanInBaseSamples(level);
    const core::i64 origin = brickX * span;
    const core::i64 step = static_cast<core::i64>(1) << level;
    constexpr core::i64 kPitch = 38; // The measured spire pitch of a Herculaneum roll, in samples.
    for (core::u32 lz = 0u; lz < voxel::kBrickEdge; ++lz)
    {
        for (core::u32 ly = 0u; ly < voxel::kBrickEdge; ++ly)
        {
            for (core::u32 lx = 0u; lx < voxel::kBrickEdge; ++lx)
            {
                const core::i64 x = origin + static_cast<core::i64>(lx) * step;
                const core::i64 phase = ((x % kPitch) + kPitch) % kPitch;
                // A ridge, not a slab: the marcher's gradient has nothing to read off a plateau,
                // and a fixture with plateaus would report a speed the real field never gives.
                const core::i64 d = phase < kPitch / 2 ? phase : kPitch - phase;
                if (d < 3)
                {
                    const core::f32 t = 1.0f - static_cast<core::f32>(d) / 3.0f;
                    bytes[(static_cast<core::usize>(lz) * voxel::kBrickEdge + ly) * voxel::kBrickEdge + lx] =
                        static_cast<core::u8>(static_cast<core::f32>(kMedium) +
                                              static_cast<core::f32>(kSheet - kMedium) * t * t);
                }
            }
        }
    }
}

} // namespace

VoxelScene makeSheetScene(core::u32 bricksPerAxis, core::u32 levels)
{
    VoxelScene scene{};
    if (bricksPerAxis == 0u)
        bricksPerAxis = 1u;
    if (levels == 0u)
        levels = 1u;

    const core::i64 edge = static_cast<core::i64>(bricksPerAxis) * voxel::kBrickEdge;
    scene.geometry.samples[0] = edge;
    scene.geometry.samples[1] = edge;
    scene.geometry.samples[2] = edge;
    scene.geometry.levels = levels;
    scene.geometry.voxelMicrometres = 7.91f;
    scene.geometry.metresPerMicrometre = 11538.0f;

    for (core::u32 level = 0u; level < levels; ++level)
    {
        const core::i32 count = scene.geometry.bricksAtLevel(0, level);
        const core::i32 span =
            count < static_cast<core::i32>(bricksPerAxis) ? count : static_cast<core::i32>(bricksPerAxis);
        for (core::i32 z = 0; z < span; ++z)
        {
            for (core::i32 y = 0; y < span; ++y)
            {
                for (core::i32 x = 0; x < span; ++x)
                {
                    scene.storage.emplace_back();
                    fillBrick(scene.storage.back(), level, x);
                }
            }
        }
    }

    // Listed only once every buffer exists: a vector that grows moves its elements, and a mosaic
    // filled during the loop would point at memory that has been relocated.
    core::usize at = 0u;
    for (core::u32 level = 0u; level < levels; ++level)
    {
        const core::i32 count = scene.geometry.bricksAtLevel(0, level);
        const core::i32 span =
            count < static_cast<core::i32>(bricksPerAxis) ? count : static_cast<core::i32>(bricksPerAxis);
        for (core::i32 z = 0; z < span; ++z)
        {
            for (core::i32 y = 0; y < span; ++y)
            {
                for (core::i32 x = 0; x < span; ++x, ++at)
                {
                    voxel::BrickView view{};
                    view.voxels = scene.storage[at].data();
                    view.key = voxel::BrickKey{level, z, y, x};
                    scene.cells.emplace_back(voxel::kOccupancyBytes);
                    voxel::summariseCells(view, scene.cells.back().data());
                    (void) scene.mosaic.insert(view);
                }
            }
        }
    }

    std::vector<voxel::BrickView> views;
    views.reserve(scene.mosaic.count());
    for (core::u32 i = 0u; i < scene.mosaic.count(); ++i)
        views.push_back(scene.mosaic.at(i));
    scene.profile = voxel::measureProfile(views.data(), static_cast<core::u32>(views.size()), 0.10f, 1.0f);
    scene.transfer = voxel::rampTransfer(scene.profile.floorSample, scene.profile.sheetSample,
                                         voxel::alphaForOpaqueAfter(5.0f, 0.85f));

    const core::f32 mps = scene.geometry.metresPerSample();
    const core::f32 middle = static_cast<core::f32>(edge / 2) * mps;
    scene.eye.position = {middle, middle, 4.0f * mps};
    scene.eye.forward = {0.0f, 0.0f, 1.0f};
    scene.eye.right = {1.0f, 0.0f, 0.0f};
    scene.eye.up = {0.0f, 1.0f, 0.0f};
    return scene;
}

void runVoxelBenchmarks()
{
    std::printf("\n=== voxel: direct volume rendering ===\n");

    constexpr core::u32 kW = 320u;
    constexpr core::u32 kH = 200u;
    std::vector<core::u32> frame(static_cast<core::usize>(kW) * kH, 0u);

    // Two working-set sizes, because the difference between them IS the memory behaviour: 27
    // bricks is 54 MiB and mostly misses, 8 is 16 MiB and mostly does not.
    for (const core::u32 bricks : {2u, 3u})
    {
        VoxelScene scene = makeSheetScene(bricks, 3u);
        char label[128];

        voxel::MarchParams params{};
        params.maxDistanceMetres = 1.0e6f;
        voxel::MarchReport last{};

        std::snprintf(label, sizeof(label), "march %ux%u, %u bricks resident", kW, kH, scene.mosaic.count());
        const Result full = run(label, [&]() {
            last = voxel::march(scene.mosaic, scene.geometry, scene.profile, scene.transfer, scene.eye, params,
                                frame.data(), kW, kH, 0u, kH);
            doNotOptimize(frame[0]);
        });
        // The counts beside the time: a frame that got faster because its rays stopped entering
        // the volume is faster and wrong, and only these tell the two apart.
        std::printf("        %llu samples, %llu gradients, %llu saturated, %llu escaped -> %.1f Msample/s\n",
                    static_cast<unsigned long long>(last.steps), static_cast<unsigned long long>(last.gradients),
                    static_cast<unsigned long long>(last.saturated), static_cast<unsigned long long>(last.escaped),
                    static_cast<core::f64>(last.steps) / (full.medianNs * 1e-9) / 1e6);

        // Where the time goes, isolated one term at a time. Each row is the same frame with one
        // thing switched off, so the difference is that thing's cost -- and the sample count
        // printed above stays put, which is what says the comparison is fair.
        voxel::MarchParams noShade = params;
        noShade.shading = 0.0f;
        noShade.boundaryOpacity = 0.0f;
        std::snprintf(label, sizeof(label), "  ... without the gradient (no shading, no boundary)");
        (void) run(label, [&]() {
            doNotOptimize(voxel::march(scene.mosaic, scene.geometry, scene.profile, scene.transfer, scene.eye, noShade,
                                       frame.data(), kW, kH, 0u, kH)
                              .steps);
        });

        voxel::MarchParams nearest = params;
        nearest.trilinear = false;
        std::snprintf(label, sizeof(label), "  ... nearest instead of trilinear");
        (void) run(label, [&]() {
            doNotOptimize(voxel::march(scene.mosaic, scene.geometry, scene.profile, scene.transfer, scene.eye, nearest,
                                       frame.data(), kW, kH, 0u, kH)
                              .steps);
        });

        voxel::MarchParams coarse = params;
        coarse.stepSamples = 2.0f;
        std::snprintf(label, sizeof(label), "  ... two samples a step");
        (void) run(label, [&]() {
            doNotOptimize(voxel::march(scene.mosaic, scene.geometry, scene.profile, scene.transfer, scene.eye, coarse,
                                       frame.data(), kW, kH, 0u, kH)
                              .steps);
        });

        voxel::MarchParams noBlend = params;
        noBlend.levelBlendSamples = 0.0f;
        std::snprintf(label, sizeof(label), "  ... without the level blend");
        (void) run(label, [&]() {
            doNotOptimize(voxel::march(scene.mosaic, scene.geometry, scene.profile, scene.transfer, scene.eye, noBlend,
                                       frame.data(), kW, kH, 0u, kH)
                              .steps);
        });
    }

    // The lookup on its own, because it is what the whole frame is made of.
    {
        VoxelScene scene = makeSheetScene(3u, 3u);
        const core::i64 edge = scene.geometry.samples[0];
        (void) run("mosaic lookup 1M, walking a line", [&]() {
            core::u32 hits = 0u;
            for (core::i64 i = 0; i < 1000000; ++i)
            {
                const core::i64 x = (i * 7) % edge;
                const core::i64 y = (i * 13) % edge;
                const core::i64 z = (i * 29) % edge;
                hits += scene.mosaic.find(z, y, x) != nullptr ? 1u : 0u;
            }
            doNotOptimize(hits);
        });
    }

    // And the tracer, because it is the other thing somebody waits for.
    {
        VoxelScene scene = makeSheetScene(2u, 1u);
        voxel::SheetTraceParams trace{};
        trace.floorSample = static_cast<core::u8>(scene.profile.mean);
        std::vector<math::Vec3<core::f32>> points(32u * 64u);
        const core::f32 middle = static_cast<core::f32>(scene.geometry.samples[0] / 2);
        (void) run("traceSheetPatch 32x64", [&]() {
            const voxel::SheetPatch patch = voxel::traceSheetPatch(
                scene.mosaic, math::Vec3<core::f32>{middle, middle, middle}, trace, 32u, 64u, points.data());
            doNotOptimize(patch.rows);
        });
    }
}

} // namespace lpl::bench
