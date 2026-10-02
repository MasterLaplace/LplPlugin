/**
 * @file test_sheet_trace.cpp
 * @brief Following a surface implied by the samples, and never finishing on the wrong one.
 *
 * @warning **The check that matters is the sheet jump.** A stack of parallel surfaces a few
 * samples apart will happily let a walk step onto the neighbour, and the path it produces is
 * smooth, plausible and about the wrong surface -- nothing in its shape says so. Every other
 * property here could hold while that one fails.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Sheet.hpp>
#include <lpl/voxel/Surface.hpp>

#include <cmath>
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
using lpl::core::u32;
using lpl::core::u8;
namespace voxel = lpl::voxel;
using Point = lpl::math::Vec3<f32>;

constexpr u8 kMedium = 120u;
constexpr u8 kPeak = 230u;

std::size_t index(u32 z, u32 y, u32 x)
{
    return (static_cast<std::size_t>(z) * voxel::kBrickEdge + y) * voxel::kBrickEdge + x;
}

/// A sheet is a ridge, not a slab: density falls off either side of its centre, which is what a
/// recentring walk has to find. A flat-topped slab has no maximum to sit on.
u8 ridgeAt(f32 distance)
{
    const f32 t = distance < 0.0f ? -distance : distance;
    if (t > 2.5f)
        return kMedium;
    const f32 f = 1.0f - t / 2.5f;
    return static_cast<u8>(static_cast<f32>(kMedium) + (static_cast<f32>(kPeak - kMedium)) * f * f);
}

/// Two flat sheets perpendicular to X, at x = centreA and centreA + spacing.
std::vector<u8> twoSheets(f32 centreA, f32 spacing)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
    for (u32 z = 0u; z < voxel::kBrickEdge; ++z)
        for (u32 y = 0u; y < voxel::kBrickEdge; ++y)
            for (u32 x = 0u; x < voxel::kBrickEdge; ++x)
            {
                const f32 fx = static_cast<f32>(x);
                const u8 a = ridgeAt(fx - centreA);
                const u8 b = ridgeAt(fx - (centreA + spacing));
                bytes[index(z, y, x)] = a > b ? a : b;
            }
    return bytes;
}

/// One sheet with a FAULT: its centre steps sideways by `offset` at z = `where`. Real stacks have
/// these, and they are the only honest way to force a large recentring on a walk that, by
/// construction, only ever travels along a sheet.
std::vector<u8> faultedSheet(f32 centre, f32 offset, u32 where)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
    for (u32 z = 0u; z < voxel::kBrickEdge; ++z)
    {
        const f32 c = centre + (z >= where ? offset : 0.0f);
        for (u32 y = 0u; y < voxel::kBrickEdge; ++y)
            for (u32 x = 0u; x < voxel::kBrickEdge; ++x)
                bytes[index(z, y, x)] = ridgeAt(static_cast<f32>(x) - c);
    }
    return bytes;
}

/// One sheet that bends: its centre follows x = centre + amplitude * sin(z / period).
std::vector<u8> curvedSheet(f32 centre, f32 amplitude, f32 period)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
    for (u32 z = 0u; z < voxel::kBrickEdge; ++z)
    {
        const f32 c = centre + amplitude * static_cast<f32>(std::sin(static_cast<double>(z) / period));
        for (u32 y = 0u; y < voxel::kBrickEdge; ++y)
            for (u32 x = 0u; x < voxel::kBrickEdge; ++x)
                bytes[index(z, y, x)] = ridgeAt(static_cast<f32>(x) - c);
    }
    return bytes;
}

voxel::BrickView viewOf(const std::vector<u8> &bytes)
{
    voxel::BrickView v{};
    v.voxels = bytes.data();
    v.key = voxel::BrickKey{0u, 0, 0, 0};
    voxel::summarise(v);
    return v;
}

} // namespace

int main()
{
    std::printf("sheet trace\n");

    voxel::SheetTraceParams params{};
    params.floorSample = kMedium + 10u;

    // ── the normal points ACROSS the sheet ─────────────────────────────────
    {
        const auto data = twoSheets(40.0f, 40.0f);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));

        Point normal{};
        check(voxel::sheetNormal(m, Point{40.0f, 64.0f, 64.0f}, 2, normal), "a normal is found on the sheet");
        // The sheets are perpendicular to X, so the normal must be X. Its sign is arbitrary: an
        // eigenvector is a direction, and reading meaning into which way it points is a bug.
        const f32 ax = normal.x < 0.0f ? -normal.x : normal.x;
        std::printf("  normal on a flat sheet: (%.3f, %.3f, %.3f)\n", static_cast<double>(normal.x),
                    static_cast<double>(normal.y), static_cast<double>(normal.z));
        check(ax > 0.95f, "and it points across the sheet, not along it");
    }

    // ── ⚠ THE ONE THAT MATTERS: two sheets six samples apart ───────────────
    {
        const f32 sheetA = 50.0f;
        const f32 spacing = 6.0f;
        const auto data = twoSheets(sheetA, spacing);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));

        std::vector<Point> path(400);
        // Walk along Z, the length of the sheet, from a seed on sheet A.
        const voxel::SheetTrace t = voxel::traceSheet(m, Point{sheetA, 64.0f, 8.0f}, Point{0.0f, 0.0f, 1.0f}, params,
                                                      path.data(), static_cast<u32>(path.size()));
        std::printf("  two sheets %.0f apart: %u points, %u recentrings, %u refusals, mean pull %.3f\n", spacing,
                    t.count, t.recentrings, t.refusedJumps,
                    t.count > 0u ? static_cast<double>(t.totalRecentre / static_cast<f32>(t.count)) : 0.0);
        check(t.count > 50u, "the walk actually walks");

        f32 worst = 0.0f;
        for (u32 i = 0u; i < t.count; ++i)
        {
            const f32 d = path[i].x - sheetA;
            const f32 a = d < 0.0f ? -d : d;
            if (a > worst)
                worst = a;
        }
        std::printf("  furthest the walk ever strayed from its own sheet: %.3f samples\n", static_cast<double>(worst));
        // Half the spacing is where the neighbour takes over. Staying well inside that is the
        // whole safety property; a walk that ends up at 6.0 has jumped and looks perfect doing it.
        check(worst < spacing * 0.5f, "the walk never crosses to the neighbouring sheet");
        check(worst < 1.5f, "and in fact stays within a sample and a half of its own");
    }

    // ── a curved sheet is followed, which is what re-projection buys ───────
    {
        const auto data = curvedSheet(64.0f, 8.0f, 12.0f);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));

        std::vector<Point> path(400);
        const voxel::SheetTrace t = voxel::traceSheet(m, Point{64.0f, 64.0f, 8.0f}, Point{0.0f, 0.0f, 1.0f}, params,
                                                      path.data(), static_cast<u32>(path.size()));
        check(t.count > 40u, "the curved sheet is walked too");

        f32 worst = 0.0f;
        f32 travel = 0.0f;
        for (u32 i = 0u; i < t.count; ++i)
        {
            const f32 want = 64.0f + 8.0f * static_cast<f32>(std::sin(static_cast<double>(path[i].z) / 12.0));
            const f32 d = path[i].x - want;
            const f32 a = d < 0.0f ? -d : d;
            if (a > worst)
                worst = a;
            const f32 off = path[i].x - 64.0f;
            const f32 ao = off < 0.0f ? -off : off;
            if (ao > travel)
                travel = ao;
        }
        std::printf("  curved sheet: %u points, worst deviation %.3f, swung %.1f samples off centre\n", t.count,
                    static_cast<double>(worst), static_cast<double>(travel));
        // The second number is the control: a walk that went straight down the middle would show a
        // tiny deviation from a straight line too, and prove nothing about following curvature.
        check(travel > 4.0f, "the walk really did follow the bend rather than run straight");
        check(worst < 2.0f, "and stayed on the sheet while doing it");
    }

    // ── the refusal fires, and it is the reason the walk stops ─────────────
    // ⚠ Two earlier versions of this check could not fail. The first passed because the walk
    // simply reached the edge of the brick. The second tried to walk ACROSS the stack, which is
    // impossible by construction -- the heading is projected into the sheet plane, so a heading
    // along the normal projects to nothing and the walk reports Degenerate before it moves.
    // A FAULT in the sheet is the honest way to force a large correction.
    {
        const auto data = faultedSheet(60.0f, 4.0f, 64u);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        std::vector<Point> path(200);

        voxel::SheetTraceParams loose = params;
        loose.maximumRecentre = 100.0f; // The control: anything goes.
        const voxel::SheetTrace permissive = voxel::traceSheet(m, Point{60.0f, 64.0f, 8.0f}, Point{0.0f, 0.0f, 1.0f},
                                                               loose, path.data(), static_cast<u32>(path.size()));
        voxel::SheetTraceParams strict = params;
        strict.maximumRecentre = 1.0f;
        const voxel::SheetTrace guarded = voxel::traceSheet(m, Point{60.0f, 64.0f, 8.0f}, Point{0.0f, 0.0f, 1.0f},
                                                            strict, path.data(), static_cast<u32>(path.size()));

        std::printf("  faulted sheet: permissive %u points (stop=%d), guarded %u points (stop=%d, %u refused)\n",
                    permissive.count, static_cast<int>(permissive.stop), guarded.count, static_cast<int>(guarded.stop),
                    guarded.refusedJumps);
        check(guarded.stop == voxel::SheetStop::WouldJump, "the guarded walk stops BECAUSE it would have jumped");
        checkEq(guarded.refusedJumps, 1, "and counts the refusal");
        // The control: without the limit the same walk crosses the fault and keeps going, so it is
        // the limit that stopped it rather than the geometry running out.
        check(permissive.count > guarded.count + 10u, "without the limit, the same walk carries on past the fault");
        check(guarded.count > 40u, "and it walked most of the way there first");
    }

    // A heading along the normal has nothing to project into the sheet, and the walk says so
    // rather than picking a direction. Guessing here would produce a path across the stack that
    // looks exactly like a path along a sheet.
    {
        const auto data = twoSheets(50.0f, 8.0f);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        std::vector<Point> path(64);
        const voxel::SheetTrace t = voxel::traceSheet(m, Point{50.0f, 64.0f, 64.0f}, Point{1.0f, 0.0f, 0.0f}, params,
                                                      path.data(), static_cast<u32>(path.size()));
        check(t.stop == voxel::SheetStop::Degenerate, "a heading across the sheet is refused, not guessed");
    }

    // ── leaving the resident set is its own answer ─────────────────────────
    {
        const auto data = twoSheets(64.0f, 40.0f);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        std::vector<Point> path(600);
        const voxel::SheetTrace t = voxel::traceSheet(m, Point{64.0f, 64.0f, 4.0f}, Point{0.0f, 0.0f, 1.0f}, params,
                                                      path.data(), static_cast<u32>(path.size()));
        // Walking the length of one brick must end at its far face, and say that is why -- not
        // "the sheet ended", which is a fact about the scroll rather than about what is in memory.
        check(t.stop == voxel::SheetStop::LeftResident || t.stop == voxel::SheetStop::Budget,
              "running out of bricks is reported as running out of bricks");
    }

    // ── a patch is what you can actually look at ───────────────────────────
    {
        const auto data = curvedSheet(64.0f, 6.0f, 14.0f);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));

        constexpr u32 kRows = 24u;
        constexpr u32 kColumns = 40u;
        std::vector<Point> points(static_cast<std::size_t>(kRows) * kColumns);
        // ⚠ Seeded ON the sheet, which at z = 64 is not at x = 64 but where the bend has taken it.
        // A seed in the medium has no normal, and the first version of this test used one -- which
        // is how the empty-patch case below came to be written.
        const f32 seedX = 64.0f + 6.0f * static_cast<f32>(std::sin(64.0 / 14.0));
        const voxel::SheetPatch patch =
            voxel::traceSheetPatch(m, Point{seedX, 64.0f, 64.0f}, params, kRows, kColumns, points.data());
        checkEq(patch.rows, kRows, "the patch has the rows asked for");
        checkEq(patch.columns, kColumns, "and the columns");
        std::printf("  patch %ux%u: %u short rows, %u refusals\n", patch.rows, patch.columns, patch.shortRows,
                    patch.refusedJumps);

        // Every point of the patch must lie on the sheet, not merely near the seed: a patch that
        // drifted would still be a smooth surface, and it would be the wrong one.
        f32 worst = 0.0f;
        for (u32 r = 0u; r < patch.rows; ++r)
        {
            for (u32 c = 0u; c < patch.columns; ++c)
            {
                const Point &p = patch.at(r, c);
                const f32 want = 64.0f + 6.0f * static_cast<f32>(std::sin(static_cast<double>(p.z) / 14.0));
                const f32 d = p.x - want;
                const f32 a = d < 0.0f ? -d : d;
                if (a > worst)
                    worst = a;
            }
        }
        std::printf("  worst point-to-sheet distance across the whole patch: %.3f samples\n",
                    static_cast<double>(worst));
        check(worst < 3.0f, "every point of the patch is on the sheet");

        // ⚠⚠ **A patch has to have AREA, and this is the check that was missing.** An earlier
        // implementation placed each row's start at a straight offset from the seed, which on a
        // curved sheet lands in the medium: the walk found nothing, and every row was padded with
        // its own start. The result was a grid of duplicated points -- a surface of zero area,
        // reporting a full size -- and because the padding sat exactly on the sheet by coincidence
        // of where the seed was, the distance check above passed. Distance from the sheet says
        // nothing at all about whether there is a patch.
        f32 lowX = 1e30f, highX = -1e30f, lowY = 1e30f, highY = -1e30f, lowZ = 1e30f, highZ = -1e30f;
        std::size_t distinct = 0;
        for (u32 r = 0u; r < patch.rows; ++r)
        {
            for (u32 c = 0u; c < patch.columns; ++c)
            {
                const Point &p = patch.at(r, c);
                lowX = p.x < lowX ? p.x : lowX;
                highX = p.x > highX ? p.x : highX;
                lowY = p.y < lowY ? p.y : lowY;
                highY = p.y > highY ? p.y : highY;
                lowZ = p.z < lowZ ? p.z : lowZ;
                highZ = p.z > highZ ? p.z : highZ;
                if (c + 1u < patch.columns)
                {
                    const Point &q = patch.at(r, c + 1u);
                    const f32 dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
                    if (dx * dx + dy * dy + dz * dz > 0.01f)
                        ++distinct;
                }
            }
        }
        std::printf("  patch extent: x %.1f, y %.1f, z %.1f samples; %zu neighbouring pairs differ of %u\n",
                    static_cast<double>(highX - lowX), static_cast<double>(highY - lowY),
                    static_cast<double>(highZ - lowZ), distinct, patch.rows * (patch.columns - 1u));
        // Half the nominal size in each in-plane direction: the sheet here is perpendicular to X,
        // so the patch spans Y (rows) and Z (columns) and is thin in X.
        check(highY - lowY > static_cast<f32>(kRows) * 0.5f, "the patch spans its rows");
        check(highZ - lowZ > static_cast<f32>(kColumns) * 0.5f, "and its columns");
        check(distinct > patch.rows * (patch.columns - 1u) * 9u / 10u,
              "and neighbouring points are actually different points");

        // ── how corrugated it is, and what relaxing does to that ───────────
        // "The trace looks crumpled" and "the sheet is crumpled" are different claims, and only
        // one of them is about the scroll. The number is what separates them.
        const f32 before = voxel::patchRoughness(patch);
        voxel::SheetPatch relaxed = patch;
        std::vector<Point> relaxedPoints(points);
        relaxed.points = relaxedPoints.data();
        voxel::relaxPatch(relaxed, 0.5f, 4u);
        const f32 after = voxel::patchRoughness(relaxed);
        std::printf("  roughness: %.4f samples, relaxed %.4f\n", static_cast<double>(before),
                    static_cast<double>(after));
        check(after < before || before < 1e-4f, "relaxing a patch makes it smoother, or it was already smooth");

        // ⚠ And it must not walk off the sheet doing it: a relaxation that flattened the surface
        // into its own average plane would score beautifully on roughness and be a different
        // surface. The border is held fixed for the same reason.
        f32 relaxedWorst = 0.0f;
        for (u32 r = 0u; r < relaxed.rows; ++r)
        {
            for (u32 c = 0u; c < relaxed.columns; ++c)
            {
                const Point &p = relaxed.at(r, c);
                const f32 want = 64.0f + 6.0f * static_cast<f32>(std::sin(static_cast<double>(p.z) / 14.0));
                const f32 d = p.x - want;
                const f32 a = d < 0.0f ? -d : d;
                if (a > relaxedWorst)
                    relaxedWorst = a;
            }
        }
        check(relaxedWorst < 3.0f, "and the relaxed patch is still on the sheet");
        check(relaxed.at(0u, 0u).x == patch.at(0u, 0u).x, "the border is left where the walk put it");

        // And a seed with no sheet under it yields an EMPTY patch, not a full-sized one whose
        // points are all at the origin. The second is a surface that looks valid, sits in the
        // corner of the volume, and is entirely fictional -- which is what this returned before.
        std::vector<Point> elsewhere(static_cast<std::size_t>(kRows) * kColumns);
        const voxel::SheetPatch nothing =
            voxel::traceSheetPatch(m, Point{8.0f, 8.0f, 8.0f}, params, kRows, kColumns, elsewhere.data());
        checkEq(nothing.rows, 0, "a seed in the medium yields no rows");
        checkEq(nothing.columns, 0, "and no columns");
        check(nothing.points == nullptr, "and nothing to read");
    }

    // ── the surface, drawn inside the scan it came from ────────────────────
    // This is the measurement the whole tool is for: a traced sheet judged against the samples
    // around it, at the scale a body walks, rather than one flat slice at a time.
    {
        const auto data = curvedSheet(64.0f, 6.0f, 14.0f);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));

        constexpr u32 kRows = 32u;
        constexpr u32 kColumns = 48u;
        std::vector<Point> points(static_cast<std::size_t>(kRows) * kColumns);
        const f32 seedX = 64.0f + 6.0f * static_cast<f32>(std::sin(64.0 / 14.0));
        const voxel::SheetPatch patch =
            voxel::traceSheetPatch(m, Point{seedX, 64.0f, 64.0f}, params, kRows, kColumns, points.data());
        check(patch.rows == kRows, "the patch traced");

        std::printf("  patch corners: (%.1f,%.1f,%.1f) (%.1f,%.1f,%.1f) (%.1f,%.1f,%.1f)\n", (double) patch.at(0, 0).x,
                    (double) patch.at(0, 0).y, (double) patch.at(0, 0).z, (double) patch.at(0, 1).x,
                    (double) patch.at(0, 1).y, (double) patch.at(0, 1).z, (double) patch.at(1, 0).x,
                    (double) patch.at(1, 0).y, (double) patch.at(1, 0).z);
        std::vector<u32> indices(static_cast<std::size_t>(kRows) * kColumns * 6u);
        const u32 written = voxel::patchIndices(patch, indices.data(), static_cast<u32>(indices.size()));
        check(written > 0u, "and turned into triangles");
        checkEq(written % 3, 0, "a whole number of them");

        voxel::SurfaceMesh mesh{};
        mesh.points = patch.points;
        mesh.indices = indices.data();
        mesh.pointCount = kRows * kColumns;
        mesh.indexCount = written;

        voxel::VolumeGeometry geometry{};
        geometry.samples[0] = 128;
        geometry.samples[1] = 128;
        geometry.samples[2] = 128;
        geometry.levels = 1u;
        geometry.voxelMicrometres = 7.91f;
        geometry.metresPerMicrometre = 11538.0f;
        const f32 mps = geometry.metresPerSample();

        constexpr u32 kW = 128u;
        constexpr u32 kH = 96u;
        std::vector<f32> depthMetres(static_cast<std::size_t>(kW) * kH);
        std::vector<f32> facing(depthMetres.size());
        voxel::SurfaceDepth depth{depthMetres.data(), facing.data(), nullptr, nullptr, kW, kH};
        voxel::clearSurfaceDepth(depth);

        voxel::Eye eye{};
        // Looking along Z at the patch, from outside the volume.
        eye.position = {64.0f * mps, 64.0f * mps, -30.0f * mps};
        eye.forward = {0.0f, 0.0f, 1.0f};
        eye.right = {1.0f, 0.0f, 0.0f};
        eye.up = {0.0f, 1.0f, 0.0f};

        const u32 drawn = voxel::rasteriseSurface(mesh, geometry, eye, depth);
        std::printf("  surface: %u triangles, %u covered pixels\n", written / 3u, drawn);
        check(drawn > 0u, "the surface covers pixels");

        std::size_t covered = 0;
        for (f32 d : depthMetres)
        {
            if (d < voxel::kNoSurface)
                ++covered;
        }
        check(covered > 200u, "and a decent patch of the frame");

        voxel::DensityProfile profile{};
        profile.floorSample = kMedium + 20u;
        profile.sheetSample = kPeak - 10u;
        profile.mean = static_cast<f32>(kMedium);
        profile.deviation = 20.0f;
        for (u32 lv = 0u; lv < voxel::kMaxPyramidLevels; ++lv)
            profile.spreadRatio[lv] = 1.0f;
        const voxel::TransferFunction curve =
            voxel::rampTransfer(profile.floorSample, profile.sheetSample, voxel::alphaForOpaqueAfter(5.0f, 0.85f));

        std::vector<u32> frame(static_cast<std::size_t>(kW) * kH, 0u);
        voxel::MarchParams march{};
        march.maxDistanceMetres = 1.0e6f;

        const voxel::MarchReport without =
            voxel::march(m, geometry, profile, curve, eye, march, frame.data(), kW, kH, 0u, kH);
        checkEq(static_cast<long long>(without.surfaceHits), 0, "no surface given, none is drawn");

        march.surfaces[0].depth = depthMetres.data();
        march.surfaces[0].facing = facing.data();
        march.surfaceCount = 1u;
        const voxel::MarchReport with =
            voxel::march(m, geometry, profile, curve, eye, march, frame.data(), kW, kH, 0u, kH);
        std::printf("  rays that crossed the surface: %llu of %llu\n",
                    static_cast<unsigned long long>(with.surfaceHits), static_cast<unsigned long long>(with.rays));
        check(with.surfaceHits > 100u, "given one, rays cross it");

        // ⚠ The property that makes it a measurement rather than a decoration: matter in FRONT of
        // the surface still occludes it. A surface drawn on top regardless would hide the very
        // evidence it exists to be checked against.
        std::vector<f32> nearDepth(depthMetres.size(), 1.0e-3f); // A surface right at the eye.
        std::vector<f32> nearFacing(depthMetres.size(), 1.0f);
        march.surfaces[0].depth = nearDepth.data();
        march.surfaces[0].facing = nearFacing.data();
        march.surfaceCount = 1u;
        std::vector<u32> nearFrame(frame.size(), 0u);
        voxel::march(m, geometry, profile, curve, eye, march, nearFrame.data(), kW, kH, 0u, kH);

        std::vector<f32> farDepth(depthMetres.size(), 1.0e6f); // Behind everything.
        march.surfaces[0].depth = farDepth.data();
        std::vector<u32> farFrame(frame.size(), 0u);
        voxel::march(m, geometry, profile, curve, eye, march, farFrame.data(), kW, kH, 0u, kH);

        auto blueness = [](const std::vector<u32> &px) {
            double b = 0.0;
            for (u32 p : px)
                b += static_cast<double>(p & 0xFFu);
            return b / static_cast<double>(px.size());
        };
        std::printf("  surface at the eye -> blue %.1f, behind everything -> blue %.1f\n", blueness(nearFrame),
                    blueness(farFrame));
        check(blueness(nearFrame) > blueness(farFrame) + 5.0,
              "a surface in front is visible and one behind the matter is not");
    }

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
