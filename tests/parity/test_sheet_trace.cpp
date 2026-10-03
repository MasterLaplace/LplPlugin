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
u8 ridgeBetween(f32 distance, u8 medium, u8 peak)
{
    const f32 t = std::fabs(distance);
    if (t > 2.5f)
        return medium;
    const f32 f = 1.0f - t / 2.5f;
    return static_cast<u8>(static_cast<f32>(medium) + (static_cast<f32>(peak) - static_cast<f32>(medium)) * f * f);
}

u8 ridgeAt(f32 distance) { return ridgeBetween(distance, kMedium, kPeak); }

/// One flat sheet through the points p where dot(normal, p) == offset, with p in (x, y, z).
std::vector<u8> planeSheet(const Point &normal, f32 offset, u8 medium, u8 peak)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, medium);
    for (u32 z = 0u; z < voxel::kBrickEdge; ++z)
        for (u32 y = 0u; y < voxel::kBrickEdge; ++y)
            for (u32 x = 0u; x < voxel::kBrickEdge; ++x)
            {
                const Point p{static_cast<f32>(x), static_cast<f32>(y), static_cast<f32>(z)};
                bytes[index(z, y, x)] = ridgeBetween(normal.dot(p) - offset, medium, peak);
            }
    return bytes;
}

/// One sheet perpendicular to X at x = `centre` that ends at z = `end`: beyond it is only medium.
std::vector<u8> endingSheet(f32 centre, u32 end)
{
    std::vector<u8> bytes(voxel::kBrickVoxels, kMedium);
    for (u32 z = 0u; z < end; ++z)
        for (u32 y = 0u; y < voxel::kBrickEdge; ++y)
            for (u32 x = 0u; x < voxel::kBrickEdge; ++x)
                bytes[index(z, y, x)] = ridgeAt(static_cast<f32>(x) - centre);
    return bytes;
}

/// Neighbouring points of a patch that are distinct, along its rows and along its columns.
struct DistinctNeighbours final {
    u32 alongRows{0};
    u32 alongColumns{0};
};

DistinctNeighbours distinctNeighbours(const voxel::SheetPatch &patch)
{
    const auto apart = [](const Point &p, const Point &q) { return (q - p).lengthSquared() > 0.01f; };
    DistinctNeighbours counted{};
    for (u32 r = 0u; r < patch.rows; ++r)
        for (u32 c = 0u; c < patch.columns; ++c)
        {
            if (c + 1u < patch.columns && apart(patch.at(r, c), patch.at(r, c + 1u)))
                ++counted.alongRows;
            if (r + 1u < patch.rows && apart(patch.at(r, c), patch.at(r + 1u, c)))
                ++counted.alongColumns;
        }
    return counted;
}

/// How far, along X, a point sits from the curved sheet x = 64 + 6 sin(z / 14).
f32 distanceFromCurvedSheet(const Point &p)
{
    return std::fabs(p.x - (64.0f + 6.0f * static_cast<f32>(std::sin(static_cast<double>(p.z) / 14.0))));
}

f32 farthestFromCurvedSheet(const voxel::SheetPatch &patch)
{
    f32 worst = 0.0f;
    for (u32 r = 0u; r < patch.rows; ++r)
        for (u32 c = 0u; c < patch.columns; ++c)
            worst = std::fmax(worst, distanceFromCurvedSheet(patch.at(r, c)));
    return worst;
}

f32 meanDistanceFromCurvedSheet(const voxel::SheetPatch &patch)
{
    double total = 0.0;
    for (u32 r = 0u; r < patch.rows; ++r)
        for (u32 c = 0u; c < patch.columns; ++c)
            total += static_cast<double>(distanceFromCurvedSheet(patch.at(r, c)));
    return static_cast<f32>(total / (static_cast<double>(patch.rows) * patch.columns));
}

/// A geometry whose level-0 sample is 7.91 um, staged at the scale the renderer walks.
voxel::VolumeGeometry walkedGeometry()
{
    voxel::VolumeGeometry geometry{};
    geometry.samples[0] = 128;
    geometry.samples[1] = 128;
    geometry.samples[2] = 128;
    geometry.levels = 1u;
    geometry.voxelMicrometres = 7.91f;
    geometry.metresPerMicrometre = 11538.0f;
    return geometry;
}

/// An eye at the origin looking along +Z, with X to its right and Y up.
voxel::Eye eyeAtOrigin(f32 horizontalFieldOfView)
{
    voxel::Eye eye{};
    eye.position = {0.0f, 0.0f, 0.0f};
    eye.forward = {0.0f, 0.0f, 1.0f};
    eye.right = {1.0f, 0.0f, 0.0f};
    eye.up = {0.0f, 1.0f, 0.0f};
    eye.horizontalFieldOfView = horizontalFieldOfView;
    return eye;
}

/// A frame's depth, facing and texture buffers, cleared.
struct DepthFrame final {
    static constexpr u32 kWidth = 128u;
    static constexpr u32 kHeight = 96u;
    std::vector<f32> metres = std::vector<f32>(static_cast<std::size_t>(kWidth) * kHeight);
    std::vector<f32> facing = std::vector<f32>(metres.size());
    std::vector<f32> u = std::vector<f32>(metres.size());
    std::vector<f32> v = std::vector<f32>(metres.size());

    [[nodiscard]] voxel::SurfaceDepth view()
    {
        return voxel::SurfaceDepth{metres.data(), facing.data(), u.data(), v.data(), kWidth, kHeight};
    }

    [[nodiscard]] std::size_t pixel(u32 column, u32 row) const
    {
        return static_cast<std::size_t>(row) * kWidth + column;
    }
};

/// A quad of two triangles, corners given in metres from an eye at the origin, as a mesh in samples.
struct Quad final {
    std::vector<Point> points;
    std::vector<u32> indices{0u, 1u, 2u, 1u, 3u, 2u};

    Quad(const Point (&metres)[4], f32 metresPerSample)
    {
        for (const Point &corner : metres)
            points.push_back(corner / metresPerSample);
    }

    [[nodiscard]] voxel::SurfaceMesh mesh() const
    {
        voxel::SurfaceMesh mesh{};
        mesh.points = points.data();
        mesh.indices = indices.data();
        mesh.pointCount = static_cast<u32>(points.size());
        mesh.indexCount = static_cast<u32>(indices.size());
        return mesh;
    }
};

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
        const f32 ax = std::fabs(normal.x);
        std::printf("  normal on a flat sheet: (%.3f, %.3f, %.3f)\n", static_cast<double>(normal.x),
                    static_cast<double>(normal.y), static_cast<double>(normal.z));
        check(ax > 0.95f, "and it points across the sheet, not along it");
    }

    // A full-contrast ridge sums gradients of 255 over the whole window: the tensor's entries reach
    // the range where a square root that loses precision hands back a normal that is not unit.
    {
        const auto data = planeSheet(Point{1.0f, 0.0f, 0.0f}, 64.0f, 0u, 255u);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        Point normal{};
        check(voxel::sheetNormal(m, Point{64.0f, 64.0f, 64.0f}, 4, normal),
              "a normal is found on a full-contrast ridge");
        std::printf("  full-contrast normal length: %.6f\n", static_cast<double>(std::sqrt(normal.lengthSquared())));
        check(std::fabs(std::sqrt(normal.lengthSquared()) - 1.0f) < 1e-3f, "and it is a unit vector");
    }

    // A sheet whose normal is orthogonal to (1, 1, 1): a power iteration started from that fixed
    // direction multiplies it into nothing and reports a flat field.
    {
        const f32 s = 0.70710678f;
        const Point across{s, -s, 0.0f};
        const auto data = planeSheet(across, 0.0f, kMedium, kPeak);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        Point normal{};
        check(voxel::sheetNormal(m, Point{64.0f, 64.0f, 30.0f}, 2, normal), "a normal is found whatever its direction");
        check(std::fabs(normal.dot(across)) > 0.99f, "and it points across that sheet");
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
            worst = std::fmax(worst, std::fabs(path[i].x - sheetA));
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
            worst = std::fmax(worst, std::fabs(path[i].x - want));
            travel = std::fmax(travel, std::fabs(path[i].x - 64.0f));
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
    //
    // ⚠ And it runs under the DEFAULT parameters. A third version tightened maximumRecentre to
    // make the refusal fire, while under the defaults the search window and the limit were both
    // three samples: no correction could exceed the limit, and the default walk crossed this fault
    // without a word (review of PR #207, 2026-10-03).
    {
        const auto data = faultedSheet(60.0f, 4.0f, 64u);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        std::vector<Point> path(200);
        const Point seed{60.0f, 64.0f, 8.0f};
        const Point along{0.0f, 0.0f, 1.0f};

        voxel::SheetTraceParams loose = params;
        loose.searchRadius = 8.0f; // The control: the window reaches past the fault, and anything goes.
        loose.maximumRecentre = 100.0f;
        const voxel::SheetTrace permissive =
            voxel::traceSheet(m, seed, along, loose, path.data(), static_cast<u32>(path.size()));
        const voxel::SheetTrace guarded =
            voxel::traceSheet(m, seed, along, params, path.data(), static_cast<u32>(path.size()));
        voxel::SheetTraceParams bounded = loose;
        bounded.maximumRecentre = 1.0f;
        const voxel::SheetTrace limited =
            voxel::traceSheet(m, seed, along, bounded, path.data(), static_cast<u32>(path.size()));

        std::printf("  faulted sheet: permissive %u points (stop=%d), default %u points (stop=%d, %u refused), "
                    "limit 1 %u points (stop=%d)\n",
                    permissive.count, static_cast<int>(permissive.stop), guarded.count, static_cast<int>(guarded.stop),
                    guarded.refusedJumps, limited.count, static_cast<int>(limited.stop));
        check(guarded.stop == voxel::SheetStop::WouldJump, "the default walk stops BECAUSE it would have jumped");
        checkEq(guarded.refusedJumps, 1, "and counts the refusal");
        // The control: without the limit the same walk crosses the fault and keeps going, so it is
        // the limit that stopped it rather than the geometry running out.
        check(permissive.count > guarded.count + 10u, "without the limit, the same walk carries on past the fault");
        check(guarded.count > 40u, "and it walked most of the way there first");
        check(limited.stop == voxel::SheetStop::WouldJump,
              "a crest inside a wide window is still refused beyond maximumRecentre");
    }

    // The refusal is about a crest beyond reach, not about a field that has nothing left to follow:
    // past the end of a sheet the window is flat, and the walk has left the matter.
    {
        const auto data = endingSheet(60.0f, 80u);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        std::vector<Point> path(200);
        const voxel::SheetTrace t = voxel::traceSheet(m, Point{60.0f, 64.0f, 8.0f}, Point{0.0f, 0.0f, 1.0f}, params,
                                                      path.data(), static_cast<u32>(path.size()));
        std::printf("  sheet ending at z = 80: %u points (stop=%d)\n", t.count, static_cast<int>(t.stop));
        check(t.stop == voxel::SheetStop::LeftMatter, "a sheet that ends is left, not refused as a jump");
        checkEq(t.refusedJumps, 0, "and no refusal is counted");
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
        // The budget of 600 steps is five times the brick, so Budget would be a wrong answer too.
        check(t.stop == voxel::SheetStop::LeftResident, "running out of bricks is reported as running out of bricks");
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
        const f32 worst = farthestFromCurvedSheet(patch);
        std::printf("  worst point-to-sheet distance across the whole patch: %.3f samples\n",
                    static_cast<double>(worst));
        check(worst < 3.0f, "every point of the patch is on the sheet");

        // ⚠⚠ **A patch has to have AREA.** Distance from the sheet says nothing about whether there
        // is a patch: a grid of duplicated points sitting on the sheet passes the check above, and
        // that is what an earlier implementation returned (the story is in the body of
        // traceSheetPatch, in Sheet.cpp).
        f32 lowX = 1e30f, highX = -1e30f, lowY = 1e30f, highY = -1e30f, lowZ = 1e30f, highZ = -1e30f;
        for (u32 r = 0u; r < patch.rows; ++r)
        {
            for (u32 c = 0u; c < patch.columns; ++c)
            {
                const Point &p = patch.at(r, c);
                lowX = std::fmin(lowX, p.x);
                highX = std::fmax(highX, p.x);
                lowY = std::fmin(lowY, p.y);
                highY = std::fmax(highY, p.y);
                lowZ = std::fmin(lowZ, p.z);
                highZ = std::fmax(highZ, p.z);
            }
        }
        const DistinctNeighbours distinct = distinctNeighbours(patch);
        std::printf("  patch extent: x %.1f, y %.1f, z %.1f samples; distinct neighbours %u of %u along rows, "
                    "%u of %u along columns\n",
                    static_cast<double>(highX - lowX), static_cast<double>(highY - lowY),
                    static_cast<double>(highZ - lowZ), distinct.alongRows, patch.rows * (patch.columns - 1u),
                    distinct.alongColumns, (patch.rows - 1u) * patch.columns);
        // Half the nominal size in each in-plane direction: the sheet here is perpendicular to X,
        // so the patch spans Y (rows) and Z (columns) and is thin in X.
        check(highY - lowY > static_cast<f32>(kRows) * 0.5f, "the patch spans its rows");
        check(highZ - lowZ > static_cast<f32>(kColumns) * 0.5f, "and its columns");
        // Every pair, not most of them: no row ended short here, so a single repeated point is the
        // seed written twice, once by each of the two walks that start from it.
        checkEq(patch.shortRows, 0, "no row ended short");
        checkEq(distinct.alongRows, patch.rows * (patch.columns - 1u), "neighbours along a row are different points");
        checkEq(distinct.alongColumns, (patch.rows - 1u) * patch.columns, "and neighbouring rows are different rows");

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
        check(farthestFromCurvedSheet(relaxed) < 3.0f, "and the relaxed patch is still on the sheet");
        check(relaxed.at(0u, 0u).x == patch.at(0u, 0u).x, "the border is left where the walk put it");

        // Relaxation on the ridge asks the samples again after every pass. Pushed 0.8 samples off
        // the crest, a patch is brought back to where the walk had it; plain relaxation, whose border
        // is fixed on the crest, leaves the interior displaced.
        std::vector<Point> displacedPoints(points);
        for (u32 r = 1u; r + 1u < patch.rows; ++r)
            for (u32 c = 1u; c + 1u < patch.columns; ++c)
                displacedPoints[static_cast<std::size_t>(r) * patch.columns + c].x += 0.8f;
        voxel::SheetPatch plain = patch;
        std::vector<Point> plainPoints(displacedPoints);
        plain.points = plainPoints.data();
        voxel::relaxPatch(plain, 0.5f, 4u);
        voxel::SheetPatch onRidge = patch;
        std::vector<Point> onRidgePoints(displacedPoints);
        onRidge.points = onRidgePoints.data();
        voxel::relaxPatchOnRidge(m, onRidge, params, 0.5f, 4u);
        const f32 traced = meanDistanceFromCurvedSheet(patch);
        std::printf("  displaced 0.8 off the crest: mean distance traced %.3f, plain relax %.3f, on the ridge %.3f "
                    "(roughness %.4f)\n",
                    static_cast<double>(traced), static_cast<double>(meanDistanceFromCurvedSheet(plain)),
                    static_cast<double>(meanDistanceFromCurvedSheet(onRidge)),
                    static_cast<double>(voxel::patchRoughness(onRidge)));
        check(meanDistanceFromCurvedSheet(onRidge) < traced + 0.1f, "relaxing on the ridge puts the patch back on it");
        check(meanDistanceFromCurvedSheet(plain) > traced + 0.5f, "where plain relaxation leaves it displaced");

        // A patch with no interior point has no roughness to report, and must not report a perfect one.
        voxel::SheetPatch corner = patch;
        corner.rows = 2u;
        corner.columns = 2u;
        check(voxel::patchRoughness(corner) < 0.0f, "a patch too small to measure says so");

        // And a seed with no sheet under it yields an EMPTY patch, not a full-sized one whose
        // points are all at the origin (@see traceSheetPatch).
        std::vector<Point> elsewhere(static_cast<std::size_t>(kRows) * kColumns);
        const voxel::SheetPatch nothing =
            voxel::traceSheetPatch(m, Point{8.0f, 8.0f, 8.0f}, params, kRows, kColumns, elsewhere.data());
        checkEq(nothing.rows, 0, "a seed in the medium yields no rows");
        checkEq(nothing.columns, 0, "and no columns");
        check(nothing.points == nullptr, "and nothing to read");
    }

    // ── a patch narrower than it is tall ───────────────────────────────────
    // The row starts are walked first and parked in the patch's own storage before the rows are
    // traced over it. With few columns, the place a row start is parked can be the place another
    // one has not been read from yet.
    {
        const auto data = curvedSheet(64.0f, 6.0f, 14.0f);
        voxel::BrickMosaic m;
        m.insert(viewOf(data));
        const f32 seedX = 64.0f + 6.0f * static_cast<f32>(std::sin(64.0 / 14.0));
        for (const u32 columns : {4u, 2u, 1u})
        {
            constexpr u32 kRows = 24u;
            std::vector<Point> points(static_cast<std::size_t>(kRows) * columns);
            const voxel::SheetPatch patch =
                voxel::traceSheetPatch(m, Point{seedX, 64.0f, 64.0f}, params, kRows, columns, points.data());
            const DistinctNeighbours distinct = distinctNeighbours(patch);
            std::printf("  narrow patch %ux%u: %u of %u neighbouring rows differ\n", patch.rows, patch.columns,
                        distinct.alongColumns, (kRows - 1u) * columns);
            checkEq(patch.rows, kRows, "a narrow patch has the rows asked for");
            checkEq(distinct.alongColumns, (kRows - 1u) * columns, "and every row is its own row");
            check(farthestFromCurvedSheet(patch) < 3.0f, "and every point of it is on the sheet");
        }
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

        const voxel::VolumeGeometry geometry = walkedGeometry();
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
        std::printf("  surface: %u triangles, %u of them covered a pixel\n", written / 3u, drawn);
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

    // ── vertex normals: area-weighted, and blind to winding ────────────────
    // Two triangles in the plane x = 0, wound opposite ways: summed as they come, their normals
    // cancel at the shared edge. The plane is not perpendicular to Z, so the fallback a cancelled
    // vertex gets cannot pass for the right answer.
    {
        const std::vector<Point> points{
            {0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 1.0f},
            {5.0f, 5.0f, 5.0f}
        };
        const std::vector<u32> indices{0u, 1u, 2u, 1u, 2u, 3u};
        voxel::SurfaceMesh mesh{};
        mesh.points = points.data();
        mesh.indices = indices.data();
        mesh.pointCount = static_cast<u32>(points.size());
        mesh.indexCount = static_cast<u32>(indices.size());

        std::vector<Point> normals(points.size());
        voxel::computeVertexNormals(mesh, normals.data());
        bool acrossThePlane = true;
        for (u32 i = 0u; i < 4u; ++i)
            acrossThePlane = acrossThePlane && std::fabs(normals[i].x) > 0.999f;
        check(acrossThePlane, "every vertex of a two-sided sheet gets the normal of its plane");
        check(normals[4].z == 1.0f && normals[4].x == 0.0f && normals[4].y == 0.0f,
              "and a vertex no triangle reaches gets the fixed fallback along Z");

        const std::vector<u32> pastTheEnd{0u, 1u, 5u};
        voxel::SurfaceMesh broken = mesh;
        broken.indices = pastTheEnd.data();
        broken.indexCount = static_cast<u32>(pastTheEnd.size());
        std::vector<Point> untouched(points.size(), Point{7.0f, 7.0f, 7.0f});
        voxel::computeVertexNormals(broken, untouched.data());
        check(untouched[0].x == 7.0f && untouched[1].x == 7.0f,
              "a mesh with an index past its vertices is refused, not half-shaded");
    }

    // ── what a rasterised pixel carries ────────────────────────────────────
    {
        const voxel::VolumeGeometry geometry = walkedGeometry();
        const f32 mps = geometry.metresPerSample();
        const voxel::Eye eye = eyeAtOrigin(1.0472f);
        const u32 middleColumn = DepthFrame::kWidth / 2u;
        const u32 middleRow = DepthFrame::kHeight / 2u;

        // A square facing the eye two metres ahead: the depth is the distance along the pixel's ray,
        // and the shading comes from the vertex normals when there are some.
        const Point facingSquare[4]{
            {-0.5f, -0.5f, 2.0f},
            {0.5f,  -0.5f, 2.0f},
            {-0.5f, 0.5f,  2.0f},
            {0.5f,  0.5f,  2.0f}
        };
        const Quad square(facingSquare, mps);
        DepthFrame flat;
        voxel::clearSurfaceDepth(flat.view());
        check(voxel::rasteriseSurface(square.mesh(), geometry, eye, flat.view()) == 2u, "a square is two triangles");
        const std::size_t centre = flat.pixel(middleColumn, middleRow);
        std::printf("  square two metres ahead: depth %.4f, facing %.4f\n", static_cast<double>(flat.metres[centre]),
                    static_cast<double>(flat.facing[centre]));
        check(std::fabs(flat.metres[centre] - 2.0f) < 1e-3f, "the depth is measured along the ray");
        check(std::fabs(flat.facing[centre] - 1.0f) < 1e-4f, "without normals, a square facing the eye faces it");

        const std::vector<Point> tilted(4u, Point{0.6f, 0.0f, 0.8f});
        voxel::SurfaceMesh shaded = square.mesh();
        shaded.normals = tilted.data();
        DepthFrame smooth;
        voxel::clearSurfaceDepth(smooth.view());
        (void) voxel::rasteriseSurface(shaded, geometry, eye, smooth.view());
        check(std::fabs(smooth.facing[centre] - 0.8f) < 1e-3f, "with normals, the pixel is shaded by them");

        // An index one past the vertices names a point that is in memory and in view: drawing it
        // would look like a surface.
        Quad overrun(facingSquare, mps);
        overrun.points.push_back(overrun.points[3]);
        overrun.indices = {0u, 1u, 4u};
        voxel::SurfaceMesh beyond = overrun.mesh();
        beyond.pointCount = 4u;
        DepthFrame refused;
        voxel::clearSurfaceDepth(refused.view());
        checkEq(voxel::rasteriseSurface(beyond, geometry, eye, refused.view()), 0,
                "a mesh with an index past its vertices is refused, not drawn");

        // ⚠ A strip receding from two to six metres, textured u = 0 on its near edge and 1 on its
        // far one. The pixel straight ahead looks at the strip's middle, u = 0.5 in perspective;
        // interpolated across the screen instead, it would read about 0.75.
        const Point recedingStrip[4]{
            {-1.0f, -0.3f, 2.0f},
            {1.0f,  -0.3f, 6.0f},
            {-1.0f, 0.3f,  2.0f},
            {1.0f,  0.3f,  6.0f}
        };
        const Quad strip(recedingStrip, mps);
        const std::vector<f32> texture{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
        voxel::SurfaceMesh textured = strip.mesh();
        textured.texture = texture.data();
        textured.textureIndices = strip.indices.data();
        textured.textureCount = 4u;
        DepthFrame painted;
        voxel::clearSurfaceDepth(painted.view());
        (void) voxel::rasteriseSurface(textured, geometry, eye, painted.view());

        const f32 tanHalf = static_cast<f32>(std::tan(0.5 * 1.0472));
        const f32 aspect = static_cast<f32>(DepthFrame::kHeight) / static_cast<f32>(DepthFrame::kWidth);
        const f32 rightSlope = (2.0f * (static_cast<f32>(middleColumn) + 0.5f) / DepthFrame::kWidth - 1.0f) * tanHalf;
        const f32 upSlope =
            (1.0f - 2.0f * (static_cast<f32>(middleRow) + 0.5f) / DepthFrame::kHeight) * aspect * tanHalf;
        const f32 wantU = (1.0f + 2.0f * rightSlope) / (2.0f - 4.0f * rightSlope);
        const f32 wantV = (upSlope * (2.0f + 4.0f * wantU) + 0.3f) / 0.6f;
        std::printf("  receding strip, straight ahead: u %.4f (want %.4f), v %.4f (want %.4f)\n",
                    static_cast<double>(painted.u[centre]), static_cast<double>(wantU),
                    static_cast<double>(painted.v[centre]), static_cast<double>(wantV));
        check(std::fabs(painted.u[centre] - wantU) < 0.01f,
              "texture is interpolated in perspective, not on the screen");
        check(std::fabs(painted.v[centre] - wantV) < 0.01f, "in both coordinates");
    }

    // ── the surface lands on the pixels whose rays reach it ────────────────
    // At ninety degrees a truncated series for tan(fov / 2) is 1.3 % short of the one the marcher
    // casts its rays with, so a surface drawn with it sits up to a pixel off the scan. The edge of
    // this square crosses the image plane between two pixel centres.
    {
        const voxel::VolumeGeometry geometry = walkedGeometry();
        const voxel::Eye eye = eyeAtOrigin(1.5707963f);
        const f32 edge = 0.91f;
        const Point square[4]{
            {-0.5f, -0.3f, 1.0f},
            {edge,  -0.3f, 1.0f},
            {-0.5f, 0.3f,  1.0f},
            {edge,  0.3f,  1.0f}
        };
        const Quad quad(square, geometry.metresPerSample());
        DepthFrame frame;
        voxel::clearSurfaceDepth(frame.view());
        (void) voxel::rasteriseSurface(quad.mesh(), geometry, eye, frame.view());

        const u32 row = DepthFrame::kHeight / 2u;
        const f32 tanHalf = static_cast<f32>(std::tan(0.25 * 3.14159265358979));
        const auto rayRightSlope = [tanHalf](u32 column) {
            return (2.0f * (static_cast<f32>(column) + 0.5f) / static_cast<f32>(DepthFrame::kWidth) - 1.0f) * tanHalf;
        };
        check(rayRightSlope(121u) < edge && rayRightSlope(122u) > edge, "the edge falls between columns 121 and 122");
        check(frame.metres[frame.pixel(121u, row)] < voxel::kNoSurface,
              "the column whose ray meets the square is drawn");
        check(frame.metres[frame.pixel(122u, row)] == voxel::kNoSurface, "and the one whose ray misses it is not");
    }

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
