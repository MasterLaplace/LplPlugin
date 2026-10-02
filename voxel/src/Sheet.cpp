/**
 * @file Sheet.cpp
 * @brief Structure tensor, parabolic recentring, and the refusal that keeps a walk honest.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/std/vector.hpp>
#include <lpl/voxel/Sheet.hpp>

namespace lpl::voxel {

namespace {

[[nodiscard]] core::i64 floorToI64(core::f32 v) noexcept
{
    const core::i64 t = static_cast<core::i64>(v);
    return (v < 0.0f && static_cast<core::f32>(t) != v) ? t - 1 : t;
}

/// Nearest sample, or -1 where nothing is resident. Negative is distinguishable from any density.
[[nodiscard]] core::f32 sampleAt(const BrickMosaic &mosaic, core::f32 z, core::f32 y, core::f32 x) noexcept
{
    const core::i64 iz = floorToI64(z);
    const core::i64 iy = floorToI64(y);
    const core::i64 ix = floorToI64(x);
    const BrickView *brick = mosaic.find(iz, iy, ix);
    if (brick == nullptr)
        return -1.0f;

    const core::u32 level = brick->key.level;
    const core::i64 span = brickSpanInBaseSamples(level);
    const core::i64 step = static_cast<core::i64>(1) << level;
    const core::i64 lz = (iz - static_cast<core::i64>(brick->key.z) * span) / step;
    const core::i64 ly = (iy - static_cast<core::i64>(brick->key.y) * span) / step;
    const core::i64 lx = (ix - static_cast<core::i64>(brick->key.x) * span) / step;
    const core::i64 edge = static_cast<core::i64>(kBrickEdge);
    if (lz < 0 || ly < 0 || lx < 0 || lz >= edge || ly >= edge || lx >= edge)
        return -1.0f;
    return static_cast<core::f32>(
        brick->at(static_cast<core::u32>(lz), static_cast<core::u32>(ly), static_cast<core::u32>(lx)));
}

[[nodiscard]] core::f32 squareRoot(core::f32 v) noexcept
{
    if (v <= 0.0f)
        return 0.0f;
    core::f32 g = v > 1.0f ? v : 1.0f;
    for (int i = 0; i < 24; ++i)
        g = 0.5f * (g + v / g);
    return g;
}

[[nodiscard]] bool normalise(math::Vec3<core::f32> &v) noexcept
{
    const core::f32 len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len2 < 1e-12f)
        return false;
    const core::f32 inv = 1.0f / squareRoot(len2);
    v.x *= inv;
    v.y *= inv;
    v.z *= inv;
    return true;
}

} // namespace

bool sheetNormal(const BrickMosaic &mosaic, const math::Vec3<core::f32> &at, core::i32 radius,
                 math::Vec3<core::f32> &outNormal) noexcept
{
    if (radius < 1)
        radius = 1;

    // Structure tensor: the outer product of the gradient, summed over a window. Its dominant
    // eigenvector is the direction the field changes fastest in, which across a sheet is its
    // normal -- and unlike a single finite difference it does not wander with noise inside the
    // matter, which is exactly where a walk needs it.
    core::f32 t[6]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // zz, zy, zx, yy, yx, xx
    core::u32 samples = 0u;
    for (core::i32 dz = -radius; dz <= radius; ++dz)
    {
        for (core::i32 dy = -radius; dy <= radius; ++dy)
        {
            for (core::i32 dx = -radius; dx <= radius; ++dx)
            {
                const core::f32 pz = at.z + static_cast<core::f32>(dz);
                const core::f32 py = at.y + static_cast<core::f32>(dy);
                const core::f32 px = at.x + static_cast<core::f32>(dx);

                const core::f32 zp = sampleAt(mosaic, pz + 1.0f, py, px);
                const core::f32 zm = sampleAt(mosaic, pz - 1.0f, py, px);
                const core::f32 yp = sampleAt(mosaic, pz, py + 1.0f, px);
                const core::f32 ym = sampleAt(mosaic, pz, py - 1.0f, px);
                const core::f32 xp = sampleAt(mosaic, pz, py, px + 1.0f);
                const core::f32 xm = sampleAt(mosaic, pz, py, px - 1.0f);
                if (zp < 0.0f || zm < 0.0f || yp < 0.0f || ym < 0.0f || xp < 0.0f || xm < 0.0f)
                    continue; // A missing neighbour contributes nothing rather than a false cliff.

                const core::f32 gz = zp - zm;
                const core::f32 gy = yp - ym;
                const core::f32 gx = xp - xm;
                t[0] += gz * gz;
                t[1] += gz * gy;
                t[2] += gz * gx;
                t[3] += gy * gy;
                t[4] += gy * gx;
                t[5] += gx * gx;
                ++samples;
            }
        }
    }
    if (samples == 0u || (t[0] + t[3] + t[5]) < 1e-6f)
        return false;

    // Power iteration for the dominant eigenvector. A closed form would need a cube root; this
    // needs nothing but multiplication, and the answer is a direction so its sign is arbitrary.
    math::Vec3<core::f32> v{0.577f, 0.577f, 0.577f};
    for (int i = 0; i < 24; ++i)
    {
        const math::Vec3<core::f32> w{t[2] * v.z + t[4] * v.y + t[5] * v.x, t[1] * v.z + t[3] * v.y + t[4] * v.x,
                                      t[0] * v.z + t[1] * v.y + t[2] * v.x};
        v = w;
        if (!normalise(v))
            return false;
    }
    outNormal = v;
    return true;
}

namespace {

/**
 * Slides @p point along @p normal onto the local ridge, sub-sample by parabolic fit.
 *
 * @return The distance moved, or a negative number when there is no ridge to sit on.
 */
/// Field value averaged over a small window: the same question, asked of a less noisy field.
[[nodiscard]] core::f32 smoothedSample(const BrickMosaic &mosaic, core::f32 z, core::f32 y, core::f32 x,
                                       core::i32 radius) noexcept
{
    if (radius <= 0)
        return sampleAt(mosaic, z, y, x);
    core::f32 total = 0.0f;
    core::u32 count = 0u;
    for (core::i32 dz = -radius; dz <= radius; ++dz)
    {
        for (core::i32 dy = -radius; dy <= radius; ++dy)
        {
            for (core::i32 dx = -radius; dx <= radius; ++dx)
            {
                const core::f32 v = sampleAt(mosaic, z + static_cast<core::f32>(dz), y + static_cast<core::f32>(dy),
                                             x + static_cast<core::f32>(dx));
                if (v < 0.0f)
                    continue; // A missing neighbour contributes nothing rather than a false zero.
                total += v;
                ++count;
            }
        }
    }
    return count == 0u ? -1.0f : total / static_cast<core::f32>(count);
}

[[nodiscard]] core::f32 recentre(const BrickMosaic &mosaic, math::Vec3<core::f32> &point,
                                 const math::Vec3<core::f32> &normal, core::f32 searchRadius,
                                 core::i32 ridgeSmoothing) noexcept
{
    core::i32 best = 0;
    core::f32 bestValue = -1.0f;
    const core::i32 span = static_cast<core::i32>(searchRadius);
    for (core::i32 i = -span; i <= span; ++i)
    {
        const core::f32 s = static_cast<core::f32>(i);
        const core::f32 v = smoothedSample(mosaic, point.z + normal.z * s, point.y + normal.y * s,
                                           point.x + normal.x * s, ridgeSmoothing);
        if (v > bestValue)
        {
            bestValue = v;
            best = i;
        }
    }
    if (bestValue < 0.0f)
        return -1.0f;

    // Sub-sample by fitting a parabola through the peak and its two neighbours. Without it the
    // walk quantises to whole samples and drifts off the sheet a fraction at a time.
    core::f32 offset = static_cast<core::f32>(best);
    if (best > -span && best < span)
    {
        const core::f32 a =
            smoothedSample(mosaic, point.z + normal.z * (offset - 1.0f), point.y + normal.y * (offset - 1.0f),
                           point.x + normal.x * (offset - 1.0f), ridgeSmoothing);
        const core::f32 b = bestValue;
        const core::f32 c =
            smoothedSample(mosaic, point.z + normal.z * (offset + 1.0f), point.y + normal.y * (offset + 1.0f),
                           point.x + normal.x * (offset + 1.0f), ridgeSmoothing);
        if (a >= 0.0f && c >= 0.0f)
        {
            const core::f32 denominator = a - 2.0f * b + c;
            if (denominator < -1e-6f || denominator > 1e-6f)
            {
                const core::f32 shift = 0.5f * (a - c) / denominator;
                if (shift > -1.0f && shift < 1.0f)
                    offset += shift;
            }
        }
    }

    point.z += normal.z * offset;
    point.y += normal.y * offset;
    point.x += normal.x * offset;
    return offset < 0.0f ? -offset : offset;
}

} // namespace

SheetTrace traceSheet(const BrickMosaic &mosaic, const math::Vec3<core::f32> &seed,
                      const math::Vec3<core::f32> &heading, const SheetTraceParams &params, math::Vec3<core::f32> *out,
                      core::u32 capacity) noexcept
{
    SheetTrace trace{};
    if (out == nullptr || capacity == 0u)
        return trace;

    math::Vec3<core::f32> point = seed;
    math::Vec3<core::f32> direction = heading;
    if (!normalise(direction))
    {
        trace.stop = SheetStop::Degenerate;
        return trace;
    }

    math::Vec3<core::f32> normal{};
    if (!sheetNormal(mosaic, point, params.tensorRadius, normal))
    {
        trace.stop = SheetStop::Degenerate;
        return trace;
    }
    // Snap onto the ridge before the first step, with a WIDER search: finding the sheet and
    // staying on it are different problems. See SheetTraceParams::seedSearchRadius.
    (void) recentre(mosaic, point, normal,
                    params.seedSearchRadius > params.searchRadius ? params.seedSearchRadius : params.searchRadius,
                    params.ridgeSmoothing);

    out[trace.count++] = point;
    const core::u32 limit = params.maximumSteps < capacity ? params.maximumSteps : capacity;

    for (core::u32 step = 1u; step < limit; ++step)
    {
        if (!sheetNormal(mosaic, point, params.tensorRadius, normal))
        {
            trace.stop = SheetStop::Degenerate;
            break;
        }

        // Project the heading into the sheet plane. Re-projecting every step is what makes the
        // walk follow curvature rather than leave on a tangent.
        const core::f32 along = direction.x * normal.x + direction.y * normal.y + direction.z * normal.z;
        math::Vec3<core::f32> inPlane{direction.x - normal.x * along, direction.y - normal.y * along,
                                      direction.z - normal.z * along};
        if (!normalise(inPlane))
        {
            trace.stop = SheetStop::Degenerate;
            break;
        }
        direction = inPlane;

        math::Vec3<core::f32> next{point.x + direction.x * params.step, point.y + direction.y * params.step,
                                   point.z + direction.z * params.step};

        const core::f32 here = sampleAt(mosaic, next.z, next.y, next.x);
        if (here < 0.0f)
        {
            trace.stop = SheetStop::LeftResident;
            break;
        }

        math::Vec3<core::f32> correctedNormal{};
        if (!sheetNormal(mosaic, next, params.tensorRadius, correctedNormal))
        {
            trace.stop = SheetStop::Degenerate;
            break;
        }
        const core::f32 moved = recentre(mosaic, next, correctedNormal, params.searchRadius, params.ridgeSmoothing);
        if (moved < 0.0f)
        {
            trace.stop = SheetStop::LeftMatter;
            break;
        }

        // ⚠ The safety property. A correction of a fraction of a sample is a follow; a correction
        // the size of an interline is a change of sheet, and the path it produces is smooth,
        // plausible and about the wrong surface.
        if (moved > params.maximumRecentre)
        {
            ++trace.refusedJumps;
            trace.stop = SheetStop::WouldJump;
            break;
        }

        if (sampleAt(mosaic, next.z, next.y, next.x) < static_cast<core::f32>(params.floorSample))
        {
            trace.stop = SheetStop::LeftMatter;
            break;
        }

        if (moved > 1e-4f)
            ++trace.recentrings;
        trace.totalRecentre += moved;
        point = next;
        out[trace.count++] = point;
    }
    return trace;
}

SheetPatch traceSheetPatch(const BrickMosaic &mosaic, const math::Vec3<core::f32> &seed, const SheetTraceParams &params,
                           core::u32 rows, core::u32 columns, math::Vec3<core::f32> *out) noexcept
{
    SheetPatch patch{};
    if (out == nullptr || rows == 0u || columns == 0u)
        return patch;

    // ⚠ The dimensions are published only once something has been written into them. An earlier
    // version set them first and returned early when the seed had no normal, which handed back a
    // patch that reported its full size with every point left at the origin -- a surface that
    // looks valid, sits at the corner of the volume, and is entirely fictional.
    math::Vec3<core::f32> normal{};
    if (!sheetNormal(mosaic, seed, params.tensorRadius, normal))
        return patch;

    // Two in-plane axes from the normal. Which two is arbitrary -- a sheet has no preferred
    // direction -- so they are built from whichever world axis is least aligned with the normal,
    // which is the standard way to avoid a degenerate cross product.
    math::Vec3<core::f32> helper{1.0f, 0.0f, 0.0f};
    if (normal.x > 0.9f || normal.x < -0.9f)
        helper = math::Vec3<core::f32>{0.0f, 1.0f, 0.0f};
    math::Vec3<core::f32> u{normal.y * helper.z - normal.z * helper.y, normal.z * helper.x - normal.x * helper.z,
                            normal.x * helper.y - normal.y * helper.x};
    if (!normalise(u))
        return patch;
    math::Vec3<core::f32> v{normal.y * u.z - normal.z * u.y, normal.z * u.x - normal.x * u.z,
                            normal.x * u.y - normal.y * u.x};
    if (!normalise(v))
        return patch;

    patch.rows = rows;
    patch.columns = columns;
    patch.points = out;

    // ⚠⚠ **Every seed is reached by WALKING, never by stepping in a straight line.** A first
    // version placed each row's start at a straight offset from the seed, and on a sheet with any
    // curvature at all that lands in the medium a few samples away -- so the walk found no normal,
    // returned nothing, and each row was padded with its own start. The patch came out as a grid
    // of duplicated points: a surface with zero area that reported a full size and, on the fixture
    // it was tested against, sat exactly on the sheet by coincidence of where the seed was. The
    // test passed. Growing the patch by tracing is not a refinement; it is the only way the seeds
    // stay on the surface.
    const core::u32 halfRows = rows / 2u;
    const core::u32 halfColumns = columns / 2u;

    // The spine: two guarded walks from the seed, one each way along v. Their points are the row
    // starts, and each one is on the sheet because it was walked there.
    core::u32 spineCount = 0u;
    for (core::u32 side = 0u; side < 2u; ++side)
    {
        const math::Vec3<core::f32> heading = side == 0u ? v : math::Vec3<core::f32>{-v.x, -v.y, -v.z};
        const core::u32 want = side == 0u ? rows - halfRows : halfRows;
        if (want == 0u)
            continue;
        // Written into the patch's own storage first, then read back out: no allocation, and the
        // rows are overwritten by their traces immediately afterwards.
        math::Vec3<core::f32> *scratch = out;
        const SheetTrace t = traceSheet(mosaic, seed, heading, params, scratch, want);
        patch.refusedJumps += t.refusedJumps;
        ++patch.stops[static_cast<core::u32>(t.stop)];
        for (core::u32 i = 0u; i < want; ++i)
        {
            // A spine that stopped short repeats its last point: the rows there will be traced
            // from the same place and come out identical, which is a visible flat edge rather
            // than an invented one.
            const math::Vec3<core::f32> point =
                i < t.count ? scratch[i] : (t.count > 0u ? scratch[t.count - 1u] : seed);
            const core::u32 row = side == 0u ? halfRows + i : halfRows - 1u - i;
            if (row >= rows)
                continue;
            // Parked at the row's first slot; the row trace below overwrites the whole row.
            out[static_cast<core::usize>(row) * columns] = point;
            ++spineCount;
        }
    }
    if (spineCount == 0u)
    {
        patch.rows = 0u;
        patch.columns = 0u;
        patch.points = nullptr;
        return patch;
    }

    for (core::u32 r = 0u; r < rows; ++r)
    {
        math::Vec3<core::f32> *row = &out[static_cast<core::usize>(r) * columns];
        const math::Vec3<core::f32> rowSeed = row[0];

        // Two walks again, and for the same reason: reaching the far end of a row by stepping
        // straight there would leave the sheet exactly as it did for the spine.
        math::Vec3<core::f32> back[kMaxPatchWidth];
        const core::u32 wantBack = halfColumns < kMaxPatchWidth ? halfColumns : kMaxPatchWidth;
        const SheetTrace tb = wantBack > 0u ? traceSheet(mosaic, rowSeed, math::Vec3<core::f32>{-u.x, -u.y, -u.z},
                                                         params, back, wantBack) :
                                              SheetTrace{};
        math::Vec3<core::f32> forward[kMaxPatchWidth];
        const core::u32 wantForward =
            (columns - halfColumns) < kMaxPatchWidth ? (columns - halfColumns) : kMaxPatchWidth;
        const SheetTrace tf = traceSheet(mosaic, rowSeed, u, params, forward, wantForward);
        patch.refusedJumps += tb.refusedJumps + tf.refusedJumps;
        ++patch.stops[static_cast<core::u32>(tb.stop)];
        ++patch.stops[static_cast<core::u32>(tf.stop)];

        bool short_ = false;
        for (core::u32 c = 0u; c < columns; ++c)
        {
            if (c < halfColumns)
            {
                const core::u32 i = halfColumns - 1u - c; // Reversed: the back walk runs outward.
                if (i < tb.count)
                    row[c] = back[i];
                else
                {
                    row[c] = tb.count > 0u ? back[tb.count - 1u] : rowSeed;
                    short_ = true;
                }
            }
            else
            {
                const core::u32 i = c - halfColumns;
                if (i < tf.count)
                    row[c] = forward[i];
                else
                {
                    row[c] = tf.count > 0u ? forward[tf.count - 1u] : rowSeed;
                    short_ = true;
                }
            }
        }
        if (short_)
            ++patch.shortRows;
    }
    return patch;
}

void relaxPatchOnRidge(const BrickMosaic &mosaic, SheetPatch &patch, const SheetTraceParams &params, core::f32 strength,
                       core::u32 iterations) noexcept
{
    if (patch.points == nullptr || patch.rows < 3u || patch.columns < 3u)
        return;
    for (core::u32 pass = 0u; pass < iterations; ++pass)
    {
        relaxPatch(patch, strength, 1u);
        // Back onto the crest. The search is TIGHT -- a wide one here would let a point that the
        // smoothing nudged toward the neighbouring sheet settle on it, which is the jump this
        // whole file exists to refuse, arriving by the back door.
        for (core::u32 r = 1u; r + 1u < patch.rows; ++r)
        {
            for (core::u32 c = 1u; c + 1u < patch.columns; ++c)
            {
                math::Vec3<core::f32> &p = patch.points[static_cast<core::usize>(r) * patch.columns + c];
                math::Vec3<core::f32> normal{};
                if (!sheetNormal(mosaic, p, params.tensorRadius, normal))
                    continue;
                math::Vec3<core::f32> moved = p;
                const core::f32 pull = recentre(mosaic, moved, normal, 1.5f, params.ridgeSmoothing);
                if (pull >= 0.0f && pull <= params.maximumRecentre)
                    p = moved;
            }
        }
    }
}

core::f32 patchRoughness(const SheetPatch &patch) noexcept
{
    if (patch.points == nullptr || patch.rows < 3u || patch.columns < 3u)
        return 0.0f;
    core::f64 total = 0.0;
    core::u32 count = 0u;
    for (core::u32 r = 1u; r + 1u < patch.rows; ++r)
    {
        for (core::u32 c = 1u; c + 1u < patch.columns; ++c)
        {
            const math::Vec3<core::f32> &p = patch.at(r, c);
            const math::Vec3<core::f32> &n = patch.at(r - 1u, c);
            const math::Vec3<core::f32> &s = patch.at(r + 1u, c);
            const math::Vec3<core::f32> &w = patch.at(r, c - 1u);
            const math::Vec3<core::f32> &e = patch.at(r, c + 1u);
            const core::f32 dx = (n.x + s.x + w.x + e.x) * 0.25f - p.x;
            const core::f32 dy = (n.y + s.y + w.y + e.y) * 0.25f - p.y;
            const core::f32 dz = (n.z + s.z + w.z + e.z) * 0.25f - p.z;
            total += static_cast<core::f64>(dx * dx + dy * dy + dz * dz);
            ++count;
        }
    }
    if (count == 0u)
        return 0.0f;
    const core::f64 mean = total / static_cast<core::f64>(count);
    core::f64 g = mean > 1.0 ? mean : 1.0;
    for (int i = 0; i < 24; ++i)
        g = 0.5 * (g + mean / g);
    return static_cast<core::f32>(g);
}

void relaxPatch(SheetPatch &patch, core::f32 strength, core::u32 iterations) noexcept
{
    if (patch.points == nullptr || patch.rows < 3u || patch.columns < 3u || strength <= 0.0f)
        return;
    if (strength > 1.0f)
        strength = 1.0f;

    // Jacobi, not Gauss-Seidel: reading points already moved this pass would make the result
    // depend on the order the grid is walked, and the same patch would relax differently
    // depending on which corner the loop started from.
    lpl::pmr::vector<math::Vec3<core::f32>> scratch(static_cast<core::usize>(patch.rows) * patch.columns);
    for (core::u32 pass = 0u; pass < iterations; ++pass)
    {
        for (core::usize i = 0u; i < scratch.size(); ++i)
            scratch[i] = patch.points[i];
        for (core::u32 r = 1u; r + 1u < patch.rows; ++r)
        {
            for (core::u32 c = 1u; c + 1u < patch.columns; ++c)
            {
                const math::Vec3<core::f32> &n = scratch[static_cast<core::usize>(r - 1u) * patch.columns + c];
                const math::Vec3<core::f32> &s = scratch[static_cast<core::usize>(r + 1u) * patch.columns + c];
                const math::Vec3<core::f32> &w = scratch[static_cast<core::usize>(r) * patch.columns + c - 1u];
                const math::Vec3<core::f32> &e = scratch[static_cast<core::usize>(r) * patch.columns + c + 1u];
                math::Vec3<core::f32> &p = patch.points[static_cast<core::usize>(r) * patch.columns + c];
                const math::Vec3<core::f32> &old = scratch[static_cast<core::usize>(r) * patch.columns + c];
                p.x = old.x + ((n.x + s.x + w.x + e.x) * 0.25f - old.x) * strength;
                p.y = old.y + ((n.y + s.y + w.y + e.y) * 0.25f - old.y) * strength;
                p.z = old.z + ((n.z + s.z + w.z + e.z) * 0.25f - old.z) * strength;
            }
        }
        // The border is left where the walk put it: letting it move shrinks the patch a little
        // every pass, and a surface that quietly retreats from its own edge is worse than a rough
        // one.
    }
}

} // namespace lpl::voxel
