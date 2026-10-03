/**
 * @file Sheet.cpp
 * @brief Structure tensor, parabolic recentring, and the refusal that keeps a walk honest.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/std/cmath.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/voxel/Sheet.hpp>

#include <optional>
#include <utility>

namespace lpl::voxel {

namespace {

/// Nearest sample, or -1 where nothing is resident. Negative is distinguishable from any density.
[[nodiscard]] core::f32 sampleAt(const BrickMosaic &mosaic, const math::Vec3<core::f32> &point) noexcept
{
    const std::optional<core::u8> sample =
        mosaic.sampleAt(floorToI64(point.z), floorToI64(point.y), floorToI64(point.x));
    return sample.has_value() ? static_cast<core::f32>(*sample) : -1.0f;
}

/// Field value averaged over a small window: the same question, asked of a less noisy field.
[[nodiscard]] core::f32 smoothedSample(const BrickMosaic &mosaic, const math::Vec3<core::f32> &point,
                                       core::i32 radius) noexcept
{
    if (radius <= 0)
        return sampleAt(mosaic, point);
    core::f32 total = 0.0f;
    core::u32 count = 0u;
    for (core::i32 dz = -radius; dz <= radius; ++dz)
    {
        for (core::i32 dy = -radius; dy <= radius; ++dy)
        {
            for (core::i32 dx = -radius; dx <= radius; ++dx)
            {
                const math::Vec3<core::f32> offset{static_cast<core::f32>(dx), static_cast<core::f32>(dy),
                                                   static_cast<core::f32>(dz)};
                const core::f32 v = sampleAt(mosaic, point + offset);
                if (v < 0.0f)
                    continue; // A missing neighbour contributes nothing rather than a false zero.
                total += v;
                ++count;
            }
        }
    }
    return count == 0u ? -1.0f : total / static_cast<core::f32>(count);
}

[[nodiscard]] bool normaliseInPlace(math::Vec3<core::f32> &v) noexcept
{
    if (v.lengthSquared() < 1e-12f)
        return false;
    v = v.normalize();
    return true;
}

/// The central difference at @p point, or nothing when a neighbour is not resident.
[[nodiscard]] std::optional<math::Vec3<core::f32>> centralGradient(const BrickMosaic &mosaic,
                                                                   const math::Vec3<core::f32> &point) noexcept
{
    const math::Vec3<core::f32> unitX = math::Vec3<core::f32>::unitX();
    const math::Vec3<core::f32> unitY = math::Vec3<core::f32>::unitY();
    const math::Vec3<core::f32> unitZ = math::Vec3<core::f32>::unitZ();
    const core::f32 xp = sampleAt(mosaic, point + unitX);
    const core::f32 xm = sampleAt(mosaic, point - unitX);
    const core::f32 yp = sampleAt(mosaic, point + unitY);
    const core::f32 ym = sampleAt(mosaic, point - unitY);
    const core::f32 zp = sampleAt(mosaic, point + unitZ);
    const core::f32 zm = sampleAt(mosaic, point - unitZ);
    if (xp < 0.0f || xm < 0.0f || yp < 0.0f || ym < 0.0f || zp < 0.0f || zm < 0.0f)
        return std::nullopt;
    return math::Vec3<core::f32>{xp - xm, yp - ym, zp - zm};
}

/// The outer product of the gradient with itself, summed over a window. Symmetric: six terms.
struct StructureTensor final {
    core::f32 xx{0.0f};
    core::f32 xy{0.0f};
    core::f32 xz{0.0f};
    core::f32 yy{0.0f};
    core::f32 yz{0.0f};
    core::f32 zz{0.0f};

    void accumulate(const math::Vec3<core::f32> &g) noexcept
    {
        xx += g.x * g.x;
        xy += g.x * g.y;
        xz += g.x * g.z;
        yy += g.y * g.y;
        yz += g.y * g.z;
        zz += g.z * g.z;
    }

    [[nodiscard]] core::f32 trace() const noexcept { return xx + yy + zz; }

    [[nodiscard]] math::Vec3<core::f32> times(const math::Vec3<core::f32> &v) const noexcept
    {
        return {xx * v.x + xy * v.y + xz * v.z, xy * v.x + yy * v.y + yz * v.z, xz * v.x + yz * v.y + zz * v.z};
    }

    /// The column whose diagonal entry is largest: never zero when the trace is not, and never
    /// orthogonal to the dominant eigenvector of a tensor that has one clear direction.
    [[nodiscard]] math::Vec3<core::f32> heaviestColumn() const noexcept
    {
        if (xx >= yy && xx >= zz)
            return {xx, xy, xz};
        if (yy >= zz)
            return {xy, yy, yz};
        return {xz, yz, zz};
    }
};

} // namespace

bool sheetNormal(const BrickMosaic &mosaic, const math::Vec3<core::f32> &at, core::i32 radius,
                 math::Vec3<core::f32> &outNormal) noexcept
{
    if (radius < 1)
        radius = 1;

    // The structure tensor's dominant eigenvector is the direction the field changes fastest in,
    // which across a sheet is its normal -- and unlike a single finite difference it does not wander
    // with noise inside the matter, which is exactly where a walk needs it.
    StructureTensor tensor{};
    core::u32 samples = 0u;
    for (core::i32 dz = -radius; dz <= radius; ++dz)
    {
        for (core::i32 dy = -radius; dy <= radius; ++dy)
        {
            for (core::i32 dx = -radius; dx <= radius; ++dx)
            {
                const math::Vec3<core::f32> offset{static_cast<core::f32>(dx), static_cast<core::f32>(dy),
                                                   static_cast<core::f32>(dz)};
                const std::optional<math::Vec3<core::f32>> gradient = centralGradient(mosaic, at + offset);
                if (!gradient.has_value())
                    continue; // A missing neighbour contributes nothing rather than a false cliff.
                tensor.accumulate(*gradient);
                ++samples;
            }
        }
    }
    if (samples == 0u || tensor.trace() < 1e-6f)
        return false;

    // Power iteration for the dominant eigenvector. A closed form would need a cube root; this
    // needs multiplication and one square root a pass, and the answer is a direction so its sign
    // is arbitrary.
    // It starts from a column of the tensor rather than a fixed direction: a fixed start that
    // happens to be orthogonal to the normal multiplies into nothing.
    math::Vec3<core::f32> direction = tensor.heaviestColumn();
    if (!normaliseInPlace(direction))
        return false;
    for (int i = 0; i < 24; ++i)
    {
        direction = tensor.times(direction);
        if (!normaliseInPlace(direction))
            return false;
    }
    outNormal = direction;
    return true;
}

namespace {

/// How far a recentring moved a point, and whether the crest was within the window searched.
struct Recentring final {
    core::f32 distance{-1.0f};     ///< Along the normal. Negative when nothing in the window is resident.
    bool crestBeyondWindow{false}; ///< The field still rises at the edge it moved to: the crest is farther.
};

/**
 * Slides @p point along @p normal onto the local ridge, sub-sample by parabolic fit.
 *
 * Only whole offsets up to @p searchRadius are looked at. A maximum at the last of them, still
 * rising, is reported as beyond the window: the point moves to that edge, and the crest is farther.
 */
[[nodiscard]] Recentring recentre(const BrickMosaic &mosaic, math::Vec3<core::f32> &point,
                                  const math::Vec3<core::f32> &normal, core::f32 searchRadius,
                                  core::i32 ridgeSmoothing) noexcept
{
    const auto sampleAlong = [&](core::f32 offset) noexcept {
        return smoothedSample(mosaic, point + normal * offset, ridgeSmoothing);
    };

    const core::i32 span = searchRadius > 0.0f ? static_cast<core::i32>(searchRadius) : 0;
    core::i32 best = 0;
    core::f32 bestValue = -1.0f;
    for (core::i32 i = -span; i <= span; ++i)
    {
        const core::f32 v = sampleAlong(static_cast<core::f32>(i));
        if (v > bestValue)
        {
            bestValue = v;
            best = i;
        }
    }
    if (bestValue < 0.0f)
        return Recentring{};

    Recentring found{};
    core::f32 offset = static_cast<core::f32>(best);
    if (best == span || best == -span)
    {
        const core::f32 inward = static_cast<core::f32>(best > 0 ? best - 1 : best + 1);
        found.crestBeyondWindow = span > 0 && sampleAlong(inward) < bestValue;
    }
    else
    {
        // Sub-sample by fitting a parabola through the peak and its two neighbours. Without it the
        // walk quantises to whole samples and drifts off the sheet a fraction at a time.
        const core::f32 a = sampleAlong(offset - 1.0f);
        const core::f32 c = sampleAlong(offset + 1.0f);
        const core::f32 denominator = a - 2.0f * bestValue + c;
        if (a >= 0.0f && c >= 0.0f && (denominator < -1e-6f || denominator > 1e-6f))
        {
            const core::f32 shift = 0.5f * (a - c) / denominator;
            if (shift > -1.0f && shift < 1.0f)
                offset += shift;
        }
    }

    point += normal * offset;
    found.distance = offset < 0.0f ? -offset : offset;
    return found;
}

/**
 * ⚠ The safety property. A correction of a fraction of a sample is a follow; a correction the
 * size of an interline is a change of sheet, and the path it produces is smooth, plausible and
 * about the wrong surface. A crest beyond the search window is the same change of sheet, seen
 * from closer: with the limit equal to the window, it is the only one that can be seen.
 */
[[nodiscard]] bool wouldJump(const Recentring &recentring, const SheetTraceParams &params) noexcept
{
    return recentring.crestBeyondWindow || recentring.distance > params.maximumRecentre;
}

/// Projects @p direction into the plane normal to @p normal. False when nothing of it is left.
[[nodiscard]] bool projectIntoPlane(math::Vec3<core::f32> &direction, const math::Vec3<core::f32> &normal) noexcept
{
    math::Vec3<core::f32> inPlane = direction - normal * direction.dot(normal);
    if (!normaliseInPlace(inPlane))
        return false;
    direction = inPlane;
    return true;
}

/// Where one step of a walk landed and how far recentring pulled it, or why the walk stops there.
struct Step final {
    math::Vec3<core::f32> landing{};
    core::f32 pull{0.0f};
    std::optional<SheetStop> stop{};
};

/// One step from @p from along @p direction, which is re-projected into the sheet plane first and
/// left there for the next step.
[[nodiscard]] Step stepAlongSheet(const BrickMosaic &mosaic, const math::Vec3<core::f32> &from,
                                  math::Vec3<core::f32> &direction, const SheetTraceParams &params) noexcept
{
    math::Vec3<core::f32> normal{};
    if (!sheetNormal(mosaic, from, params.tensorRadius, normal) || !projectIntoPlane(direction, normal))
        return Step{.stop = SheetStop::Degenerate};

    Step step{.landing = from + direction * params.step};
    if (sampleAt(mosaic, step.landing) < 0.0f)
        return Step{.stop = SheetStop::LeftResident};

    math::Vec3<core::f32> landingNormal{};
    if (!sheetNormal(mosaic, step.landing, params.tensorRadius, landingNormal))
        return Step{.stop = SheetStop::Degenerate};
    const Recentring recentring =
        recentre(mosaic, step.landing, landingNormal, params.searchRadius, params.ridgeSmoothing);
    if (wouldJump(recentring, params))
        return Step{.stop = SheetStop::WouldJump};
    if (sampleAt(mosaic, step.landing) < static_cast<core::f32>(params.floorSample))
        return Step{.stop = SheetStop::LeftMatter};

    // Never negative here: the landing is resident, and the search samples it.
    step.pull = recentring.distance;
    return step;
}

} // namespace

SheetTrace traceSheet(const BrickMosaic &mosaic, const math::Vec3<core::f32> &seed,
                      const math::Vec3<core::f32> &heading, const SheetTraceParams &params, math::Vec3<core::f32> *out,
                      core::u32 capacity) noexcept
{
    SheetTrace trace{};
    if (out == nullptr || capacity == 0u)
        return trace;

    math::Vec3<core::f32> direction = heading;
    math::Vec3<core::f32> normal{};
    if (!normaliseInPlace(direction) || !sheetNormal(mosaic, seed, params.tensorRadius, normal))
    {
        trace.stop = SheetStop::Degenerate;
        return trace;
    }

    // Snap onto the ridge before the first step, with a WIDER search: finding the sheet and
    // staying on it are different problems. See SheetTraceParams::seedSearchRadius.
    const core::f32 seedSearchRadius =
        params.seedSearchRadius > params.searchRadius ? params.seedSearchRadius : params.searchRadius;
    math::Vec3<core::f32> point = seed;
    (void) recentre(mosaic, point, normal, seedSearchRadius, params.ridgeSmoothing);
    out[trace.count++] = point;

    const core::u32 limit = params.maximumSteps < capacity ? params.maximumSteps : capacity;
    while (trace.count < limit)
    {
        const Step step = stepAlongSheet(mosaic, point, direction, params);
        if (step.stop.has_value())
        {
            trace.stop = *step.stop;
            if (trace.stop == SheetStop::WouldJump)
                ++trace.refusedJumps;
            return trace;
        }
        if (step.pull > 1e-4f)
            ++trace.recentrings;
        trace.totalRecentre += step.pull;
        point = step.landing;
        out[trace.count++] = point;
    }
    return trace;
}

namespace {

/// Two unit vectors spanning the plane normal to @p normal. Which two is arbitrary: a sheet has no
/// preferred direction.
[[nodiscard]] bool inPlaneBasis(const math::Vec3<core::f32> &normal, math::Vec3<core::f32> &first,
                                math::Vec3<core::f32> &second) noexcept
{
    // Crossed with X, unless the normal is close to X: the cross product of two aligned vectors
    // vanishes.
    const math::Vec3<core::f32> helper =
        (normal.x > 0.9f || normal.x < -0.9f) ? math::Vec3<core::f32>::unitY() : math::Vec3<core::f32>::unitX();
    first = normal.cross(helper);
    if (!normaliseInPlace(first))
        return false;
    second = normal.cross(first);
    return normaliseInPlace(second);
}

/// What walking a line both ways from its middle produced.
struct LineWalk final {
    SheetTrace backward{};
    SheetTrace forward{};
    bool reachedBothEnds{false};
};

/**
 * A backward walk is written from line[0] outward, and the line reads toward its middle: reverses
 * the @p count points written so the first lands on line[@p middle], and repeats the farthest one
 * into the slots the walk did not reach, or @p seed when it reached none.
 *
 * @pre @p count <= @p middle + 1.
 */
void turnBackwardWalkAround(math::Vec3<core::f32> *line, core::u32 middle, core::u32 count,
                            const math::Vec3<core::f32> &seed) noexcept
{
    for (core::u32 low = 0u, high = count; low + 1u < high; ++low, --high)
        std::swap(line[low], line[high - 1u]);
    const core::u32 unreached = middle + 1u - count;
    for (core::u32 i = count; i > 0u; --i)
        line[i - 1u + unreached] = line[i - 1u];
    const math::Vec3<core::f32> farthest = count > 0u ? line[unreached] : seed;
    for (core::u32 i = 0u; i < unreached; ++i)
        line[i] = farthest;
}

/// Repeats the last of the @p count points of @p line into the rest of its @p length, or @p seed
/// when there is none.
void repeatLastPoint(math::Vec3<core::f32> *line, core::u32 length, core::u32 count,
                     const math::Vec3<core::f32> &seed) noexcept
{
    const math::Vec3<core::f32> last = count > 0u ? line[count - 1u] : seed;
    for (core::u32 i = count; i < length; ++i)
        line[i] = last;
}

/**
 * Fills line[0, @p length) with one surface walked from @p seed both ways along @p axis: the snapped
 * seed at line[@p length / 2], the backward walk before it and the forward walk after it. Both walks
 * start with the same snapped seed, so the backward one is asked for one point more and its first
 * is the one the forward walk writes again.
 *
 * Stepping straight to the far ends would leave the sheet wherever it curves: each end is walked.
 */
[[nodiscard]] LineWalk walkBothWays(const BrickMosaic &mosaic, const math::Vec3<core::f32> &seed,
                                    const math::Vec3<core::f32> &axis, const SheetTraceParams &params,
                                    math::Vec3<core::f32> *line, core::u32 length) noexcept
{
    const core::u32 middle = length / 2u;
    LineWalk walk{};
    walk.backward = traceSheet(mosaic, seed, -axis, params, line, middle + 1u);
    turnBackwardWalkAround(line, middle, walk.backward.count, seed);
    walk.forward = traceSheet(mosaic, seed, axis, params, line + middle, length - middle);
    repeatLastPoint(line + middle, length - middle, walk.forward.count, seed);
    walk.reachedBothEnds = walk.backward.count == middle + 1u && walk.forward.count == length - middle;
    return walk;
}

void countWalk(SheetPatch &patch, const SheetTrace &trace) noexcept
{
    patch.refusedJumps += trace.refusedJumps;
    ++patch.stops[static_cast<core::u32>(trace.stop)];
}

void countWalks(SheetPatch &patch, const LineWalk &walk) noexcept
{
    countWalk(patch, walk.backward);
    countWalk(patch, walk.forward);
}

} // namespace

SheetPatch traceSheetPatch(const BrickMosaic &mosaic, const math::Vec3<core::f32> &seed, const SheetTraceParams &params,
                           core::u32 rows, core::u32 columns, math::Vec3<core::f32> *out) noexcept
{
    SheetPatch patch{};
    if (out == nullptr || rows == 0u || columns == 0u)
        return patch;

    // ⚠ The dimensions are published only once something has been written into them. An earlier
    // version (PR #207, 2026-10-03) set them first and returned early when the seed had no normal, which handed back a
    // patch that reported its full size with every point left at the origin -- a surface that
    // looks valid, sits at the corner of the volume, and is entirely fictional.
    math::Vec3<core::f32> normal{};
    math::Vec3<core::f32> alongRows{};
    math::Vec3<core::f32> acrossRows{};
    if (!sheetNormal(mosaic, seed, params.tensorRadius, normal) || !inPlaneBasis(normal, alongRows, acrossRows))
        return patch;

    // ⚠⚠ **Every seed is reached by WALKING, never by stepping in a straight line.** A first
    // version (PR #207, 2026-10-03) placed each row's start at a straight offset from the seed, and on a sheet with any
    // curvature at all that lands in the medium a few samples away -- so the walk found no normal,
    // returned nothing, and each row was padded with its own start. The patch came out as a grid
    // of duplicated points: a surface with zero area that reported a full size and, on the fixture
    // it was tested against, sat exactly on the sheet by coincidence of where the seed was. The
    // test passed. Growing the patch by tracing is not a refinement; it is the only way the seeds
    // stay on the surface.
    //
    // The row starts are walked into out[0, rows), row r's start at out[r]. Rows are then traced
    // from the last to the first: row r fills out[r * columns, (r + 1) * columns), which begins at
    // or after out[r], so it never covers a start still to be read -- those are below out[r].
    const LineWalk rowStarts = walkBothWays(mosaic, seed, acrossRows, params, out, rows);
    countWalks(patch, rowStarts);
    for (core::u32 row = rows; row > 0u; --row)
    {
        math::Vec3<core::f32> *line = &out[static_cast<core::usize>(row - 1u) * columns];
        const math::Vec3<core::f32> rowStart = out[row - 1u];
        const LineWalk walk = walkBothWays(mosaic, rowStart, alongRows, params, line, columns);
        countWalks(patch, walk);
        if (!walk.reachedBothEnds)
            ++patch.shortRows;
    }

    patch.rows = rows;
    patch.columns = columns;
    patch.points = out;
    return patch;
}

namespace {

/// The average of the four neighbours of an interior point of a row-major grid.
[[nodiscard]] math::Vec3<core::f32> neighbourAverage(const math::Vec3<core::f32> *points, core::u32 columns,
                                                     core::u32 row, core::u32 column) noexcept
{
    const auto at = [&](core::u32 r, core::u32 c) noexcept -> const math::Vec3<core::f32> & {
        return points[static_cast<core::usize>(r) * columns + c];
    };
    return (at(row - 1u, column) + at(row + 1u, column) + at(row, column - 1u) + at(row, column + 1u)) * 0.25f;
}

/**
 * One pass of relaxation over the interior of @p patch, with @p before as room for a copy of it.
 *
 * Jacobi, not Gauss-Seidel: reading points already moved this pass would make the result depend
 * on the order the grid is walked, and the same patch would relax differently depending on which
 * corner the loop started from. The border is left where the walk put it.
 */
void relaxOnce(SheetPatch &patch, core::f32 strength, lpl::pmr::vector<math::Vec3<core::f32>> &before) noexcept
{
    for (core::usize i = 0u; i < before.size(); ++i)
        before[i] = patch.points[i];
    for (core::u32 r = 1u; r + 1u < patch.rows; ++r)
    {
        for (core::u32 c = 1u; c + 1u < patch.columns; ++c)
        {
            const math::Vec3<core::f32> &old = before[static_cast<core::usize>(r) * patch.columns + c];
            patch.at(r, c) = old + (neighbourAverage(before.data(), patch.columns, r, c) - old) * strength;
        }
    }
}

[[nodiscard]] bool hasInterior(const SheetPatch &patch) noexcept
{
    return patch.points != nullptr && patch.rows >= 3u && patch.columns >= 3u;
}

[[nodiscard]] core::usize pointCount(const SheetPatch &patch) noexcept
{
    return static_cast<core::usize>(patch.rows) * patch.columns;
}

} // namespace

void relaxPatchOnRidge(const BrickMosaic &mosaic, SheetPatch &patch, const SheetTraceParams &params, core::f32 strength,
                       core::u32 iterations) noexcept
{
    if (!hasInterior(patch))
        return;
    const core::f32 clamped = strength < 0.0f ? 0.0f : (strength > 1.0f ? 1.0f : strength);

    // Back onto the crest after every pass. The search is TIGHT -- a wide one here would let a
    // point that the smoothing nudged toward the neighbouring sheet settle on it, which is the
    // jump this whole file exists to refuse, arriving by the back door.
    constexpr core::f32 kTightSearchRadius = 1.0f;
    lpl::pmr::vector<math::Vec3<core::f32>> before(pointCount(patch));
    for (core::u32 pass = 0u; pass < iterations; ++pass)
    {
        relaxOnce(patch, clamped, before);
        for (core::u32 r = 1u; r + 1u < patch.rows; ++r)
        {
            for (core::u32 c = 1u; c + 1u < patch.columns; ++c)
            {
                math::Vec3<core::f32> &point = patch.at(r, c);
                math::Vec3<core::f32> normal{};
                if (!sheetNormal(mosaic, point, params.tensorRadius, normal))
                    continue;
                math::Vec3<core::f32> moved = point;
                const Recentring recentring =
                    recentre(mosaic, moved, normal, kTightSearchRadius, params.ridgeSmoothing);
                if (recentring.distance >= 0.0f && recentring.distance <= params.maximumRecentre)
                    point = moved;
            }
        }
    }
}

core::f32 patchRoughness(const SheetPatch &patch) noexcept
{
    if (!hasInterior(patch))
        return -1.0f;
    core::f64 total = 0.0;
    core::u32 count = 0u;
    for (core::u32 r = 1u; r + 1u < patch.rows; ++r)
    {
        for (core::u32 c = 1u; c + 1u < patch.columns; ++c)
        {
            const math::Vec3<core::f32> deviation =
                neighbourAverage(patch.points, patch.columns, r, c) - patch.at(r, c);
            total += static_cast<core::f64>(deviation.lengthSquared());
            ++count;
        }
    }
    return static_cast<core::f32>(pmr::sqrt(total / static_cast<core::f64>(count)));
}

void relaxPatch(SheetPatch &patch, core::f32 strength, core::u32 iterations) noexcept
{
    if (!hasInterior(patch) || strength <= 0.0f)
        return;
    const core::f32 clamped = strength > 1.0f ? 1.0f : strength;
    lpl::pmr::vector<math::Vec3<core::f32>> before(pointCount(patch));
    for (core::u32 pass = 0u; pass < iterations; ++pass)
        relaxOnce(patch, clamped, before);
}

} // namespace lpl::voxel
