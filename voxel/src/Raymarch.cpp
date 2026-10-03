/**
 * @file Raymarch.cpp
 * @brief Front-to-back volume integration over a resident brick mosaic.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/math/Cordic.hpp>
#include <lpl/std/cmath.hpp>
#include <lpl/voxel/Raymarch.hpp>

#include <optional>

namespace lpl::voxel {

namespace {

/**
 * Axis mapping, stated once because getting it wrong is invisible.
 *
 * A zarr volume is indexed (z, y, x) and the walked world is (x, y, z). World X is the volume's
 * fastest axis, world Z its slowest. A viewer that swapped them would render a perfectly plausible
 * object that is the subject transposed -- and nothing about the picture would say so.
 */
constexpr core::u32 kAxisOfWorldX = 2u;
constexpr core::u32 kAxisOfWorldY = 1u;
constexpr core::u32 kAxisOfWorldZ = 0u;

struct Slab final {
    core::f32 enter{0.0f};
    core::f32 exit{0.0f};
    bool hit{false};
};

/// Ray against the axis-aligned box [0, size] in metres.
[[nodiscard]] Slab intersectBox(const math::Vec3<core::f32> &origin, const math::Vec3<core::f32> &direction,
                                const core::f32 size[3]) noexcept
{
    const core::f32 o[3]{origin.x, origin.y, origin.z};
    const core::f32 d[3]{direction.x, direction.y, direction.z};

    core::f32 tMin = 0.0f;
    core::f32 tMax = 3.0e38f;
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        if (d[a] > -1e-9f && d[a] < 1e-9f)
        {
            // Parallel to this pair of faces: either always inside them or never.
            if (o[a] < 0.0f || o[a] > size[a])
                return Slab{};
            continue;
        }
        const core::f32 inv = 1.0f / d[a];
        core::f32 t0 = (0.0f - o[a]) * inv;
        core::f32 t1 = (size[a] - o[a]) * inv;
        if (t0 > t1)
        {
            const core::f32 s = t0;
            t0 = t1;
            t1 = s;
        }
        if (t0 > tMin)
            tMin = t0;
        if (t1 < tMax)
            tMax = t1;
        if (tMin > tMax)
            return Slab{};
    }
    return Slab{tMin, tMax, true};
}

/// World (x, y, z) into volume (z, y, x) order, scaled. @see kAxisOfWorldX
void toVolumeAxes(const math::Vec3<core::f32> &world, core::f32 scale, core::f32 volume[3]) noexcept
{
    volume[kAxisOfWorldX] = world.x * scale;
    volume[kAxisOfWorldY] = world.y * scale;
    volume[kAxisOfWorldZ] = world.z * scale;
}

/// Rounds a density to the nearest byte, saturating at both ends.
[[nodiscard]] constexpr core::u8 toByte(core::f32 v) noexcept
{
    return static_cast<core::u8>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v + 0.5f));
}

[[nodiscard]] constexpr core::u32 packColour(core::f32 r, core::f32 g, core::f32 b) noexcept
{
    const auto q = [](core::f32 v) noexcept -> core::u32 {
        const core::f32 c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return static_cast<core::u32>(c * 255.0f + 0.5f);
    };
    return 0xFF000000u | (q(r) << 16) | (q(g) << 8) | q(b);
}

/// Trilinear blend of a cell's eight corners, indexed [z][y][x], at fractions inside the cell.
[[nodiscard]] constexpr core::f32 interpolateCorners(const core::f32 corner[2][2][2], core::f32 fz, core::f32 fy,
                                                     core::f32 fx) noexcept
{
    const core::f32 c00 = corner[0][0][0] * (1.0f - fx) + corner[0][0][1] * fx;
    const core::f32 c01 = corner[0][1][0] * (1.0f - fx) + corner[0][1][1] * fx;
    const core::f32 c10 = corner[1][0][0] * (1.0f - fx) + corner[1][0][1] * fx;
    const core::f32 c11 = corner[1][1][0] * (1.0f - fx) + corner[1][1][1] * fx;
    const core::f32 c0 = c00 * (1.0f - fy) + c01 * fy;
    const core::f32 c1 = c10 * (1.0f - fy) + c11 * fy;
    return c0 * (1.0f - fz) + c1 * fz;
}

struct Rgb final {
    core::f32 red{0.0f};
    core::f32 green{0.0f};
    core::f32 blue{0.0f};
};

/// Colour and opacity gathered along one ray, front to back. Colour is premultiplied by alpha.
struct Accumulated final {
    core::f32 red{0.0f};
    core::f32 green{0.0f};
    core::f32 blue{0.0f};
    core::f32 alpha{0.0f};
};

/// Moves @p colour toward @p tint by @p mix, in [0,1].
constexpr void tintTowards(Rgb &colour, const Rgb &tint, core::f32 mix) noexcept
{
    colour.red += (tint.red - colour.red) * mix;
    colour.green += (tint.green - colour.green) * mix;
    colour.blue += (tint.blue - colour.blue) * mix;
}

/**
 * How far a prediction may tint: its value above @p floor as a fraction of @p range, capped at
 * one, times what the map is worth. @see MarchParams::overlayConfidence
 */
[[nodiscard]] constexpr core::f32 tintMix(core::u8 value, core::u8 floor, core::f32 range,
                                          core::f32 confidence) noexcept
{
    core::f32 strength = static_cast<core::f32>(value - floor) / range;
    if (strength > 1.0f)
        strength = 1.0f;
    return strength * confidence;
}

/// One sample of the field at a level-0 position, and which brick, at which level, answered.
struct Sample final {
    core::u8 density{0};
    core::u32 level{0};
    bool inside{false};
    const BrickView *brick{nullptr};
};

/// Density of one brick at a level-0 position, without any level blending.
[[nodiscard]] core::u8 sampleBrick(const BrickView &brick, const core::f32 position[3], bool trilinear) noexcept
{
    const core::f32 scale = 1.0f / static_cast<core::f32>(static_cast<core::i64>(1) << brick.key.level);
    const core::i64 lastIndex = static_cast<core::i64>(kBrickEdge) - 1;
    core::f32 local[3];
    core::i64 cell[3];
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        local[a] = (position[a] - static_cast<core::f32>(brickOriginInBaseSamples(brick.key, a))) * scale;
        const core::i64 index = floorToI64(local[a]);
        cell[a] = index < 0 ? 0 : (index > lastIndex ? lastIndex : index);
    }

    if (!trilinear)
        return brick.at(static_cast<core::u32>(cell[0]), static_cast<core::u32>(cell[1]),
                        static_cast<core::u32>(cell[2]));
    if (cell[0] >= lastIndex || cell[1] >= lastIndex || cell[2] >= lastIndex)
        return 0u; // Caller resolves the corners through the mosaic; see crossBrickSample.

    core::f32 corner[2][2][2];
    for (core::u32 dz = 0u; dz < 2u; ++dz)
    {
        for (core::u32 dy = 0u; dy < 2u; ++dy)
        {
            for (core::u32 dx = 0u; dx < 2u; ++dx)
                corner[dz][dy][dx] = static_cast<core::f32>(brick.at(static_cast<core::u32>(cell[0] + dz),
                                                                     static_cast<core::u32>(cell[1] + dy),
                                                                     static_cast<core::u32>(cell[2] + dx)));
        }
    }
    return toByte(interpolateCorners(corner, local[0] - static_cast<core::f32>(cell[0]),
                                     local[1] - static_cast<core::f32>(cell[1]),
                                     local[2] - static_cast<core::f32>(cell[2])));
}

/**
 * Trilinear sample that resolves each corner through the mosaic when the cell straddles a brick.
 *
 * @warning **The cheap version -- clamp to the brick's edge -- leaves a one-sample seam on every
 * brick face, and that is not the harmless thing it sounds like.** Put the eye exactly on such a
 * plane, which happens whenever a coordinate is a multiple of the brick edge, and every ray along
 * it runs down the seam: a hard horizontal band across the whole picture. It was the second
 * defect in the first real renders and it survived four other fixes. The extra lookups are paid
 * only on the last cell of each axis -- about one sample in forty at level 0.
 */
[[nodiscard]] core::u8 crossBrickSample(const BrickMosaic &mosaic, const BrickView &brick, const core::f32 position[3],
                                        bool trilinear) noexcept
{
    if (!trilinear)
        return sampleBrick(brick, position, false);

    // Snap to this brick's sample lattice so the eight corners are the same eight everywhere in
    // the cell -- otherwise neighbouring rays interpolate over different corners and the surface
    // shimmers.
    const core::i64 step = static_cast<core::i64>(1) << brick.key.level;
    const core::f32 stepF = static_cast<core::f32>(step);
    const core::i64 span = brickSpanInBaseSamples(brick.key.level);
    core::i64 lattice[3];
    core::f32 fraction[3];
    bool cellInsideBrick = true;
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        const core::f32 inLattice = position[a] / stepF;
        const core::i64 whole = floorToI64(inLattice);
        lattice[a] = whole * step;
        fraction[a] = inLattice - static_cast<core::f32>(whole);
        if (lattice[a] + step >= brickOriginInBaseSamples(brick.key, a) + span)
            cellInsideBrick = false;
    }
    if (cellInsideBrick)
        return sampleBrick(brick, position, true);

    core::f32 corner[2][2][2];
    for (core::u32 dz = 0u; dz < 2u; ++dz)
    {
        for (core::u32 dy = 0u; dy < 2u; ++dy)
        {
            for (core::u32 dx = 0u; dx < 2u; ++dx)
            {
                const core::f32 at[3]{static_cast<core::f32>(lattice[0] + static_cast<core::i64>(dz) * step),
                                      static_cast<core::f32>(lattice[1] + static_cast<core::i64>(dy) * step),
                                      static_cast<core::f32>(lattice[2] + static_cast<core::i64>(dx) * step)};
                const BrickView *owner = mosaic.find(floorToI64(at[0]), floorToI64(at[1]), floorToI64(at[2]));
                // No neighbour: the brick's own edge sample stands in. Reporting zero would make
                // the boundary look like a wall of vacuum.
                corner[dz][dy][dx] = static_cast<core::f32>(sampleBrick(owner != nullptr ? *owner : brick, at, false));
            }
        }
    }
    return toByte(interpolateCorners(corner, fraction[0], fraction[1], fraction[2]));
}

/**
 * The brick that answers at a level-0 point, reusing @p hint when it provably would.
 *
 * The lookup is the expensive half of a sample, and the gradient asks for six of them a step --
 * almost always inside the brick the centre is already in. The hint skips the index for those and
 * nothing else: an earlier version shortcut the whole sample instead, skipped the level blend, and
 * moved the picture. It is only trusted at the finest resident level, where nothing can shadow it;
 * two samples away from a centre in a coarse brick, a finer brick may well take over.
 */
[[nodiscard]] const BrickView *answeringBrick(const BrickMosaic &mosaic, const core::i64 base[3],
                                              const BrickView *hint) noexcept
{
    if (hint != nullptr && hint->key.level == mosaic.finestLevel() &&
        brickIndexOfBase(base[0], hint->key.level) == hint->key.z &&
        brickIndexOfBase(base[1], hint->key.level) == hint->key.y &&
        brickIndexOfBase(base[2], hint->key.level) == hint->key.x)
        return hint;
    return mosaic.find(base[0], base[1], base[2]);
}

/// The nearest face of a brick to a point inside it: how deep, on which axis, and on which side.
struct NearestFace final {
    core::f32 depth{3.0e38f};
    core::u32 axis{0u};
    bool low{true};
};

[[nodiscard]] NearestFace nearestFace(const BrickView &brick, const core::f32 position[3]) noexcept
{
    const core::f32 span = static_cast<core::f32>(brickSpanInBaseSamples(brick.key.level));
    NearestFace nearest{};
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        const core::f32 fromLow = position[a] - static_cast<core::f32>(brickOriginInBaseSamples(brick.key, a));
        const core::f32 fromHigh = span - fromLow;
        if (fromLow < nearest.depth)
            nearest = NearestFace{fromLow, a, true};
        if (fromHigh < nearest.depth)
            nearest = NearestFace{fromHigh, a, false};
    }
    return nearest;
}

/**
 * @p density faded into the coarser brick behind @p brick, within @p blendBand of a face where the
 * detail stops. @see MarchParams::levelBlendSamples
 */
[[nodiscard]] core::u8 fadeIntoCoarser(const BrickMosaic &mosaic, const BrickView &brick, core::u8 density,
                                       const core::f32 position[3], const core::i64 base[3], core::f32 blendBand,
                                       bool trilinear) noexcept
{
    const NearestFace face = nearestFace(brick, position);
    if (face.depth >= blendBand)
        return density;

    core::f32 beyond[3]{position[0], position[1], position[2]};
    beyond[face.axis] += face.low ? -(blendBand + 1.0f) : (blendBand + 1.0f);
    const BrickView *neighbour = mosaic.find(floorToI64(beyond[0]), floorToI64(beyond[1]), floorToI64(beyond[2]));
    if (neighbour != nullptr && neighbour->key.level == brick.key.level)
        return density;

    const BrickView *coarse = mosaic.findCoarserThan(brick.key.level, base[0], base[1], base[2]);
    if (coarse == nullptr)
        return density;

    const core::f32 inward = face.depth / blendBand; // 0 at the face, 1 at the inner edge of the band.
    const core::f32 fine = static_cast<core::f32>(density);
    const core::f32 far = static_cast<core::f32>(crossBrickSample(mosaic, *coarse, position, trilinear));
    return toByte(far + (fine - far) * inward);
}

[[nodiscard]] Sample sampleField(const BrickMosaic &mosaic, const core::f32 position[3], bool trilinear,
                                 core::f32 blendBand, const BrickView *hint = nullptr) noexcept
{
    const core::i64 base[3]{floorToI64(position[0]), floorToI64(position[1]), floorToI64(position[2])};
    const BrickView *brick = answeringBrick(mosaic, base, hint);
    if (brick == nullptr)
        return Sample{};

    Sample out{};
    out.brick = brick;
    out.level = brick->key.level;
    out.inside = true;
    out.density = crossBrickSample(mosaic, *brick, position, trilinear);
    if (blendBand > 0.0f)
        out.density = fadeIntoCoarser(mosaic, *brick, out.density, position, base, blendBand, trilinear);
    return out;
}

/**
 * Distance along the ray, in level-0 samples, from @p position to where it leaves the box, which
 * must contain it: 0 when it sits on the low face it is leaving through.
 */
[[nodiscard]] core::f32 samplesToBoxExit(const core::f32 lowBound[3], core::f32 span, const core::f32 position[3],
                                         const core::f32 direction[3]) noexcept
{
    core::f32 best = 3.0e38f;
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        if (direction[a] > -1e-9f && direction[a] < 1e-9f)
            continue;
        const core::f32 bound = direction[a] > 0.0f ? lowBound[a] + span : lowBound[a];
        // Zero is an exit, not a miss: refusing it sent a ray sitting on a low face, moving down,
        // a whole box further, over whatever the box below held.
        const core::f32 t = (bound - position[a]) / direction[a];
        if (t >= 0.0f && t < best)
            best = t;
    }
    return best >= 3.0e38f ? span : best;
}

/**
 * Distance to where the ray leaves the finest-level brick box around @p position: the largest hop
 * that cannot leap over a resident brick. @see BrickMosaic::finestLevel
 *
 * @warning **A hop always ends at a box exit, never a span further on.** A ray usually enters a
 * box partway through, so advancing by a whole span lands inside the NEXT box and skips whatever
 * was visible in between; the amount skipped depends on where the ray crossed, so the picture
 * comes out in rectangular patches. It is one of the defects @ref DebugView lists.
 */
[[nodiscard]] core::f32 samplesToFinestBoxExit(const BrickMosaic &mosaic, const core::f32 position[3],
                                               const core::f32 direction[3]) noexcept
{
    const core::u32 level = mosaic.finestLevel();
    const core::i64 span = brickSpanInBaseSamples(level);
    core::f32 lowBound[3];
    for (core::u32 a = 0u; a < 3u; ++a)
        lowBound[a] =
            static_cast<core::f32>(static_cast<core::i64>(brickIndexOfBase(floorToI64(position[a]), level)) * span);
    return samplesToBoxExit(lowBound, static_cast<core::f32>(span), position, direction);
}

/**
 * Distance to where the ray leaves the occupancy cell around @p position, when no sample in that
 * cell reaches the curve; nothing when one does. @see BrickView::occupancy
 */
[[nodiscard]] std::optional<core::f32> samplesToEmptyCellExit(const BrickView &brick, const TransferFunction &curve,
                                                              const core::f32 position[3],
                                                              const core::f32 direction[3]) noexcept
{
    const core::u32 level = brick.key.level;
    core::i64 origin[3];
    core::u32 local[3];
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        origin[a] = brickOriginInBaseSamples(brick.key, a);
        local[a] = static_cast<core::u32>((floorToI64(position[a]) - origin[a]) >> level);
        if (local[a] >= kBrickEdge)
            return std::nullopt;
    }
    if (curve.reachesVisible(brick.cellHighest(local[0], local[1], local[2])))
        return std::nullopt;

    const core::u32 cellShift = kOccupancyShift + level;
    core::f32 lowBound[3];
    for (core::u32 a = 0u; a < 3u; ++a)
        lowBound[a] =
            static_cast<core::f32>(origin[a] + (static_cast<core::i64>(local[a] >> kOccupancyShift) << cellShift));
    return samplesToBoxExit(lowBound, static_cast<core::f32>(static_cast<core::i64>(1) << cellShift), position,
                            direction);
}

/// Opacity of one step of @p samplesPerStep level-0 samples, from the curve's opacity per sample.
[[nodiscard]] core::f32 opacityOverStep(core::f32 perSample, core::f32 samplesPerStep) noexcept
{
    const core::f32 a = 1.0f - transparencyAfter(perSample, samplesPerStep);
    return a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
}

/// Whole steps that cover @p distance, and at least one, so a hop always moves the ray forward.
[[nodiscard]] core::i64 stepsCovering(core::f32 distance, core::f32 stepLength) noexcept
{
    const core::f32 steps = distance / stepLength;
    if (!(steps > 1.0f))
        return 1;
    core::i64 whole = static_cast<core::i64>(steps);
    if (static_cast<core::f32>(whole) < steps)
        ++whole;
    return whole;
}

/// What every ray of one call shares, derived once.
struct Frame final {
    const BrickMosaic &mosaic;
    const MarchParams &params;
    TransferFunction curves[kMaxPyramidLevels]; ///< Derived once per call: per sample, it would dominate the frame.
    core::f32 metresPerSample{0.0f};
    core::f32 samplesPerMetre{0.0f};
    core::f32 stepMetres{0.0f};
};

/// The intermediate quantity a debug view paints, taken at the first sample that paints.
struct DebugCapture final {
    bool captured{false};
    core::f32 value{0.0f};
    core::u32 level{0u};
    core::f32 normal[3]{0.0f, 0.0f, 0.0f}; ///< World x, y, z.
};

/// How one sample is lit, and the opacity per level-0 sample it ends up with.
struct Lighting final {
    core::f32 alphaPerSample{0.0f};
    core::f32 shade{1.0f};
    core::f32 gradientLength{0.0f};
    core::f32 normal[3]{0.0f, 0.0f, 0.0f}; ///< World x, y, z.
};

/**
 * Shades a sample from the local gradient and raises its opacity at boundaries.
 *
 * @warning **A neighbour outside the resident set is NOT density zero.** Treating a missing brick
 * as empty makes the difference across a brick face enormous, so every face of every resident
 * brick lights up as a false surface, and because the resident set is a grid the picture comes out
 * in rectangles; it is one of the defects @ref DebugView lists. Where there is no neighbour, the
 * centre stands in, which reports "flat" rather than "cliff".
 */
[[nodiscard]] Lighting lightFromGradient(const Frame &frame, const Sample &sample, core::f32 alphaPerSample,
                                         const core::f32 position[3], const math::Vec3<core::f32> &direction,
                                         MarchReport &report) noexcept
{
    const MarchParams &params = frame.params;
    Lighting lit{};
    lit.alphaPerSample = alphaPerSample;
    ++report.gradients;

    const core::f32 here = static_cast<core::f32>(sample.density);
    // Nearest probes with the centre's brick as a hint: @see MarchParams::gradientSpread.
    const auto probe = [&](core::u32 axis, core::f32 offset) noexcept {
        core::f32 at[3]{position[0], position[1], position[2]};
        at[axis] += offset;
        const Sample s = sampleField(frame.mosaic, at, false, params.levelBlendSamples, sample.brick);
        return s.inside ? static_cast<core::f32>(s.density) : here;
    };
    const core::f32 d = params.gradientSpread;
    const core::f32 gx = probe(kAxisOfWorldX, d) - probe(kAxisOfWorldX, -d);
    const core::f32 gy = probe(kAxisOfWorldY, d) - probe(kAxisOfWorldY, -d);
    const core::f32 gz = probe(kAxisOfWorldZ, d) - probe(kAxisOfWorldZ, -d);

    const core::f32 len2 = gx * gx + gy * gy + gz * gz;
    if (!(len2 > 1e-6f))
    {
        lit.alphaPerSample *= 1.0f - params.boundaryOpacity;
        return lit;
    }

    const core::f32 g = pmr::sqrt(len2);
    lit.gradientLength = g;
    const core::f32 inv = 1.0f / g;
    lit.normal[0] = gx * inv;
    lit.normal[1] = gy * inv;
    lit.normal[2] = gz * inv;
    // Two-sided: a sheet has no outside, so a normal that points away from the eye is the same
    // surface seen from behind, not an unlit one.
    core::f32 lambert = -(lit.normal[0] * direction.x + lit.normal[1] * direction.y + lit.normal[2] * direction.z);
    if (lambert < 0.0f)
        lambert = -lambert;
    lit.shade = params.ambient + (1.0f - params.ambient) * lambert;
    lit.shade = 1.0f + params.shading * (lit.shade - 1.0f);

    // Boundary opacity: homogeneous matter stays translucent, edges turn solid.
    const core::f32 strength = g * (1.0f / (96.0f * params.gradientSpread));
    const core::f32 edge = strength > 1.0f ? 1.0f : strength;
    lit.alphaPerSample *= 1.0f - params.boundaryOpacity * (1.0f - edge);
    return lit;
}

/**
 * Called only for a sample that paints: tints its colour where the overlay is strong. That caller
 * is what keeps the overlay on matter the scan already put here; a prediction floating in a void
 * would be the model drawing rather than reading. @see MarchParams::overlay
 */
void tintWithOverlay(const MarchParams &params, const core::f32 position[3], core::f32 shade, Rgb &colour,
                     MarchReport &report) noexcept
{
    if (params.overlay == nullptr || !(params.overlayConfidence > 0.0f))
        return;
    const Sample o = sampleField(*params.overlay, position, params.trilinear, 0.0f);
    if (!o.inside || o.density <= params.overlayFloor)
        return;

    const core::f32 range = static_cast<core::f32>(
        params.overlayFull > params.overlayFloor ? params.overlayFull - params.overlayFloor : 1u);
    tintTowards(colour, Rgb{params.overlayRed * shade, params.overlayGreen * shade, params.overlayBlue * shade},
                tintMix(o.density, params.overlayFloor, range, params.overlayConfidence));
    ++report.overlaid;
}

/// Tints a surface's colour with the prediction painted on it, at this pixel's texture coordinates.
void tintWithInk(const SurfaceLayer &surface, const SurfaceTexture *texture, core::usize pixelIndex, Rgb &colour,
                 MarchReport &report) noexcept
{
    if (!surface.textured() || texture == nullptr || texture->u == nullptr || texture->v == nullptr)
        return;
    const core::f32 tu = texture->u[pixelIndex];
    const core::f32 tv = texture->v[pixelIndex];
    if (!(tu >= 0.0f && tu <= 1.0f && tv >= 0.0f && tv <= 1.0f))
        return;

    core::u32 ix = static_cast<core::u32>(tu * static_cast<core::f32>(surface.inkWidth - 1u));
    // v runs up in a texture and down in an image; flipping here is the one place that convention
    // lives, rather than in every caller.
    core::u32 iy = static_cast<core::u32>((1.0f - tv) * static_cast<core::f32>(surface.inkHeight - 1u));
    if (ix >= surface.inkWidth)
        ix = surface.inkWidth - 1u;
    if (iy >= surface.inkHeight)
        iy = surface.inkHeight - 1u;
    const core::u8 value = surface.ink[static_cast<core::usize>(iy) * surface.inkWidth + ix];
    if (value <= surface.inkFloor)
        return;

    tintTowards(
        colour, Rgb{surface.inkRed, surface.inkGreen, surface.inkBlue},
        tintMix(value, surface.inkFloor, static_cast<core::f32>(255u - surface.inkFloor), surface.inkConfidence));
    ++report.inkPainted;
}

/**
 * Composites every traced surface the ray has reached by @p t, once each. Front-to-back, so
 * whatever the scan put in front of a surface still occludes it -- unless that layer asked to be
 * drawn through matter. @see SurfaceLayer::throughMatter
 */
void compositeSurfaces(const Frame &frame, core::f32 t, core::usize pixelIndex, bool layerPainted[kMaxSurfaceLayers],
                       Accumulated &acc, MarchReport &report) noexcept
{
    const MarchParams &params = frame.params;
    for (core::u32 layer = 0u; layer < params.surfaceCount && layer < kMaxSurfaceLayers; ++layer)
    {
        const SurfaceLayer &surface = params.surfaces[layer];
        if (layerPainted[layer] || !surface.valid() || t < surface.depth[pixelIndex])
            continue;
        layerPainted[layer] = true;
        ++report.surfaceHits;

        const core::f32 facing = surface.facing != nullptr ? surface.facing[pixelIndex] : 1.0f;
        const core::f32 lit = 0.35f + 0.65f * facing;
        Rgb colour{surface.red, surface.green, surface.blue};
        tintWithInk(surface, params.surfaceTexture != nullptr ? &params.surfaceTexture[layer] : nullptr, pixelIndex,
                    colour, report);

        if (surface.throughMatter)
        {
            const core::f32 keep = 1.0f - surface.opacity;
            acc.red = acc.red * keep + surface.opacity * colour.red * lit;
            acc.green = acc.green * keep + surface.opacity * colour.green * lit;
            acc.blue = acc.blue * keep + surface.opacity * colour.blue * lit;
            acc.alpha = acc.alpha * keep + surface.opacity;
            continue;
        }
        const core::f32 w = (1.0f - acc.alpha) * surface.opacity;
        acc.red += w * colour.red * lit;
        acc.green += w * colour.green * lit;
        acc.blue += w * colour.blue * lit;
        acc.alpha += w;
    }
}

void captureForDebug(DebugView view, const Sample &sample, const Lighting &lit, DebugCapture &capture) noexcept
{
    capture.captured = true;
    capture.level = sample.level;
    switch (view)
    {
    case DebugView::Level: capture.value = static_cast<core::f32>(sample.level); break;
    case DebugView::GradientMagnitude: capture.value = lit.gradientLength; break;
    case DebugView::Shade: capture.value = lit.shade; break;
    case DebugView::Normal:
        capture.value = 1.0f;
        capture.normal[0] = lit.normal[0];
        capture.normal[1] = lit.normal[1];
        capture.normal[2] = lit.normal[2];
        break;
    default: capture.value = 0.0f; break;
    }
}

[[nodiscard]] core::u32 debugPixel(DebugView view, const DebugCapture &capture, core::u32 takenSteps) noexcept
{
    core::f32 v = 0.0f;
    switch (view)
    {
    case DebugView::Level:
        v = capture.captured ?
                (static_cast<core::f32>(capture.level) + 1.0f) / static_cast<core::f32>(kMaxPyramidLevels) :
                0.0f;
        break;
    case DebugView::GradientMagnitude: v = capture.captured ? capture.value * (1.0f / 128.0f) : 0.0f; break;
    case DebugView::Shade: v = capture.captured ? capture.value : 0.0f; break;
    case DebugView::StepCount: v = static_cast<core::f32>(takenSteps) * (1.0f / 512.0f); break;
    case DebugView::Normal:
        // The view that finds geometry trouble fastest: a normal quantised onto a handful of
        // directions comes out as flat patches of pure colour. A grey pixel is a sample with no
        // gradient at all.
        return packColour(capture.normal[0] * 0.5f + 0.5f, capture.normal[1] * 0.5f + 0.5f,
                          capture.normal[2] * 0.5f + 0.5f);
    default: break;
    }
    return packColour(v, v, v);
}

struct RayResult final {
    Accumulated acc{};
    core::u32 takenSteps{0u};
    DebugCapture debug{};
};

/// Integrates one ray, front to back, from where it enters the volume to where it stops.
[[nodiscard]] RayResult marchRay(const Frame &frame, const math::Vec3<core::f32> &origin,
                                 const math::Vec3<core::f32> &direction, const Slab &slab, core::usize pixelIndex,
                                 MarchReport &report) noexcept
{
    const MarchParams &params = frame.params;
    core::f32 volumeDirection[3];
    toVolumeAxes(direction, 1.0f, volumeDirection);

    // A hop lands on a point of the step grid: the position is an INTEGER count of steps from where
    // the ray entered, never an accumulated float, so a skipped ray samples exactly the points the
    // unskipped ray samples after the skip.
    core::f32 t = slab.enter > 0.0f ? slab.enter : 0.0f;
    const core::f32 tOrigin = t;
    core::i64 stepIndex = 0;
    const auto advanceBySteps = [&](core::i64 steps) noexcept {
        stepIndex += steps;
        t = tOrigin + static_cast<core::f32>(stepIndex) * frame.stepMetres;
    };
    const auto hop = [&](core::f32 samples) noexcept {
        advanceBySteps(frame.stepMetres > 0.0f ? stepsCovering(samples * frame.metresPerSample, frame.stepMetres) : 0);
    };
    const core::f32 tEnd = slab.exit < t + params.maxDistanceMetres ? slab.exit : t + params.maxDistanceMetres;

    RayResult ray{};
    Accumulated &acc = ray.acc;
    bool layerPainted[kMaxSurfaceLayers]{};
    for (; ray.takenSteps < params.maxSteps && t < tEnd && acc.alpha < params.opaqueAt; ++ray.takenSteps)
    {
        compositeSurfaces(frame, t, pixelIndex, layerPainted, acc, report);
        if (acc.alpha >= params.opaqueAt)
            break;

        core::f32 position[3];
        toVolumeAxes(origin + direction * t, frame.samplesPerMetre, position);

        const Sample sample = sampleField(frame.mosaic, position, params.trilinear, params.levelBlendSamples);
        ++report.steps;

        if (!sample.inside)
        {
            ++report.missingBricks;
            hop(samplesToFinestBoxExit(frame.mosaic, position, volumeDirection));
            continue;
        }

        const TransferFunction &curve = frame.curves[sample.level];
        if (params.skipEmptyCells && sample.brick->occupancy != nullptr)
        {
            const std::optional<core::f32> cell =
                samplesToEmptyCellExit(*sample.brick, curve, position, volumeDirection);
            if (cell.has_value())
            {
                ++report.skippedCells;
                const core::f32 finest = samplesToFinestBoxExit(frame.mosaic, position, volumeDirection);
                hop(*cell < finest ? *cell : finest);
                continue;
            }
        }
        if (!curve.reachesVisible(sample.brick->highest))
        {
            ++report.skippedBricks;
            hop(samplesToFinestBoxExit(frame.mosaic, position, volumeDirection));
            continue;
        }

        Lighting lit{};
        lit.alphaPerSample = curve.alpha[sample.density];
        // Gated on what the sample will actually CONTRIBUTE, not on whether it is visible at all:
        // later samples are weighted by (1 - accumulated alpha), so the threshold tightens by
        // itself as a ray fills. @see MarchParams::shadingCutoff
        if (lit.alphaPerSample > 0.0f && (1.0f - acc.alpha) * lit.alphaPerSample > params.shadingCutoff &&
            (params.shading > 0.0f || params.boundaryOpacity > 0.0f))
            lit = lightFromGradient(frame, sample, lit.alphaPerSample, position, direction, report);

        if (lit.alphaPerSample > 0.0f)
        {
            if (params.debug != DebugView::Off && !ray.debug.captured)
                captureForDebug(params.debug, sample, lit, ray.debug);

            // Opacity is declared per level-0 sample of path, so a longer step must absorb more.
            // Skipping this makes the same matter look denser purely because the camera moved
            // back, which reads as the object changing at a level boundary.
            const core::f32 w = (1.0f - acc.alpha) * opacityOverStep(lit.alphaPerSample, params.stepSamples);
            Rgb colour{curve.red[sample.density] * lit.shade, curve.green[sample.density] * lit.shade,
                       curve.blue[sample.density] * lit.shade};
            tintWithOverlay(params, position, lit.shade, colour, report);
            acc.red += w * colour.red;
            acc.green += w * colour.green;
            acc.blue += w * colour.blue;
            acc.alpha += w;
        }
        advanceBySteps(1);
    }
    return ray;
}

[[nodiscard]] core::u32 renderPixel(const Frame &frame, const math::Vec3<core::f32> &origin,
                                    const math::Vec3<core::f32> &direction, const core::f32 boxSize[3],
                                    core::usize pixelIndex, MarchReport &report) noexcept
{
    const MarchParams &params = frame.params;
    const Slab slab = intersectBox(origin, direction, boxSize);
    if (!slab.hit || slab.exit <= 0.0f)
    {
        ++report.escaped;
        return params.background;
    }

    const RayResult ray = marchRay(frame, origin, direction, slab, pixelIndex, report);
    if (ray.acc.alpha >= params.opaqueAt)
        ++report.saturated;
    else
        ++report.escaped;

    if (params.debug != DebugView::Off)
        return debugPixel(params.debug, ray.debug, ray.takenSteps);

    const core::f32 bgR = static_cast<core::f32>((params.background >> 16) & 0xFFu) * (1.0f / 255.0f);
    const core::f32 bgG = static_cast<core::f32>((params.background >> 8) & 0xFFu) * (1.0f / 255.0f);
    const core::f32 bgB = static_cast<core::f32>(params.background & 0xFFu) * (1.0f / 255.0f);
    const core::f32 rest = 1.0f - ray.acc.alpha;
    return packColour(ray.acc.red + rest * bgR, ray.acc.green + rest * bgG, ray.acc.blue + rest * bgB);
}

} // namespace

core::f32 wrapAngle(core::f32 radians) noexcept
{
    constexpr core::f64 kPi = 3.14159265358979323846;
    constexpr core::f64 kTwoPi = 2.0 * kPi;
    // In double: the float error of 2 pi, times the number of turns, would otherwise grow into a
    // visible error a few thousand radians out. And only so far: past about 1e16 turns a double no
    // longer holds the count exactly, and the remainder can land several radians off the circle.
    constexpr core::f64 kLargestTurnCount = 1.0e8;
    const core::f64 a = static_cast<core::f64>(radians);
    const core::f64 turns = (a + kPi) / kTwoPi;
    if (!(turns > -kLargestTurnCount && turns < kLargestTurnCount))
        return 0.0f;
    return static_cast<core::f32>(a - static_cast<core::f64>(floorToI64(turns)) * kTwoPi);
}

core::f32 wrappedSine(core::f32 radians) noexcept
{
    constexpr core::f32 kPi = 3.14159265358979f;
    core::f32 a = wrapAngle(radians);

    // sin(pi - a) == sin(a) and sin(-pi - a) == sin(a): both reflections are sign-preserving, so
    // the polynomial only ever sees |a| <= pi/2, where its first omitted term stays under two parts
    // in ten thousand.
    if (a > kPi * 0.5f)
        a = kPi - a;
    else if (a < -kPi * 0.5f)
        a = -kPi - a;

    const core::f32 a2 = a * a;
    return a * (1.0f - a2 * (1.0f / 6.0f - a2 * (1.0f / 120.0f - a2 * (1.0f / 5040.0f))));
}

core::f32 wrappedCosine(core::f32 radians) noexcept { return wrappedSine(wrapAngle(radians) + 1.57079632679f); }

void FreeCamera::turn(core::f32 dYaw, core::f32 dPitch) noexcept
{
    constexpr core::f32 kLimit = 1.55334306f; // Just short of straight up: 89 degrees.
    _yaw = wrapAngle(_yaw + dYaw);
    _pitch += dPitch;
    if (_pitch > kLimit)
        _pitch = kLimit;
    if (_pitch < -kLimit)
        _pitch = -kLimit;
}

void FreeCamera::move(core::f32 forward, core::f32 strafe, core::f32 rise, core::f32 distance) noexcept
{
    const Eye e = eye();
    math::Vec3<core::f32> step = e.forward * forward + e.right * strafe;
    step.y += rise;
    if (step.lengthSquared() < 1e-12f)
        return;
    position += step.normalize() * distance;
}

Eye FreeCamera::eye() const noexcept
{
    const core::f32 cy = wrappedCosine(_yaw);
    const core::f32 sy = wrappedSine(_yaw);
    const core::f32 cp = wrappedCosine(_pitch);
    const core::f32 sp = wrappedSine(_pitch);

    Eye e{};
    e.position = position;
    e.forward = math::Vec3<core::f32>{sy * cp, sp, cy * cp}.normalize();
    e.right = math::Vec3<core::f32>{cy, 0.0f, -sy}.normalize();
    e.up = math::Vec3<core::f32>{-sy * sp, cp, -cy * sp}.normalize();
    e.horizontalFieldOfView = horizontalFieldOfView;
    return e;
}

ImagePlane imagePlane(const Eye &eye, core::u32 width, core::u32 height) noexcept
{
    // tan(fov/2) from CORDIC, as Mat4::perspective takes it: no libm, and no truncated series,
    // whose error grows with the angle -- three terms were already 1.3 % short at 90 degrees.
    math::Fixed32 halfSine{};
    math::Fixed32 halfCosine{};
    math::Cordic::sincos(math::Fixed32::fromFloat(eye.horizontalFieldOfView * 0.5f), halfSine, halfCosine);
    return ImagePlane{.tanHalfFieldOfView = halfSine.toFloat() / halfCosine.toFloat(),
                      .aspect = static_cast<core::f32>(height) / static_cast<core::f32>(width),
                      .width = static_cast<core::f32>(width),
                      .height = static_cast<core::f32>(height)};
}

MarchReport march(const BrickMosaic &mosaic, const VolumeGeometry &geometry, const DensityProfile &profile,
                  const TransferFunction &transfer, const Eye &eye, const MarchParams &params, core::u32 *pixels,
                  core::u32 width, core::u32 height, core::u32 rowFirst, core::u32 rowCount) noexcept
{
    MarchReport report{};
    if (pixels == nullptr || width == 0u || height == 0u || !geometry.valid())
        return report;

    const core::f32 metresPerSample = geometry.metresPerSample();
    Frame frame{.mosaic = mosaic,
                .params = params,
                .curves = {},
                .metresPerSample = metresPerSample,
                .samplesPerMetre = metresPerSample > 0.0f ? 1.0f / metresPerSample : 0.0f,
                .stepMetres = params.stepSamples * metresPerSample};
    for (core::u32 level = 0u; level < kMaxPyramidLevels; ++level)
        frame.curves[level] = transfer.forLevel(profile, level);

    const core::f32 boxSize[3]{geometry.extentMetres(kAxisOfWorldX), geometry.extentMetres(kAxisOfWorldY),
                               geometry.extentMetres(kAxisOfWorldZ)};

    const ImagePlane plane = imagePlane(eye, width, height);
    const core::u32 rowEnd = (rowFirst + rowCount > height) ? height : rowFirst + rowCount;
    for (core::u32 py = rowFirst; py < rowEnd; ++py)
    {
        const core::f32 upSlope = plane.upSlope(static_cast<core::f32>(py) + 0.5f);
        for (core::u32 px = 0u; px < width; ++px)
        {
            const core::f32 rightSlope = plane.rightSlope(static_cast<core::f32>(px) + 0.5f);
            const math::Vec3<core::f32> direction =
                (eye.forward + eye.right * rightSlope + eye.up * upSlope).normalize();
            ++report.rays;
            const core::usize pixelIndex = static_cast<core::usize>(py) * width + px;
            pixels[pixelIndex] = renderPixel(frame, eye.position, direction, boxSize, pixelIndex, report);
        }
    }
    return report;
}

core::u32 foldFrame(const core::u32 *pixels, core::u32 count) noexcept
{
    core::u32 hash = 0x811C9DC5u;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const core::u32 v = pixels[i];
        for (core::u32 b = 0u; b < 4u; ++b)
        {
            hash ^= (v >> (b * 8u)) & 0xFFu;
            hash *= 0x01000193u;
        }
    }
    return hash;
}

} // namespace lpl::voxel
