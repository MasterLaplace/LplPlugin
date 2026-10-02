/**
 * @file Raymarch.cpp
 * @brief Front-to-back volume integration over a resident brick mosaic.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Raymarch.hpp>

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

[[nodiscard]] constexpr core::i64 floorToI64(core::f32 v) noexcept
{
    const core::i64 t = static_cast<core::i64>(v);
    return (v < 0.0f && static_cast<core::f32>(t) != v) ? t - 1 : t;
}

[[nodiscard]] constexpr core::u32 packColour(core::f32 r, core::f32 g, core::f32 b) noexcept
{
    const auto q = [](core::f32 v) noexcept -> core::u32 {
        const core::f32 c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return static_cast<core::u32>(c * 255.0f + 0.5f);
    };
    return 0xFF000000u | (q(r) << 16) | (q(g) << 8) | q(b);
}

/**
 * One sample of the field at a level-0 position, plus which level answered.
 *
 * @warning Trilinear interpolation stops at a brick edge, on purpose. Doing it properly across the
 * seam costs eight mosaic lookups per sample instead of one, and the mosaic lookup is the
 * expensive part of a step; the price is a one-sample seam between bricks, which at these
 * densities is below the noise the scan already has. Somebody who needs the seam gone should pad
 * bricks by one on load, not make every sample pay for it.
 */
struct Sample final {
    core::u8 density{0};
    core::u32 level{0};
    bool inside{false};
    const BrickView *brick{nullptr};
};

/// Density of one brick at a level-0 position, without any level blending.
[[nodiscard]] core::u8 sampleBrick(const BrickView &brick, core::f32 sz, core::f32 sy, core::f32 sx,
                                   bool trilinear) noexcept;

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
[[nodiscard]] core::u8 crossBrickSample(const BrickMosaic &mosaic, const BrickView &brick, core::f32 sz, core::f32 sy,
                                        core::f32 sx, bool trilinear) noexcept;

[[nodiscard]] Sample sampleField(const BrickMosaic &mosaic, core::f32 sz, core::f32 sy, core::f32 sx, bool trilinear,
                                 core::f32 blendBand, const BrickView *hint = nullptr) noexcept
{
    Sample out{};
    const core::i64 iz = floorToI64(sz);
    const core::i64 iy = floorToI64(sy);
    const core::i64 ix = floorToI64(sx);

    // ⚡ **The lookup is the expensive half of a sample, and the gradient asks for six of them a
    // step -- almost always inside the brick the centre is already in.** A hint skips the index
    // entirely for those, WITHOUT skipping anything else the function does: the level blend still
    // runs, so the gradient and the colour agree at the level boundaries the blend exists to hide.
    // An earlier version shortcut the whole function instead and moved the picture.
    //
    // ⚠ The hint is only trusted at the FINEST resident level, where nothing can shadow it. Two
    // samples away from a centre that sits in a coarse brick, a finer brick may well take over,
    // and reusing the coarse one there would quietly lower the detail of every gradient near a
    // level boundary.
    const BrickView *brick = nullptr;
    if (hint != nullptr && hint->key.level == mosaic.finestLevel() &&
        brickIndexOfBase(iz, hint->key.level) == hint->key.z && brickIndexOfBase(iy, hint->key.level) == hint->key.y &&
        brickIndexOfBase(ix, hint->key.level) == hint->key.x)
        brick = hint;
    else
        brick = mosaic.find(iz, iy, ix);
    if (brick == nullptr)
        return out;

    out.brick = brick;
    out.level = brick->key.level;
    out.inside = true;
    out.density = crossBrickSample(mosaic, *brick, sz, sy, sx, trilinear);

    if (blendBand <= 0.0f)
        return out;

    // How deep inside this brick the point sits, on the axis where it is shallowest.
    const core::i64 span = brickSpanInBaseSamples(brick->key.level);
    const core::f32 low[3]{sz - static_cast<core::f32>(static_cast<core::i64>(brick->key.z) * span),
                           sy - static_cast<core::f32>(static_cast<core::i64>(brick->key.y) * span),
                           sx - static_cast<core::f32>(static_cast<core::i64>(brick->key.x) * span)};
    core::f32 depth = 3.0e38f;
    core::u32 axis = 0u;
    bool towardsLow = true;
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        const core::f32 dLow = low[a];
        const core::f32 dHigh = static_cast<core::f32>(span) - low[a];
        if (dLow < depth)
        {
            depth = dLow;
            axis = a;
            towardsLow = true;
        }
        if (dHigh < depth)
        {
            depth = dHigh;
            axis = a;
            towardsLow = false;
        }
    }
    if (depth >= blendBand)
        return out;

    // Only blend where the detail actually STOPS. If the neighbour at this level is resident there
    // is no seam to hide, and softening across it would throw away detail that is present.
    core::f32 probe[3]{sz, sy, sx};
    probe[axis] += towardsLow ? -(blendBand + 1.0f) : (blendBand + 1.0f);
    const BrickView *neighbour = mosaic.find(floorToI64(probe[0]), floorToI64(probe[1]), floorToI64(probe[2]));
    if (neighbour != nullptr && neighbour->key.level == brick->key.level)
        return out;

    const BrickView *coarse = mosaic.findCoarserThan(brick->key.level, iz, iy, ix);
    if (coarse == nullptr)
        return out;

    const core::f32 w = depth / blendBand; // 0 at the face, 1 at the inner edge of the band.
    const core::f32 fine = static_cast<core::f32>(out.density);
    const core::f32 far = static_cast<core::f32>(crossBrickSample(mosaic, *coarse, sz, sy, sx, trilinear));
    const core::f32 mixed = far + (fine - far) * w;
    out.density = static_cast<core::u8>(mixed < 0.0f ? 0.0f : (mixed > 255.0f ? 255.0f : mixed + 0.5f));
    return out;
}

core::u8 crossBrickSample(const BrickMosaic &mosaic, const BrickView &brick, core::f32 sz, core::f32 sy, core::f32 sx,
                          bool trilinear) noexcept
{
    const core::u32 level = brick.key.level;
    const core::i64 step = static_cast<core::i64>(1) << level;
    const core::f32 stepF = static_cast<core::f32>(step);

    // Snap to this brick's sample lattice so the eight corners are the same eight everywhere in
    // the cell -- otherwise neighbouring rays interpolate over different corners and the surface
    // shimmers.
    const core::f32 base[3]{sz, sy, sx};
    core::i64 corner[3];
    core::f32 frac[3];
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        const core::f32 inLattice = base[a] / stepF;
        const core::i64 whole = floorToI64(inLattice);
        corner[a] = whole * step;
        frac[a] = inLattice - static_cast<core::f32>(whole);
    }

    if (!trilinear)
        return sampleBrick(brick, sz, sy, sx, false);

    // Fast path: the whole cell is inside this brick.
    const core::i64 span = brickSpanInBaseSamples(level);
    bool inside = true;
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        const core::i64 origin =
            (a == 0u ? brick.key.z : (a == 1u ? brick.key.y : brick.key.x)) * static_cast<core::i64>(span);
        if (corner[a] + step >= origin + span)
            inside = false;
    }
    if (inside)
        return sampleBrick(brick, sz, sy, sx, true);

    core::f32 c[2][2][2];
    for (core::u32 dz = 0u; dz < 2u; ++dz)
    {
        for (core::u32 dy = 0u; dy < 2u; ++dy)
        {
            for (core::u32 dx = 0u; dx < 2u; ++dx)
            {
                const core::f32 pz = static_cast<core::f32>(corner[0] + static_cast<core::i64>(dz) * step);
                const core::f32 py = static_cast<core::f32>(corner[1] + static_cast<core::i64>(dy) * step);
                const core::f32 px = static_cast<core::f32>(corner[2] + static_cast<core::i64>(dx) * step);
                const BrickView *owner = mosaic.find(floorToI64(pz), floorToI64(py), floorToI64(px));
                // No neighbour: the brick's own edge sample stands in. Reporting zero would make
                // the boundary look like a wall of vacuum.
                c[dz][dy][dx] = static_cast<core::f32>(owner != nullptr ? sampleBrick(*owner, pz, py, px, false) :
                                                                          sampleBrick(brick, pz, py, px, false));
            }
        }
    }

    const core::f32 c00 = c[0][0][0] * (1.0f - frac[2]) + c[0][0][1] * frac[2];
    const core::f32 c01 = c[0][1][0] * (1.0f - frac[2]) + c[0][1][1] * frac[2];
    const core::f32 c10 = c[1][0][0] * (1.0f - frac[2]) + c[1][0][1] * frac[2];
    const core::f32 c11 = c[1][1][0] * (1.0f - frac[2]) + c[1][1][1] * frac[2];
    const core::f32 c0 = c00 * (1.0f - frac[1]) + c01 * frac[1];
    const core::f32 c1 = c10 * (1.0f - frac[1]) + c11 * frac[1];
    const core::f32 v = c0 * (1.0f - frac[0]) + c1 * frac[0];
    return static_cast<core::u8>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v + 0.5f));
}

core::u8 sampleBrick(const BrickView &brick, core::f32 sz, core::f32 sy, core::f32 sx, bool trilinear) noexcept
{
    const core::u32 level = brick.key.level;
    const core::i64 span = brickSpanInBaseSamples(level);
    const core::i64 originZ = static_cast<core::i64>(brick.key.z) * span;
    const core::i64 originY = static_cast<core::i64>(brick.key.y) * span;
    const core::i64 originX = static_cast<core::i64>(brick.key.x) * span;

    const core::f32 scale = 1.0f / static_cast<core::f32>(static_cast<core::i64>(1) << level);
    const core::f32 lz = (sz - static_cast<core::f32>(originZ)) * scale;
    const core::f32 ly = (sy - static_cast<core::f32>(originY)) * scale;
    const core::f32 lx = (sx - static_cast<core::f32>(originX)) * scale;

    core::i64 cz = floorToI64(lz);
    core::i64 cy = floorToI64(ly);
    core::i64 cx = floorToI64(lx);
    const core::i64 edge = static_cast<core::i64>(kBrickEdge);
    if (cz < 0)
        cz = 0;
    if (cy < 0)
        cy = 0;
    if (cx < 0)
        cx = 0;
    if (cz >= edge)
        cz = edge - 1;
    if (cy >= edge)
        cy = edge - 1;
    if (cx >= edge)
        cx = edge - 1;

    if (!trilinear)
        return brick.at(static_cast<core::u32>(cz), static_cast<core::u32>(cy), static_cast<core::u32>(cx));
    if (cz + 1 >= edge || cy + 1 >= edge || cx + 1 >= edge)
        return 0u; // Caller resolves the corners through the mosaic; see crossBrickSample.

    const core::f32 fz = lz - static_cast<core::f32>(cz);
    const core::f32 fy = ly - static_cast<core::f32>(cy);
    const core::f32 fx = lx - static_cast<core::f32>(cx);
    const auto g = [&](core::i64 dz, core::i64 dy, core::i64 dx) noexcept -> core::f32 {
        return static_cast<core::f32>(brick.at(static_cast<core::u32>(cz + dz), static_cast<core::u32>(cy + dy),
                                               static_cast<core::u32>(cx + dx)));
    };
    const core::f32 c00 = g(0, 0, 0) * (1.0f - fx) + g(0, 0, 1) * fx;
    const core::f32 c01 = g(0, 1, 0) * (1.0f - fx) + g(0, 1, 1) * fx;
    const core::f32 c10 = g(1, 0, 0) * (1.0f - fx) + g(1, 0, 1) * fx;
    const core::f32 c11 = g(1, 1, 0) * (1.0f - fx) + g(1, 1, 1) * fx;
    const core::f32 c0 = c00 * (1.0f - fy) + c01 * fy;
    const core::f32 c1 = c10 * (1.0f - fy) + c11 * fy;
    const core::f32 v = c0 * (1.0f - fz) + c1 * fz;
    return static_cast<core::u8>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v + 0.5f));
}

/**
 * Distance along the ray, in level-0 samples, from the current point to where it leaves the brick
 * it is in.
 *
 * @warning **Jumping a whole brick span instead is a real bug, and it looks like tiling.** A ray
 * usually enters a brick partway through, so advancing by a full span from wherever it happens to
 * be lands somewhere inside the NEXT brick -- skipping whatever was visible in between. Because
 * the amount skipped depends on where the ray crossed, the error is different for every brick, and
 * the picture comes out in rectangular patches of differing brightness. That is exactly what the
 * first real renders showed, and reading the code had blamed the level of detail for it.
 */
[[nodiscard]] core::f32 samplesToBoxExit(const core::f32 lowBound[3], core::f32 span, core::f32 pz, core::f32 py,
                                         core::f32 px, core::f32 dz, core::f32 dy, core::f32 dx) noexcept
{
    const core::f32 p[3]{pz, py, px};
    const core::f32 d[3]{dz, dy, dx};

    core::f32 best = 3.0e38f;
    for (core::u32 a = 0u; a < 3u; ++a)
    {
        if (d[a] > -1e-9f && d[a] < 1e-9f)
            continue;
        const core::f32 bound = d[a] > 0.0f ? lowBound[a] + span : lowBound[a];
        const core::f32 t = (bound - p[a]) / d[a];
        if (t > 0.0f && t < best)
            best = t;
    }
    return best >= 3.0e38f ? span : best;
}

/// The brick's own box, in the same terms.
[[nodiscard]] core::f32 samplesToBrickExit(core::u32 level, const core::i32 index[3], core::f32 pz, core::f32 py,
                                           core::f32 px, core::f32 dz, core::f32 dy, core::f32 dx) noexcept
{
    const core::i64 span = brickSpanInBaseSamples(level);
    const core::f32 lowBound[3]{static_cast<core::f32>(static_cast<core::i64>(index[0]) * span),
                                static_cast<core::f32>(static_cast<core::i64>(index[1]) * span),
                                static_cast<core::f32>(static_cast<core::i64>(index[2]) * span)};
    return samplesToBoxExit(lowBound, static_cast<core::f32>(span), pz, py, px, dz, dy, dx);
}

/// (1 - a)^n without libm: n is a small integer count of level-0 samples per step.
[[nodiscard]] core::f32 opacityOverStep(core::f32 perSample, core::f32 samplesPerStep) noexcept
{
    if (perSample <= 0.0f)
        return 0.0f;
    if (perSample >= 1.0f)
        return 1.0f;
    // Repeated squaring on the transparency, with the fractional remainder handled linearly. An
    // exp/log pair would be one line and would put a transcendental on the hot path of every
    // sample, which this tree does not spend.
    core::f32 keep = 1.0f - perSample;
    core::f32 acc = 1.0f;
    core::u32 whole = static_cast<core::u32>(samplesPerStep);
    core::f32 base = keep;
    while (whole != 0u)
    {
        if ((whole & 1u) != 0u)
            acc *= base;
        base *= base;
        whole >>= 1u;
    }
    const core::f32 frac = samplesPerStep - static_cast<core::f32>(static_cast<core::u32>(samplesPerStep));
    acc *= 1.0f - frac * (1.0f - keep);
    const core::f32 a = 1.0f - acc;
    return a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
}

} // namespace

core::f32 wrappedSine(core::f32 radians) noexcept
{
    constexpr core::f32 kPi = 3.14159265358979f;
    constexpr core::f32 kTwoPi = 6.28318530717959f;

    // Bounded, not `while`: an angle that arrived as a very large number -- or as an infinity from
    // a divide somewhere upstream -- would spin here forever, and a renderer that hangs is worse
    // than one that draws the wrong direction.
    core::f32 a = radians;
    for (int i = 0; i < 64 && (a > kPi || a < -kPi); ++i)
        a += a > kPi ? -kTwoPi : kTwoPi;
    if (a > kPi || a < -kPi)
        return 0.0f;

    // sin(pi - a) == sin(a) and sin(-pi - a) == sin(a): both reflections are sign-preserving, so
    // the polynomial only ever sees |a| <= pi/2 where it is accurate to a few parts in a million.
    if (a > kPi * 0.5f)
        a = kPi - a;
    else if (a < -kPi * 0.5f)
        a = -kPi - a;

    const core::f32 a2 = a * a;
    return a * (1.0f - a2 * (1.0f / 6.0f - a2 * (1.0f / 120.0f - a2 * (1.0f / 5040.0f))));
}

core::f32 wrappedCosine(core::f32 radians) noexcept { return wrappedSine(radians + 1.57079632679f); }

void FreeCamera::turn(core::f32 dYaw, core::f32 dPitch) noexcept
{
    constexpr core::f32 kTwoPi = 6.28318530717959f;
    constexpr core::f32 kLimit = 1.55334306f; // Just short of straight up: 89 degrees.
    yaw += dYaw;
    for (int i = 0; i < 64 && (yaw > kTwoPi || yaw < -kTwoPi); ++i)
        yaw += yaw > kTwoPi ? -kTwoPi : kTwoPi;
    pitch += dPitch;
    if (pitch > kLimit)
        pitch = kLimit;
    if (pitch < -kLimit)
        pitch = -kLimit;
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
    const core::f32 cy = wrappedCosine(yaw);
    const core::f32 sy = wrappedSine(yaw);
    const core::f32 cp = wrappedCosine(pitch);
    const core::f32 sp = wrappedSine(pitch);

    Eye e{};
    e.position = position;
    e.forward = math::Vec3<core::f32>{sy * cp, sp, cy * cp}.normalize();
    e.right = math::Vec3<core::f32>{cy, 0.0f, -sy}.normalize();
    e.up = math::Vec3<core::f32>{-sy * sp, cp, -cy * sp}.normalize();
    e.horizontalFieldOfView = horizontalFieldOfView;
    return e;
}

MarchReport march(const BrickMosaic &mosaic, const VolumeGeometry &geometry, const DensityProfile &profile,
                  const TransferFunction &transfer, const Eye &eye, const MarchParams &params, core::u32 *pixels,
                  core::u32 width, core::u32 height, core::u32 rowFirst, core::u32 rowCount) noexcept
{
    MarchReport report{};
    if (pixels == nullptr || width == 0u || height == 0u || !geometry.valid())
        return report;

    // One curve per level, derived once. Deriving it per sample would dominate the frame, and
    // deriving it per ray would still be thousands of times more often than it changes.
    TransferFunction byLevel[kMaxPyramidLevels];
    for (core::u32 lv = 0u; lv < kMaxPyramidLevels; ++lv)
        byLevel[lv] = transfer.forLevel(profile, lv);

    const core::f32 metresPerSample = geometry.metresPerSample();
    const core::f32 samplesPerMetre = metresPerSample > 0.0f ? 1.0f / metresPerSample : 0.0f;
    const core::f32 boxSize[3]{geometry.extentMetres(kAxisOfWorldX), geometry.extentMetres(kAxisOfWorldY),
                               geometry.extentMetres(kAxisOfWorldZ)};

    const core::f32 aspect = static_cast<core::f32>(height) / static_cast<core::f32>(width);
    // tan(fov/2) without libm: the small-angle-safe rational that Cordic-free code in this tree
    // already uses for a field of view. Accurate to a fraction of a pixel over any sane fov.
    const core::f32 h = eye.horizontalFieldOfView * 0.5f;
    const core::f32 h2 = h * h;
    const core::f32 tanHalf = h * (1.0f + h2 * (1.0f / 3.0f + h2 * (2.0f / 15.0f)));

    const core::f32 stepMetres = params.stepSamples * metresPerSample;
    const core::f32 blendBand = params.levelBlendSamples;
    const core::u32 rowEnd = (rowFirst + rowCount > height) ? height : rowFirst + rowCount;

    for (core::u32 py = rowFirst; py < rowEnd; ++py)
    {
        const core::f32 ndcY =
            (1.0f - 2.0f * (static_cast<core::f32>(py) + 0.5f) / static_cast<core::f32>(height)) * aspect * tanHalf;
        for (core::u32 px = 0u; px < width; ++px)
        {
            const core::f32 ndcX =
                (2.0f * (static_cast<core::f32>(px) + 0.5f) / static_cast<core::f32>(width) - 1.0f) * tanHalf;

            math::Vec3<core::f32> dir = eye.forward + eye.right * ndcX + eye.up * ndcY;
            dir = dir.normalize();

            ++report.rays;
            core::u32 *out = &pixels[static_cast<core::usize>(py) * width + px];

            const Slab slab = intersectBox(eye.position, dir, boxSize);
            if (!slab.hit || slab.exit <= 0.0f)
            {
                *out = params.background;
                ++report.escaped;
                continue;
            }

            core::f32 t = slab.enter > 0.0f ? slab.enter : 0.0f;
            const core::f32 tOrigin = t;
            // ⚠⚠ **Every jump lands back on the step lattice, and that is what makes skipping
            // EXACT instead of merely fast.** A jump to a cell's exit leaves the ray at an
            // arbitrary offset, so every sample after it falls between the ones the unskipped
            // march would have taken -- and where the ray re-enters matter it starts at a
            // different phase of the ramp, so the colour differs. Measured: without the snap the
            // same frame folded to a different signature, which is a renderer whose picture
            // depends on an optimisation. Snapping costs one rounding per jump.
            // The ray's position is an INTEGER count of steps from where it entered, never an
            // accumulated float: accumulation drifts over hundreds of steps, and a jump computed
            // as a distance lands between lattice points. Counting makes both exact.
            core::i64 stepIndex = 0;
            const auto jumpBy = [&](core::f32 distance) noexcept {
                if (stepMetres <= 0.0f)
                    return;
                const core::f32 steps = distance / stepMetres;
                core::i64 whole = static_cast<core::i64>(steps);
                if (static_cast<core::f32>(whole) < steps)
                    ++whole;
                stepIndex += whole < 1 ? 1 : whole;
            };
            core::f32 tEnd = slab.exit;
            if (tEnd > t + params.maxDistanceMetres)
                tEnd = t + params.maxDistanceMetres;

            const core::usize pixelIndex = static_cast<core::usize>(py) * width + px;
            bool layerPainted[kMaxSurfaceLayers]{};
            core::f32 debugValue = -1.0f;
            core::u32 debugLevel = 0u;
            core::f32 debugNormal[3]{0.0f, 0.0f, 0.0f};
            core::f32 accR = 0.0f;
            core::f32 accG = 0.0f;
            core::f32 accB = 0.0f;
            core::f32 accA = 0.0f;

            core::u32 takenSteps = 0u;
            for (; takenSteps < params.maxSteps && t < tEnd && accA < params.opaqueAt; ++takenSteps)
            {
                // Each traced surface is composited when the ray reaches its depth: front-to-back,
                // so whatever the scan put in front of it still occludes it. A surface drawn on
                // top regardless would hide the very evidence it is supposed to be checked against.
                for (core::u32 layer = 0u; layer < params.surfaceCount && layer < kMaxSurfaceLayers; ++layer)
                {
                    const SurfaceLayer &surface = params.surfaces[layer];
                    if (layerPainted[layer] || !surface.valid() || t < surface.depth[pixelIndex])
                        continue;
                    layerPainted[layer] = true;
                    ++report.surfaceHits;

                    const core::f32 facing = surface.facing != nullptr ? surface.facing[pixelIndex] : 1.0f;
                    const core::f32 lit = 0.35f + 0.65f * facing;

                    core::f32 sr = surface.red;
                    core::f32 sg = surface.green;
                    core::f32 sb = surface.blue;
                    if (surface.textured() && params.surfaceTexture != nullptr &&
                        params.surfaceTexture[layer].u != nullptr)
                    {
                        // The prediction lives in the segment's own flattened coordinates; the
                        // rasteriser already carried them here, perspective-correct.
                        const core::f32 tu = params.surfaceTexture[layer].u[pixelIndex];
                        const core::f32 tv = params.surfaceTexture[layer].v[pixelIndex];
                        if (tu >= 0.0f && tu <= 1.0f && tv >= 0.0f && tv <= 1.0f)
                        {
                            core::u32 ix = static_cast<core::u32>(tu * static_cast<core::f32>(surface.inkWidth - 1u));
                            // v runs up in a texture and down in an image; flipping here is the one
                            // place that convention lives, rather than in every caller.
                            core::u32 iy =
                                static_cast<core::u32>((1.0f - tv) * static_cast<core::f32>(surface.inkHeight - 1u));
                            if (ix >= surface.inkWidth)
                                ix = surface.inkWidth - 1u;
                            if (iy >= surface.inkHeight)
                                iy = surface.inkHeight - 1u;
                            const core::u8 value = surface.ink[static_cast<core::usize>(iy) * surface.inkWidth + ix];
                            if (value > surface.inkFloor)
                            {
                                const core::f32 span = static_cast<core::f32>(255u - surface.inkFloor);
                                core::f32 strength = static_cast<core::f32>(value - surface.inkFloor) / span;
                                if (strength > 1.0f)
                                    strength = 1.0f;
                                // Confidence multiplies the tint. A map worth 0.55 must not look
                                // as solid as one worth 0.95.
                                const core::f32 mix = strength * surface.inkConfidence;
                                sr += (surface.inkRed - sr) * mix;
                                sg += (surface.inkGreen - sg) * mix;
                                sb += (surface.inkBlue - sb) * mix;
                                ++report.inkPainted;
                            }
                        }
                    }
                    if (surface.throughMatter)
                    {
                        // X-ray: put it in front of everything gathered so far. See the field's
                        // own warning for why this is not the default.
                        const core::f32 keep = 1.0f - surface.opacity;
                        accR = accR * keep + surface.opacity * sr * lit;
                        accG = accG * keep + surface.opacity * sg * lit;
                        accB = accB * keep + surface.opacity * sb * lit;
                        accA = accA * keep + surface.opacity;
                    }
                    else
                    {
                        const core::f32 w = (1.0f - accA) * surface.opacity;
                        accR += w * sr * lit;
                        accG += w * sg * lit;
                        accB += w * sb * lit;
                        accA += w;
                    }
                }
                if (accA >= params.opaqueAt)
                    break;

                const math::Vec3<core::f32> p = eye.position + dir * t;
                const core::f32 sx = p.x * samplesPerMetre;
                const core::f32 sy = p.y * samplesPerMetre;
                const core::f32 sz = p.z * samplesPerMetre;
                // World axes back to volume axes; see kAxisOfWorld*.
                const core::f32 vz = sz;
                const core::f32 vy = sy;
                const core::f32 vx = sx;

                const Sample sample = sampleField(mosaic, vz, vy, vx, params.trilinear, blendBand);
                ++report.steps;

                if (!sample.inside)
                {
                    // No resident brick here. Advance to where the coarsest brick that COULD be
                    // here would end, so a hole is crossed in one hop rather than in thousands of
                    // lookups that all fail -- and to the exit rather than by a fixed span, or the
                    // jump overshoots into whatever comes next.
                    ++report.missingBricks;
                    const core::u32 level = mosaic.coarsestLevel();
                    const core::i32 index[3]{brickIndexOfBase(floorToI64(vz), level),
                                             brickIndexOfBase(floorToI64(vy), level),
                                             brickIndexOfBase(floorToI64(vx), level)};
                    const core::f32 exit =
                        samplesToBrickExit(level, index, vz, vy, vx, dir.z, dir.y, dir.x) * metresPerSample;
                    jumpBy(exit > stepMetres ? exit : stepMetres);
                    t = tOrigin + static_cast<core::f32>(stepIndex) * stepMetres;
                    continue;
                }

                const TransferFunction &curve = byLevel[sample.level];

                // ⚡ **The cell before the brick, because the medium is almost all of a scan.**
                // Ninety-eight per cent of the samples a frame takes paint nothing: they sit in
                // the material between sheets, which is dark and NOT empty, so the whole-brick
                // summary can never skip it. A cell that cannot reach the visible band is left in
                // one comparison rather than sixteen samples -- and it is exact, not a heuristic:
                // the test is on the cell's highest sample, so nothing visible is stepped over.
                if (params.skipEmptyCells && sample.brick->occupancy != nullptr)
                {
                    const core::u32 level = sample.brick->key.level;
                    const core::i64 span = brickSpanInBaseSamples(level);
                    const core::i64 originZ = static_cast<core::i64>(sample.brick->key.z) * span;
                    const core::i64 originY = static_cast<core::i64>(sample.brick->key.y) * span;
                    const core::i64 originX = static_cast<core::i64>(sample.brick->key.x) * span;
                    const core::u32 lz = static_cast<core::u32>((floorToI64(vz) - originZ) >> level);
                    const core::u32 ly = static_cast<core::u32>((floorToI64(vy) - originY) >> level);
                    const core::u32 lx = static_cast<core::u32>((floorToI64(vx) - originX) >> level);
                    if (lz < kBrickEdge && ly < kBrickEdge && lx < kBrickEdge &&
                        sample.brick->cellHighest(lz, ly, lx) < curve.firstVisible)
                    {
                        ++report.skippedCells;
                        const core::f32 cellSpan =
                            static_cast<core::f32>(static_cast<core::i64>(1) << (kOccupancyShift + level));
                        const core::f32 low[3]{
                            static_cast<core::f32>(
                                originZ + (static_cast<core::i64>(lz >> kOccupancyShift) << (kOccupancyShift + level))),
                            static_cast<core::f32>(
                                originY + (static_cast<core::i64>(ly >> kOccupancyShift) << (kOccupancyShift + level))),
                            static_cast<core::f32>(originX + (static_cast<core::i64>(lx >> kOccupancyShift)
                                                              << (kOccupancyShift + level)))};
                        const core::f32 exit =
                            samplesToBoxExit(low, cellSpan, vz, vy, vx, dir.z, dir.y, dir.x) * metresPerSample;
                        jumpBy(exit > stepMetres ? exit : stepMetres);
                        t = tOrigin + static_cast<core::f32>(stepIndex) * stepMetres;
                        continue;
                    }
                }

                if (!curve.anyVisible(sample.brick->lowest, sample.brick->highest))
                {
                    // Nothing in this whole brick can paint. One comparison instead of two million
                    // byte reads -- this is what the summary is for -- and the ray leaves at the
                    // brick's own exit, never by a blind span.
                    ++report.skippedBricks;
                    const core::i32 index[3]{sample.brick->key.z, sample.brick->key.y, sample.brick->key.x};
                    const core::f32 exit =
                        samplesToBrickExit(sample.level, index, vz, vy, vx, dir.z, dir.y, dir.x) * metresPerSample;
                    jumpBy(exit > stepMetres ? exit : stepMetres);
                    t = tOrigin + static_cast<core::f32>(stepIndex) * stepMetres;
                    continue;
                }

                core::f32 aPerSample = curve.alpha[sample.density];
                core::f32 shade = 1.0f;
                core::f32 gradientLength = 0.0f;
                core::f32 normalX = 0.0f;
                core::f32 normalY = 0.0f;
                core::f32 normalZ = 0.0f;
                // ⚠ **Gated on what the sample will actually CONTRIBUTE, not on whether it is
                // visible at all.** A ray's later samples are weighted by (1 - accumulated alpha),
                // so once it is nearly opaque they change nothing -- and paying six scattered
                // probes to shade a contribution below a thousandth is the frame's largest waste.
                // The threshold is on the weighted term, so it tightens by itself as a ray fills.
                if (aPerSample > 0.0f && (1.0f - accA) * aPerSample > params.shadingCutoff &&
                    (params.shading > 0.0f || params.boundaryOpacity > 0.0f))
                {
                    // Six extra samples, and only where something is actually going to be painted:
                    // computing a gradient in matter the curve maps to nothing would be the whole
                    // cost of the frame spent on invisible space.
                    ++report.gradients;

                    // ⚠⚠ **The stencil is TRILINEAR and wider than one sample, and both halves of
                    // that matter.** Samples are single bytes and local contrast inside a scan is
                    // low, so a nearest-neighbour difference over one sample is usually 0, 1 or 2
                    // -- a handful of integers, which quantises the normal into a handful of
                    // directions and paints the picture in flat polygons. A debug view of the
                    // shading term alone showed it as large grey rectangles, which is what finally
                    // identified this after four wrong diagnoses from reading the code.
                    // Interpolating gives a continuous difference; widening the stencil lifts it
                    // clear of the quantisation floor.
                    const core::f32 d = params.gradientSpread;

                    // ⚠ **A neighbour outside the resident set is NOT density zero.** Treating a
                    // missing brick as empty makes the difference across a brick face enormous, so
                    // every face of every resident brick lights up as a false surface -- and
                    // because the resident set is a grid, the picture comes out in rectangles. It
                    // was the most visible defect in the first real renders, and reading the code
                    // blamed the level of detail for it twice before an instrument settled it: the
                    // same frame with shading off has no tiles at all. Where there is no
                    // neighbour, the centre stands in, which reports "flat" rather than "cliff".
                    const core::f32 here = static_cast<core::f32>(sample.density);

                    // Every probe goes through the same function as the centre, with the
                    // centre's brick as a hint: same blending, same answer, no index lookup for
                    // the ones that land in the same brick -- which is nearly all of them.
                    const BrickView *own = sample.brick;
                    const auto neighbour = [&](core::f32 nz, core::f32 ny, core::f32 nx) noexcept {
                        // ⚠ **Nearest, not trilinear, and the wide stencil is what pays for it.**
                        // Measured: the six gradient probes are 73 % of a frame, and trilinear
                        // makes each of them eight scattered reads instead of one. What the
                        // interpolation was added for was the QUANTISATION of a one-sample
                        // difference on byte data -- and widening the stencil already fixes that,
                        // because the difference it takes is over four samples rather than two.
                        // The centre sample stays interpolated: that one decides the colour.
                        const Sample s = sampleField(mosaic, nz, ny, nx, false, blendBand, own);
                        return s.inside ? static_cast<core::f32>(s.density) : here;
                    };

                    const core::f32 gx = neighbour(vz, vy, vx + d) - neighbour(vz, vy, vx - d);
                    const core::f32 gy = neighbour(vz, vy + d, vx) - neighbour(vz, vy - d, vx);
                    const core::f32 gz = neighbour(vz + d, vy, vx) - neighbour(vz - d, vy, vx);

                    core::f32 len2 = gx * gx + gy * gy + gz * gz;
                    if (len2 > 1e-6f)
                    {
                        // Newton on the reciprocal square root: no libm, same discipline as the
                        // rest of the tree.
                        core::f32 r = len2;
                        core::f32 g = r > 1.0f ? r : 1.0f;
                        for (int k = 0; k < 12; ++k)
                            g = 0.5f * (g + r / g);
                        gradientLength = g;
                        const core::f32 inv = 1.0f / g;
                        // The gradient points across the surface; world axes again, so the dot
                        // with the view direction has to be taken in the same frame.
                        const core::f32 nx = gx * inv;
                        const core::f32 ny = gy * inv;
                        const core::f32 nz = gz * inv;
                        normalX = nx;
                        normalY = ny;
                        normalZ = nz;
                        core::f32 lambert = -(nx * dir.x + ny * dir.y + nz * dir.z);
                        // Two-sided: a sheet has no outside, so a normal that points away from the
                        // eye is the same surface seen from behind, not an unlit one.
                        if (lambert < 0.0f)
                            lambert = -lambert;
                        shade = params.ambient + (1.0f - params.ambient) * lambert;
                        shade = 1.0f + params.shading * (shade - 1.0f);

                        // Boundary opacity: homogeneous matter stays translucent, edges turn solid.
                        const core::f32 strength = g * (1.0f / (96.0f * params.gradientSpread));
                        const core::f32 edge = strength > 1.0f ? 1.0f : strength;
                        aPerSample *= 1.0f - params.boundaryOpacity * (1.0f - edge);
                    }
                    else
                    {
                        aPerSample *= 1.0f - params.boundaryOpacity;
                    }
                }
                if (params.debug != DebugView::Off && debugValue < 0.0f && aPerSample > 0.0f)
                {
                    debugLevel = sample.level;
                    switch (params.debug)
                    {
                    case DebugView::Level: debugValue = static_cast<core::f32>(sample.level); break;
                    case DebugView::GradientMagnitude: debugValue = gradientLength; break;
                    case DebugView::Shade: debugValue = shade; break;
                    case DebugView::Normal:
                        debugValue = 1.0f;
                        debugNormal[0] = normalX;
                        debugNormal[1] = normalY;
                        debugNormal[2] = normalZ;
                        break;
                    default: debugValue = 0.0f; break;
                    }
                }
                if (aPerSample > 0.0f)
                {
                    // Opacity is declared per level-0 sample of path, so a longer step must absorb
                    // more. Skipping this makes the same matter look denser purely because the
                    // camera moved back, which reads as the object changing at a level boundary.
                    const core::f32 a = opacityOverStep(aPerSample, params.stepSamples);
                    const core::f32 w = (1.0f - accA) * a;
                    core::f32 cr = curve.red[sample.density] * shade;
                    core::f32 cg = curve.green[sample.density] * shade;
                    core::f32 cb = curve.blue[sample.density] * shade;

                    if (params.overlay != nullptr && params.overlayConfidence > 0.0f)
                    {
                        // The overlay only ever tints matter the scan already put here: a
                        // prediction floating in a void would be the model drawing rather than
                        // reading, and the reader could not tell which they were looking at.
                        const Sample o = sampleField(*params.overlay, vz, vy, vx, params.trilinear, 0.0f);
                        if (o.inside && o.density > params.overlayFloor)
                        {
                            const core::f32 up = static_cast<core::f32>(params.overlayFull > params.overlayFloor ?
                                                                            params.overlayFull - params.overlayFloor :
                                                                            1u);
                            core::f32 strength = static_cast<core::f32>(o.density - params.overlayFloor) / up;
                            if (strength > 1.0f)
                                strength = 1.0f;
                            // Confidence multiplies the tint rather than being written under the
                            // picture: a map worth 0.55 must not look as solid as one worth 0.95.
                            const core::f32 mix = strength * params.overlayConfidence;
                            cr += (params.overlayRed * shade - cr) * mix;
                            cg += (params.overlayGreen * shade - cg) * mix;
                            cb += (params.overlayBlue * shade - cb) * mix;
                            ++report.overlaid;
                        }
                    }

                    accR += w * cr;
                    accG += w * cg;
                    accB += w * cb;
                    accA += w;
                }
                ++stepIndex;
                t = tOrigin + static_cast<core::f32>(stepIndex) * stepMetres;
            }

            if (accA >= params.opaqueAt)
                ++report.saturated;
            else
                ++report.escaped;

            if (params.debug != DebugView::Off)
            {
                core::f32 v = 0.0f;
                switch (params.debug)
                {
                case DebugView::Level:
                    v = debugValue < 0.0f ? 0.0f : (static_cast<core::f32>(debugLevel) + 1.0f) / 6.0f;
                    break;
                case DebugView::GradientMagnitude: v = debugValue < 0.0f ? 0.0f : debugValue * (1.0f / 128.0f); break;
                case DebugView::Shade: v = debugValue < 0.0f ? 0.0f : debugValue; break;
                case DebugView::StepCount: v = static_cast<core::f32>(takenSteps) * (1.0f / 512.0f); break;
                case DebugView::Normal:
                    // The classic normal-as-colour view, and it is the one that finds geometry
                    // trouble fastest: a normal quantised onto a handful of directions comes out
                    // as flat patches of pure colour, which no amount of reading the shading code
                    // makes as obvious. A grey pixel is a sample with no gradient at all.
                    *out = packColour(debugNormal[0] * 0.5f + 0.5f, debugNormal[1] * 0.5f + 0.5f,
                                      debugNormal[2] * 0.5f + 0.5f);
                    continue;
                default: break;
                }
                *out = packColour(v, v, v);
                continue;
            }

            const core::f32 bgR = static_cast<core::f32>((params.background >> 16) & 0xFFu) * (1.0f / 255.0f);
            const core::f32 bgG = static_cast<core::f32>((params.background >> 8) & 0xFFu) * (1.0f / 255.0f);
            const core::f32 bgB = static_cast<core::f32>(params.background & 0xFFu) * (1.0f / 255.0f);
            const core::f32 rest = 1.0f - accA;
            *out = packColour(accR + rest * bgR, accG + rest * bgG, accB + rest * bgB);
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
