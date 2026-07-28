/**
 * @file Heightfield.cpp
 * @brief Implementation of the terrain grid and its shaping passes.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Heightfield.hpp>

#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

} // namespace

math::Fixed32 sampleNoiseAt(core::i32 worldX, core::i32 worldZ, const NoiseParams &params)
{
    const math::Fixed32 frequency = math::Fixed32::fromFloat(params.frequency);
    const math::Fixed32 amplitude = math::Fixed32::fromFloat(params.amplitude);
    const math::Fixed32 lacunarity = math::Fixed32::fromFloat(params.lacunarity);
    const math::Fixed32 persistence = math::Fixed32::fromFloat(params.persistence);
    const math::Fixed32 base = math::Fixed32::fromFloat(params.baseHeight);

    math::Fixed32 nx = math::Fixed32::fromInt(worldX);
    math::Fixed32 nz = math::Fixed32::fromInt(worldZ);
    // Warp in cell space, before the frequency scaling, so warpStrength reads in
    // cells rather than in whatever units the frequency happens to imply.
    ValueNoise2D::warp(nx, nz, params.seed, math::Fixed32::fromFloat(params.warpStrength), frequency);
    nx = nx * frequency;
    nz = nz * frequency;

    switch (params.kind)
    {
    case NoiseKind::Ridged:
        // Ridged and billow both return [0, 1); recentre them so a layer's mean
        // stays near zero and layering one over another does not drift upward.
        return base + (ValueNoise2D::ridged(nx, nz, params.octaves, params.seed, lacunarity, persistence) -
                       math::Fixed32::half()) *
                          math::Fixed32::fromInt(2) * amplitude;
    case NoiseKind::Billow:
        return base + (ValueNoise2D::billow(nx, nz, params.octaves, params.seed, lacunarity, persistence) -
                       math::Fixed32::half()) *
                          math::Fixed32::fromInt(2) * amplitude;
    case NoiseKind::Fbm: break;
    }
    return base + ValueNoise2D::fbm(nx, nz, params.octaves, params.seed, lacunarity, persistence) * amplitude;
}

Heightfield generateNoiseHeightfield(core::u32 width, core::u32 depth, const NoiseParams &params)
{
    if (width == 0u || depth == 0u)
        return Heightfield{};

    Heightfield field{width, depth, math::Fixed32::zero()};

    for (core::u32 z = 0u; z < depth; ++z)
        for (core::u32 x = 0u; x < width; ++x)
            field.at(x, z) = sampleNoiseAt(static_cast<core::i32>(x), static_cast<core::i32>(z), params);

    return field;
}

void addNoiseLayer(Heightfield &field, const NoiseParams &params)
{
    if (field.empty())
        return;

    for (core::u32 z = 0u; z < field.depth(); ++z)
        for (core::u32 x = 0u; x < field.width(); ++x)
            field.at(x, z) =
                field.at(x, z) + sampleNoiseAt(static_cast<core::i32>(x), static_cast<core::i32>(z), params);
}

bool heightRange(const Heightfield &field, math::Fixed32 &outMin, math::Fixed32 &outMax)
{
    if (field.empty())
        return false;

    math::Fixed32 low = field[0];
    math::Fixed32 high = field[0];
    for (core::u32 i = 1u; i < field.cellCount(); ++i)
    {
        if (field[i] < low)
            low = field[i];
        if (field[i] > high)
            high = field[i];
    }
    outMin = low;
    outMax = high;
    return true;
}

void normalizeHeights(Heightfield &field, math::Fixed32 low, math::Fixed32 high)
{
    math::Fixed32 currentMin{};
    math::Fixed32 currentMax{};
    if (!heightRange(field, currentMin, currentMax))
        return;

    const math::Fixed32 span = currentMax - currentMin;
    if (span.raw() == 0)
    {
        // A flat field has no range to stretch; put it at the target floor
        // rather than dividing by zero.
        field.fill(low);
        return;
    }

    const math::Fixed32 targetSpan = high - low;
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
    {
        const math::Fixed32 normalized = (field[i] - currentMin) / span;
        field[i] = low + normalized * targetSpan;
    }
}

void clampToSeaLevel(Heightfield &field, math::Fixed32 level)
{
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
        if (field[i] < level)
            field[i] = level;
}

void terrace(Heightfield &field, core::u32 steps)
{
    if (steps < 2u || field.empty())
        return;

    math::Fixed32 low{};
    math::Fixed32 high{};
    if (!heightRange(field, low, high))
        return;

    const math::Fixed32 span = high - low;
    if (span.raw() == 0)
        return;

    // `steps` bands means `steps` distinct heights, so the top band is steps - 1
    // and the mapping back divides by steps - 1. Dividing by `steps` instead
    // leaves an extra level that only the single highest cell can reach — a
    // one-cell plateau perched above the terrace below it.
    const math::Fixed32 bandCount = math::Fixed32::fromInt(static_cast<core::i32>(steps));
    const math::Fixed32 topBand = math::Fixed32::fromInt(static_cast<core::i32>(steps - 1u));
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
    {
        const math::Fixed32 normalized = (field[i] - low) / span;
        core::i32 band = (normalized * bandCount).toInt();
        if (band < 0)
            band = 0;
        if (band > static_cast<core::i32>(steps - 1u))
            band = static_cast<core::i32>(steps - 1u);
        field[i] = low + span * (math::Fixed32::fromInt(band) / topBand);
    }
}

void smoothHeights(Heightfield &field, core::u32 iterations)
{
    if (field.empty() || iterations == 0u)
        return;

    // Double-buffered: smoothing in place would feed already-smoothed cells back
    // into the average, which is a different (and direction-dependent) filter.
    Heightfield scratch{field.width(), field.depth(), math::Fixed32::zero()};
    const math::Fixed32 nine = math::Fixed32::fromInt(9);

    for (core::u32 pass = 0u; pass < iterations; ++pass)
    {
        for (core::u32 z = 0u; z < field.depth(); ++z)
        {
            for (core::u32 x = 0u; x < field.width(); ++x)
            {
                math::Fixed32 sum = field.at(x, z);
                for (core::u32 n = 0u; n < 8u; ++n)
                    sum = sum + field.clamped(static_cast<core::i32>(x) + kNeighbor8X[n],
                                              static_cast<core::i32>(z) + kNeighbor8Z[n]);
                scratch.at(x, z) = sum / nine;
            }
        }
        for (core::u32 i = 0u; i < field.cellCount(); ++i)
            field[i] = scratch[i];
    }
}

math::Fixed32 slopeAt(const Heightfield &field, core::u32 x, core::u32 z)
{
    if (field.empty())
        return math::Fixed32::zero();

    const math::Fixed32 here = field.at(x, z);
    math::Fixed32 steepest = math::Fixed32::zero();
    for (core::u32 n = 0u; n < 4u; ++n)
    {
        const math::Fixed32 drop =
            (here - field.clamped(static_cast<core::i32>(x) + kNeighbor4X[n],
                                  static_cast<core::i32>(z) + kNeighbor4Z[n]))
                .abs();
        if (drop > steepest)
            steepest = drop;
    }
    return steepest;
}

core::u32 foldHeightfield(const Heightfield &field)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
        hash = (hash ^ static_cast<core::u32>(field[i].raw())) * kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
