/**
 * @file Climate.cpp
 * @brief Implementation of the six-axis climate field.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Climate.hpp>
#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// Histogram resolution for @ref rankNormalize. A power of two so the bucket
/// index is a shift rather than a division, and large enough that two values a
/// thousandth apart still rank apart.
constexpr core::u32 kRankBuckets = 1024u;

/**
 * @brief A 0..1 warmth value: 1 at the equator, 0 at the poles.
 *
 * The map's Z axis stands in for latitude, so a single world contains a climate
 * gradient instead of one uniform climate. This used to live inside the biome
 * classification, where nothing else could see it; the temperature axis is the
 * same quantity, so it is defined once and read from here.
 *
 * @param z        Row.
 * @param depth    Grid depth.
 * @param coldEdge Share of the map at each edge that is fully polar.
 */
math::Fixed32 latitudeWarmth(core::u32 z, core::u32 depth, math::Fixed32 coldEdge)
{
    if (depth <= 1u)
        return math::Fixed32::one();

    // Distance from the equator (mid-map), normalised to [0, 1].
    const math::Fixed32 position = math::Fixed32::fromInt(static_cast<core::i32>(z)) /
                                   math::Fixed32::fromInt(static_cast<core::i32>(depth - 1u));
    const math::Fixed32 fromEquator = (position - math::Fixed32::half()).abs() * math::Fixed32::fromInt(2);

    // Beyond the polar edge, warmth is 0; inside, it falls off quadratically.
    //
    // Quadratically, not linearly: insolation goes as the cosine of the latitude,
    // so the tropics are a broad band and the drop is concentrated toward the
    // poles. A linear ramp instead makes the equator a single warm line — measured
    // on a 128-row map it left two thirds of the world classified as polar, which
    // is not a climate, it is an ice age.
    const math::Fixed32 temperate = math::Fixed32::one() - coldEdge;
    if (temperate.raw() <= 0)
        return fromEquator > math::Fixed32::zero() ? math::Fixed32::zero() : math::Fixed32::one();
    if (fromEquator > temperate)
        return math::Fixed32::zero();

    const math::Fixed32 normalized = fromEquator / temperate;
    return math::Fixed32::one() - normalized * normalized;
}

} // namespace

void normalizeUnit(Heightfield &field)
{
    if (field.empty())
        return;

    math::Fixed32 lowest{};
    math::Fixed32 highest{};
    if (!heightRange(field, lowest, highest))
        return;

    const math::Fixed32 span = highest - lowest;
    if (span.raw() == 0)
    {
        // A constant axis carries no information; putting it at the midpoint
        // means it neither attracts nor repels any profile.
        field.fill(math::Fixed32::half());
        return;
    }
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
        field[i] = (field[i] - lowest) / span;
}

void rankNormalize(Heightfield &field)
{
    if (field.empty())
        return;

    math::Fixed32 lowest{};
    math::Fixed32 highest{};
    if (!heightRange(field, lowest, highest))
        return;

    const math::Fixed32 span = highest - lowest;
    if (span.raw() == 0)
    {
        field.fill(math::Fixed32::half());
        return;
    }

    // Bucket every value, then walk the histogram once to turn counts into a
    // cumulative distribution. A cell's new value is the share of the map at or
    // below it — which is what "rank" means, and which is flat by construction
    // whatever the input's shape was.
    lpl::pmr::vector<core::u32> histogram(kRankBuckets, 0u);
    const core::u32 cells = field.cellCount();

    const auto bucketOf = [&](math::Fixed32 value) -> core::u32 {
        const math::Fixed32 t = (value - lowest) / span;
        core::i64 index = static_cast<core::i64>(t.raw()) * static_cast<core::i64>(kRankBuckets - 1u);
        index >>= 16;
        if (index < 0)
            index = 0;
        if (index > static_cast<core::i64>(kRankBuckets - 1u))
            index = static_cast<core::i64>(kRankBuckets - 1u);
        return static_cast<core::u32>(index);
    };

    for (core::u32 i = 0u; i < cells; ++i)
        ++histogram[bucketOf(field[i])];

    // Cumulative counts, taken at the *middle* of each bucket's run: a value
    // sitting in the busiest bucket should not be reported as being above every
    // other value in it.
    lpl::pmr::vector<math::Fixed32> rank(kRankBuckets, math::Fixed32::zero());
    core::u32 below = 0u;
    for (core::u32 b = 0u; b < kRankBuckets; ++b)
    {
        const core::u32 here = histogram[b];
        const core::u32 midpoint = below + here / 2u;
        rank[b] = math::Fixed32::fromInt(static_cast<core::i32>(midpoint)) /
                  math::Fixed32::fromInt(static_cast<core::i32>(cells));
        below += here;
    }

    for (core::u32 i = 0u; i < cells; ++i)
        field[i] = rank[bucketOf(field[i])];
}

ClimateField computeClimate(const Heightfield &field, const Heightfield &moisture, const DrainageNetwork &network,
                            const ClimateParams &params)
{
    ClimateField climate{};
    if (field.empty() || moisture.width() != field.width() || moisture.depth() != field.depth())
        return climate;

    const core::u32 width = field.width();
    const core::u32 depth = field.depth();
    for (core::u32 a = 0u; a < kClimateAxisCount; ++a)
        climate.axes[a] = Heightfield{width, depth, math::Fixed32::zero()};

    math::Fixed32 lowest{};
    math::Fixed32 highest{};
    (void) heightRange(field, lowest, highest);
    const math::Fixed32 span = highest - lowest;

    const math::Fixed32 coldEdge = math::Fixed32::fromFloat(params.coldLatitude);
    const math::Fixed32 lapse = math::Fixed32::fromFloat(params.lapseRate);
    const math::Fixed32 seaLevel = math::Fixed32::fromFloat(params.seaLevel);

    // ── Temperature: latitude, less what altitude takes away ────────────────
    Heightfield &temperature = climate[ClimateAxis::Temperature];
    for (core::u32 z = 0u; z < depth; ++z)
    {
        const math::Fixed32 latitudeTerm = latitudeWarmth(z, depth, coldEdge);
        for (core::u32 x = 0u; x < width; ++x)
        {
            math::Fixed32 warmth = latitudeTerm;
            if (span.raw() != 0)
                warmth = warmth - lapse * ((field.at(x, z) - lowest) / span);
            if (warmth < math::Fixed32::zero())
                warmth = math::Fixed32::zero();
            if (warmth > math::Fixed32::one())
                warmth = math::Fixed32::one();
            temperature.at(x, z) = warmth;
        }
    }

    // ── Moisture: already built, already in [0, 1] ──────────────────────────
    climate[ClimateAxis::Moisture] = moisture;

    // ── Continentalness: how far the sea is, saturating at coastReach ───────
    const core::u32 longAxis = width > depth ? width : depth;
    const math::Fixed32 axisLength = math::Fixed32::fromInt(static_cast<core::i32>(longAxis));
    const math::Fixed32 coastRange = math::Fixed32::fromFloat(params.coastReach) * axisLength;
    const Grid<core::u32> seaDistance = distanceToSea(field, seaLevel);

    Heightfield &continental = climate[ClimateAxis::Continentalness];
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
    {
        if (seaDistance[i] == kUnreachedFromSea || coastRange.raw() <= 0)
        {
            // No sea anywhere on this map: everything is interior. Saying
            // "coastal" instead would put beaches in the middle of a continent.
            continental[i] = math::Fixed32::one();
            continue;
        }
        math::Fixed32 t = math::Fixed32::fromInt(static_cast<core::i32>(seaDistance[i])) / coastRange;
        if (t > math::Fixed32::one())
            t = math::Fixed32::one();
        continental[i] = t;
    }

    // ── Erosion: worn ground scores high, jagged ground low ─────────────────
    //
    // Built from the local slope and then INVERTED, because the axis names the
    // result of erosion rather than its potential: a floodplain is what erosion
    // leaves behind, a spire is what it has not reached yet.
    Heightfield &erosion = climate[ClimateAxis::Erosion];
    for (core::u32 z = 0u; z < depth; ++z)
        for (core::u32 x = 0u; x < width; ++x)
            erosion.at(x, z) = slopeAt(field, x, z);

    // Rank, not min-max: slope is heavy-tailed (a few cliffs, mostly gentle
    // ground), so scaling by the maximum would report almost the entire map as
    // unworn and the axis would separate nothing.
    rankNormalize(erosion);
    for (core::u32 i = 0u; i < erosion.cellCount(); ++i)
        erosion[i] = math::Fixed32::one() - erosion[i];

    // ── Depth: which layer this field describes ─────────────────────────────
    //
    // Constant across a surface field, and that is the point: the same profile
    // table then classifies a cave layer by handing it a different depth, instead
    // of needing a second table for underground biomes.
    math::Fixed32 depthValue = math::Fixed32::fromFloat(params.surfaceDepth);
    if (depthValue < math::Fixed32::zero())
        depthValue = math::Fixed32::zero();
    if (depthValue > math::Fixed32::one())
        depthValue = math::Fixed32::one();
    climate[ClimateAxis::Depth].fill(depthValue);

    // ── Weirdness: an independent field, deliberately correlated with nothing ─
    //
    // Its whole job is to be uncorrelated with the terrain. Rare biomes keyed to
    // it therefore appear where the land gives no warning, which is what makes
    // them read as anomalies rather than as a consequence of altitude.
    const math::Fixed32 weirdFrequency = math::Fixed32::fromFloat(params.weirdnessBelts) / axisLength;
    Heightfield &weird = climate[ClimateAxis::Weirdness];
    for (core::u32 z = 0u; z < depth; ++z)
        for (core::u32 x = 0u; x < width; ++x)
        {
            const math::Fixed32 n =
                ValueNoise2D::fbm(math::Fixed32::fromInt(static_cast<core::i32>(x)) * weirdFrequency,
                                  math::Fixed32::fromInt(static_cast<core::i32>(z)) * weirdFrequency,
                                  params.weirdnessOctaves, params.weirdnessSeed);
            weird.at(x, z) = (n + math::Fixed32::one()) * math::Fixed32::half();
        }

    (void) network; // The drainage term folds into the slope already; kept in the
                    // signature because the erosion axis is the natural place to
                    // add accumulation weighting, and callers should not have to
                    // change when it does.
    return climate;
}

core::u32 foldClimateField(const ClimateField &climate)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 a = 0u; a < kClimateAxisCount; ++a)
    {
        const Heightfield &axis = climate.axes[a];
        for (core::u32 i = 0u; i < axis.cellCount(); ++i)
        {
            const core::u32 raw = static_cast<core::u32>(axis[i].raw());
            hash = (hash ^ (raw & 0xFFu)) * kFnv1aPrime;
            hash = (hash ^ ((raw >> 8) & 0xFFu)) * kFnv1aPrime;
            hash = (hash ^ ((raw >> 16) & 0xFFu)) * kFnv1aPrime;
            hash = (hash ^ ((raw >> 24) & 0xFFu)) * kFnv1aPrime;
        }
    }
    return hash;
}

} // namespace lpl::procgen
