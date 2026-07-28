/**
 * @file Biome.cpp
 * @brief Implementation of the Whittaker-style biome classification.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Biome.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// Shorthand for writing the table below without six lines per entry.
constexpr ClimateVector vec(core::f32 t, core::f32 m, core::f32 c, core::f32 e, core::f32 d, core::f32 w)
{
    ClimateVector v{};
    v.axis[0] = math::Fixed32::fromFloat(t);
    v.axis[1] = math::Fixed32::fromFloat(m);
    v.axis[2] = math::Fixed32::fromFloat(c);
    v.axis[3] = math::Fixed32::fromFloat(e);
    v.axis[4] = math::Fixed32::fromFloat(d);
    v.axis[5] = math::Fixed32::fromFloat(w);
    return v;
}

// The Whittaker diagram as positions rather than branches.
//
// Read a row as: "this biome wants that climate (centre), and here is how much
// each axis matters to it (weight)". The weights are where the judgement lives —
// a desert is defined by being dry far more than by being anywhere in particular,
// so its moisture weight dominates; a marsh is defined by saturation AND by flat
// worn ground, so it weighs erosion too.
//
// Weirdness carries a small weight on marsh and rock only. That is deliberate and
// it is the axis' entire job: those two then appear in pockets the temperature and
// rainfall maps give no warning of, which is what makes them read as oddities
// rather than as another consequence of altitude.
//                                            T     M     C     E     D     W
constexpr BiomeProfile kProfiles[] = {
    {BiomeId::Snow,       vec(0.05f, 0.50f, 0.50f, 0.50f, 0.0f, 0.50f), vec(1.00f, 0.20f, 0.00f, 0.00f, 0.0f, 0.00f)},
    {BiomeId::Tundra,     vec(0.20f, 0.25f, 0.50f, 0.50f, 0.0f, 0.50f), vec(1.00f, 0.80f, 0.00f, 0.00f, 0.0f, 0.00f)},
    {BiomeId::Taiga,      vec(0.25f, 0.75f, 0.50f, 0.50f, 0.0f, 0.50f), vec(1.00f, 0.80f, 0.00f, 0.00f, 0.0f, 0.00f)},
    {BiomeId::Rock,       vec(0.35f, 0.15f, 0.50f, 0.15f, 0.0f, 0.80f), vec(0.40f, 0.60f, 0.00f, 0.80f, 0.0f, 0.30f)},
    {BiomeId::Desert,     vec(0.85f, 0.12f, 0.85f, 0.50f, 0.0f, 0.50f), vec(1.00f, 1.00f, 0.40f, 0.00f, 0.0f, 0.00f)},
    {BiomeId::Savanna,    vec(0.80f, 0.45f, 0.60f, 0.50f, 0.0f, 0.50f), vec(1.00f, 0.90f, 0.20f, 0.00f, 0.0f, 0.00f)},
    // Grassland's moisture weight is deliberately the lowest of the temperate
    // band. It is the corridor biome: without a profile that spans a WIDE
    // moisture range at middling temperature, a dry cell warming from tundra to
    // desert has nothing to pass through, and the map grows a seam where a
    // steppe belongs.
    {BiomeId::Grassland,  vec(0.50f, 0.40f, 0.50f, 0.50f, 0.0f, 0.50f), vec(0.90f, 0.55f, 0.00f, 0.00f, 0.0f, 0.00f)},
    {BiomeId::Forest,     vec(0.50f, 0.75f, 0.50f, 0.50f, 0.0f, 0.50f), vec(0.90f, 1.00f, 0.00f, 0.00f, 0.0f, 0.00f)},
    {BiomeId::Rainforest, vec(0.90f, 0.88f, 0.25f, 0.50f, 0.0f, 0.50f), vec(1.00f, 1.00f, 0.30f, 0.00f, 0.0f, 0.00f)},
    {BiomeId::Marsh,      vec(0.55f, 0.95f, 0.30f, 0.85f, 0.0f, 0.75f), vec(0.30f, 1.20f, 0.00f, 0.60f, 0.0f, 0.35f)},
};

constexpr core::u32 kProfileCount = static_cast<core::u32>(sizeof(kProfiles) / sizeof(kProfiles[0]));

} // namespace

const BiomeProfile *biomeProfiles(core::u32 &outCount) noexcept
{
    outCount = kProfileCount;
    return kProfiles;
}

BiomeId nearestBiomeProfile(const ClimateVector &climate, math::Fixed32 &outDistance) noexcept
{
    BiomeId best = BiomeId::Grassland;
    math::Fixed32 bestDistance = math::Fixed32::zero();
    bool haveBest = false;

    for (core::u32 p = 0u; p < kProfileCount; ++p)
    {
        const BiomeProfile &profile = kProfiles[p];
        math::Fixed32 distance = math::Fixed32::zero();
        for (core::u32 a = 0u; a < kClimateAxisCount; ++a)
        {
            const math::Fixed32 w = profile.weight.axis[a];
            if (w.raw() == 0)
                continue; // This biome has no opinion on this axis.
            const math::Fixed32 delta = climate.axis[a] - profile.center.axis[a];
            distance = distance + w * delta * delta;
        }

        // Strictly less, so a tie goes to the earlier row. The table's order is
        // therefore part of the result and must not be shuffled casually — which
        // is exactly why it is one static table and not a runtime registry.
        if (!haveBest || distance < bestDistance)
        {
            best = profile.id;
            bestDistance = distance;
            haveBest = true;
        }
    }

    outDistance = bestDistance;
    return best;
}

BiomeMap classifyBiomes(const Heightfield &field, const ClimateField &climate, const BiomeParams &params,
                        const Grid<core::u8> *lakes)
{
    if (field.empty() || climate.width() != field.width() || climate.depth() != field.depth())
        return BiomeMap{};

    BiomeMap map{field.width(), field.depth(), BiomeId::Grassland};

    const math::Fixed32 seaLevel = math::Fixed32::fromFloat(params.seaLevel);
    const math::Fixed32 beachTop = seaLevel + math::Fixed32::fromFloat(params.beachHeight);
    const math::Fixed32 mountain = math::Fixed32::fromFloat(params.mountainHeight);
    const math::Fixed32 snow = math::Fixed32::fromFloat(params.snowHeight);
    const math::Fixed32 snowline = math::Fixed32::fromFloat(params.snowlineWarmth);

    const bool haveLakes = lakes != nullptr && lakes->width() == field.width() && lakes->depth() == field.depth();

    for (core::u32 z = 0u; z < field.depth(); ++z)
    {
        for (core::u32 x = 0u; x < field.width(); ++x)
        {
            const math::Fixed32 height = field.at(x, z);

            // ── Altitude and hydrology decide first ─────────────────────────
            //
            // Not out of laziness: these three are facts the climate axes cannot
            // represent. Sea level is a hard boundary, a shore is a metre-wide
            // band, and a lake is a basin the flow routing proved had no outlet.
            // A nearest-profile lookup asked to reproduce them would need a
            // biome centred on "below sea level", which is a height, not a
            // climate.
            if (height <= seaLevel)
            {
                map.at(x, z) = BiomeId::Ocean;
                continue;
            }
            if (height <= beachTop)
            {
                map.at(x, z) = BiomeId::Beach;
                continue;
            }
            if (height >= snow)
            {
                map.at(x, z) = BiomeId::Snow;
                continue;
            }
            if (height >= mountain)
            {
                // Cold peaks hold snow at a lower altitude than warm ones.
                const math::Fixed32 warmth = climate[ClimateAxis::Temperature].at(x, z);
                map.at(x, z) = warmth < snowline ? BiomeId::Snow : BiomeId::Rock;
                continue;
            }
            if (haveLakes && lakes->at(x, z) != 0u)
            {
                // After the altitude tests, not before: a basin below sea level
                // is ocean, and calling it a lake would put fresh water under
                // the sea.
                map.at(x, z) = BiomeId::Lake;
                continue;
            }

            // ── Everything else: nearest profile in climate space ───────────
            math::Fixed32 distance{};
            map.at(x, z) = nearestBiomeProfile(climate.at(x, z), distance);
        }
    }
    return map;
}

const char *biomeName(BiomeId biome) noexcept
{
    switch (biome)
    {
    case BiomeId::Ocean: return "ocean";
    case BiomeId::Beach: return "beach";
    case BiomeId::Snow: return "snow";
    case BiomeId::Tundra: return "tundra";
    case BiomeId::Taiga: return "taiga";
    case BiomeId::Rock: return "rock";
    case BiomeId::Desert: return "desert";
    case BiomeId::Savanna: return "savanna";
    case BiomeId::Grassland: return "grassland";
    case BiomeId::Forest: return "forest";
    case BiomeId::Rainforest: return "rainforest";
    case BiomeId::Marsh: return "marsh";
    case BiomeId::Lake: return "lake";
    case BiomeId::Count: break;
    }
    return "unknown";
}

BiomeId biomeIdByName(const char *name) noexcept
{
    if (name == nullptr)
        return BiomeId::Count;
    // Compared against biomeName's own output, so the two can never disagree
    // about a spelling — there is one table, read in both directions.
    for (core::u32 i = 0u; i < static_cast<core::u32>(BiomeId::Count); ++i)
    {
        const char *candidate = biomeName(static_cast<BiomeId>(i));
        core::u32 c = 0u;
        while (candidate[c] != '\0' && name[c] == candidate[c])
            ++c;
        if (candidate[c] == '\0' && name[c] == '\0')
            return static_cast<BiomeId>(i);
    }
    return BiomeId::Count;
}

void countBiomes(const BiomeMap &map, core::u32 *outCounts)
{
    if (outCounts == nullptr)
        return;
    for (core::u32 i = 0u; i < static_cast<core::u32>(BiomeId::Count); ++i)
        outCounts[i] = 0u;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
    {
        const core::u32 index = static_cast<core::u32>(map[i]);
        if (index < static_cast<core::u32>(BiomeId::Count))
            ++outCounts[index];
    }
}

core::u32 foldBiomeMap(const BiomeMap &map)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        hash = (hash ^ static_cast<core::u32>(map[i])) * kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
