#include <lpl/procgen/Landmark.hpp>

#include <lpl/math/Random.hpp>
#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

namespace {

/**
 * @brief A salt per kind, so the two lattices are independent rather than the same one
 *        filtered twice — otherwise every village would sit on a cave.
 */
[[nodiscard]] core::u32 kindSalt(LandmarkKind kind) noexcept
{
    switch (kind)
    {
    case LandmarkKind::CaveMouth: return 0xCA5E3011u;
    case LandmarkKind::Settlement: return 0x51EE71A9u;
    default: return 0u;
    }
}

} // namespace

bool landmarkAt(const ChunkParams &params, const LandmarkParams &landmarks, LandmarkKind kind, core::f32 seaLevel,
                core::i32 landmarkX, core::i32 landmarkZ, LandmarkSite &out)
{
    if (params.size == 0u || landmarks.cellSpan == 0u || landmarks.oneIn == 0u)
        return false;

    const core::u32 mixed = ValueNoise2D::hash2(landmarkX, landmarkZ, params.worldSeed ^ kindSalt(kind));
    math::Random draw = math::deriveStream(mixed, 0x1A4D3Fu);

    if (draw.below(landmarks.oneIn) != 0u)
        return false;

    const core::i32 span = static_cast<core::i32>(landmarks.cellSpan);
    const core::i32 radius = static_cast<core::i32>(landmarks.radius);
    const core::i32 free = span - 2 * radius;
    core::i32 offsetX = radius;
    core::i32 offsetZ = radius;
    if (free > 0)
    {
        offsetX += static_cast<core::i32>(draw.below(static_cast<core::u32>(free)));
        offsetZ += static_cast<core::i32>(draw.below(static_cast<core::u32>(free)));
    }
    else
    {
        offsetX = span / 2;
        offsetZ = span / 2;
    }

    const core::i32 cellX = landmarkX * span + offsetX;
    const core::i32 cellZ = landmarkZ * span + offsetZ;

    const core::f32 here = sampleWorldHeight(params, cellX, cellZ).toFloat();
    if (here < seaLevel + landmarks.clearanceAboveSea)
        return false;
    if (here > landmarks.maxHeight)
        return false;

    core::f32 lowest = here;
    core::f32 highest = here;
    core::u32 facing = 0u;
    core::f32 bestDrop = 0.0f;
    for (core::u32 n = 0u; n < 8u; ++n)
    {
        const core::f32 neighbour =
            sampleWorldHeight(params, cellX + kNeighbor8X[n] * radius, cellZ + kNeighbor8Z[n] * radius).toFloat();
        if (neighbour < lowest)
            lowest = neighbour;
        if (neighbour > highest)
            highest = neighbour;
        const core::f32 drop = here - neighbour;
        if (drop > bestDrop)
        {
            bestDrop = drop;
            facing = n;
        }
    }

    const core::f32 relief = highest - lowest;
    if (relief < landmarks.minRelief || relief > landmarks.maxRelief)
        return false;

    out.cellX = cellX;
    out.cellZ = cellZ;
    out.height = here;
    out.relief = relief;
    out.seed = draw.state();
    out.facing = facing;
    out.radius = landmarks.radius;
    out.kind = kind;
    return true;
}

core::f32 calibrateLandmarkRelief(const ChunkParams &params, const LandmarkParams &landmarks, LandmarkKind kind,
                                  core::f32 seaLevel, core::f32 quantile)
{
    if (params.size == 0u || landmarks.cellSpan == 0u || quantile <= 0.0f)
        return 0.0f;

    LandmarkParams probe = landmarks;
    probe.minRelief = 0.0f;
    probe.maxRelief = 1.0e9f;

    constexpr core::i32 kWindow = 10;
    lpl::pmr::vector<core::f32> reliefs;
    for (core::i32 lz = -kWindow; lz <= kWindow; ++lz)
        for (core::i32 lx = -kWindow; lx <= kWindow; ++lx)
        {
            LandmarkSite site;
            if (!landmarkAt(params, probe, kind, seaLevel, lx, lz, site))
                continue;
            reliefs.push_back(site.relief);
        }
    if (reliefs.empty())
        return 0.0f;

    for (core::u32 i = 1u; i < reliefs.size(); ++i)
    {
        const core::f32 key = reliefs[i];
        core::u32 j = i;
        while (j > 0u && reliefs[j - 1u] > key)
        {
            reliefs[j] = reliefs[j - 1u];
            --j;
        }
        reliefs[j] = key;
    }

    const core::f32 clamped = quantile > 1.0f ? 1.0f : quantile;
    const core::u32 last = static_cast<core::u32>(reliefs.size()) - 1u;
    core::u32 index = static_cast<core::u32>(clamped * static_cast<core::f32>(last));
    if (index > last)
        index = last;
    return reliefs[index];
}

bool chunkOwnsLandmark(const ChunkParams &params, const LandmarkSite &site, ChunkCoord coord)
{
    if (params.size == 0u)
        return false;
    const core::i32 cells = static_cast<core::i32>(params.size);
    const core::i32 originX = coord.x * cells;
    const core::i32 originZ = coord.z * cells;
    return site.cellX >= originX && site.cellX < originX + cells && site.cellZ >= originZ &&
           site.cellZ < originZ + cells;
}

VillagePlan planVillage(const ChunkParams &params, const LandmarkSite &site)
{
    VillagePlan plan;
    plan.site = site;
    if (params.size == 0u || site.radius == 0u)
        return plan;

    const core::u32 side = site.radius * 2u + 1u;
    plan.side = side;
    plan.originX = site.cellX - static_cast<core::i32>(site.radius);
    plan.originZ = site.cellZ - static_cast<core::i32>(site.radius);
    plan.padHeight = site.height;

    Heightfield pad{side, side, math::Fixed32::fromFloat(site.height)};

    SettlementParams settlement;
    settlement.width = side;
    settlement.depth = side;
    settlement.seed = site.seed;
    settlement.districtSize = side / 2u == 0u ? 1u : side / 2u;
    settlement.roadWidth = 1u;
    settlement.plazaRadius = 1u;
    settlement.minPlot = 2u;
    settlement.maxPlot = 4u;
    settlement.plotDensity = 0.7f;
    settlement.maxSlope = 1.5f;
    settlement.minHeight = site.height - 1.0f;

    plan.map = generateSettlementOnTerrain(settlement, pad, &plan.plots, nullptr);
    return plan;
}

} // namespace lpl::procgen
