#ifndef LPL_PROCGEN_WORLDRECIPE_INL
#define LPL_PROCGEN_WORLDRECIPE_INL

namespace lpl::procgen {

namespace detail {

constexpr bool sameName(const char *a, const char *b) noexcept
{
    if (a == nullptr || b == nullptr)
        return false;
    while (*a != '\0' && *a == *b)
    {
        ++a;
        ++b;
    }
    return *a == *b;
}

} // namespace detail

constexpr const char *caveKindName(CaveKind kind) noexcept
{
    switch (kind)
    {
    case CaveKind::Cellular: return "cellular";
    case CaveKind::Bsp: return "bsp";
    case CaveKind::Dla: return "dla";
    case CaveKind::Layered: return "layered";
    case CaveKind::Auto: return "auto";
    }
    return "cellular";
}

constexpr bool caveKindByName(const char *name, CaveKind &outKind) noexcept
{
    for (core::u32 i = 0u; i <= static_cast<core::u32>(CaveKind::Auto); ++i)
    {
        const CaveKind kind = static_cast<CaveKind>(i);
        if (detail::sameName(name, caveKindName(kind)))
        {
            outKind = kind;
            return true;
        }
    }
    return false;
}

constexpr const char *distanceMetricName(DistanceMetric metric) noexcept
{
    switch (metric)
    {
    case DistanceMetric::Euclidean: return "euclidean";
    case DistanceMetric::Manhattan: return "manhattan";
    case DistanceMetric::Chebyshev: return "chebyshev";
    }
    return "euclidean";
}

constexpr bool distanceMetricByName(const char *name, DistanceMetric &outMetric) noexcept
{
    for (core::u32 i = 0u; i <= static_cast<core::u32>(DistanceMetric::Chebyshev); ++i)
    {
        const DistanceMetric metric = static_cast<DistanceMetric>(i);
        if (detail::sameName(name, distanceMetricName(metric)))
        {
            outMetric = metric;
            return true;
        }
    }
    return false;
}

constexpr WorldRecipe parityWorldRecipe() noexcept
{
    WorldRecipe recipe{};

    recipe.seed = 1337u;
    recipe.width = 24u;
    recipe.depth = 24u;
    recipe.cellSize = 0.5f;

    recipe.terrain.seed = 1337u;
    recipe.terrain.frequency = 0.15f;
    recipe.terrain.amplitude = 12.0f;
    recipe.terrain.octaves = 4u;
    recipe.terrain.lacunarity = 2.0f;
    recipe.terrain.persistence = 0.5f;
    recipe.terrain.kind = NoiseKind::Fbm;

    recipe.heightLow = -8.0f;
    recipe.heightHigh = 16.0f;

    recipe.thermal.iterations = 8u;
    recipe.hydraulic.iterations = 12u;

    recipe.caves.width = 24u;
    recipe.caves.depth = 24u;
    recipe.caves.seed = 0xCA4Eu;
    recipe.caves.minRegionSize = 12u;

    recipe.settlement.districtSize = 8u;

    recipe.roads.iterations = 3u;
    recipe.roads.stepLength = 2u;

    recipe.gate.minPathLength = 4u;
    recipe.gate.minWalkableCells = 16u;

    recipe.scatter[0].biome = BiomeId::Grassland;
    recipe.scatter[0].density = 0.06f;
    recipe.scatter[0].halfExtent = 0.2f;
    recipe.scatter[0].tag = 1u;
    recipe.scatterCount = 1u;

    return recipe;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLDRECIPE_INL
