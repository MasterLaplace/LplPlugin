#include <lpl/procgen/WorldRecipe.hpp>

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::procgen {

namespace {

using FVec3 = math::Vec3<math::Fixed32>;

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Raises the finished field until its lowest cell reaches @p clearance.
 *
 * normalize maps min and max onto the two bounds linearly, so asking for the same span
 * translated is an exact shift, not a rescale.
 *
 * @return The shift actually applied; 0 when the ground was already high enough,
 *         which is also what every caller adds to its absolute thresholds.
 */
[[nodiscard]] core::f32 liftGround(WorldBuilder &builder, core::f32 clearance)
{
    if (clearance <= 0.0f)
        return 0.0f;

    math::Fixed32 low{};
    math::Fixed32 high{};
    if (!heightRange(builder.heightfield(), low, high))
        return 0.0f;

    const core::f32 lift = clearance - low.toFloat();
    if (lift <= 0.0f)
        return 0.0f;

    builder.normalize(low.toFloat() + lift, high.toFloat() + lift);
    return lift;
}

} // namespace

core::u32 foldWorldState(const ecs::Registry &registry) noexcept
{
    core::u32 hash = kFnv1aOffsetBasis;
    const auto fold = [&hash](core::u32 word) { hash = (hash ^ word) * kFnv1aPrime; };

    for (const auto &partition : registry.partitions())
    {
        if (!partition)
            continue;
        const bool hasPosition = partition->archetype().has(ecs::ComponentId::Position);
        const bool hasAabb = partition->archetype().has(ecs::ComponentId::AABB);
        if (!hasPosition)
            continue;

        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const core::u32 count = chunk->count();
            const auto *position = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
            const auto *aabb =
                hasAabb ? static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::AABB)) : nullptr;
            if (position == nullptr)
                continue;
            for (core::u32 i = 0u; i < count; ++i)
            {
                fold(static_cast<core::u32>(position[i].x.raw()));
                fold(static_cast<core::u32>(position[i].y.raw()));
                fold(static_cast<core::u32>(position[i].z.raw()));
                if (aabb != nullptr)
                {
                    fold(static_cast<core::u32>(aabb[i].x.raw()));
                    fold(static_cast<core::u32>(aabb[i].y.raw()));
                    fold(static_cast<core::u32>(aabb[i].z.raw()));
                }
            }
        }
    }
    return hash;
}

core::f32 applyRecipe(WorldBuilder &builder, const WorldRecipe &recipe, const ReliefBlend *relief)
{
    builder.cellSize(recipe.cellSize);
    builder.terrain(recipe.width, recipe.depth, recipe.terrain);

    if (relief != nullptr)
        builder.relief(*relief);

    if (recipe.normalizeTerrain)
        builder.normalize(recipe.heightLow, recipe.heightHigh);

    if (recipe.terraceSteps != 0u)
        builder.terraces(recipe.terraceSteps);

    if (recipe.erodeTerrain)
    {
        builder.erodeThermal(recipe.thermal);
        builder.erodeHydraulic(recipe.hydraulic);
    }

    if (recipe.carveRivers)
        builder.rivers(recipe.rivers);

    const core::f32 lift = liftGround(builder, recipe.groundClearance);

    if (recipe.partitionRegions)
        builder.regions(recipe.provinces);

    if (recipe.classifyBiomes)
    {
        BiomeParams biomes = recipe.biomes;
        ClimateParams axes = recipe.axes;
        biomes.seaLevel += lift;
        biomes.mountainHeight += lift;
        biomes.snowHeight += lift;
        axes.seaLevel += lift;

        builder.climate(recipe.climate);
        builder.climateAxes(axes);
        builder.biomes(biomes);
    }

    const core::u32 rules = recipe.scatterCount < kMaxScatterRules ? recipe.scatterCount : kMaxScatterRules;
    for (core::u32 i = 0u; i < rules; ++i)
        builder.scatter(recipe.scatter[i]);

    if (recipe.carveCaves)
    {
        CaveKind kind = recipe.caveKind;
        if (kind == CaveKind::Auto)
        {
            CaveContext context;
            context.settled = recipe.placeSettlement;
            context.layerCount = recipe.caveSystem.layers;
            const Heightfield &wet = builder.moisture();
            if (!wet.empty())
            {
                math::Fixed32 total{};
                for (core::u32 z = 0u; z < wet.depth(); ++z)
                    for (core::u32 x = 0u; x < wet.width(); ++x)
                        total = total + wet.at(x, z);
                context.wetness = total.toFloat() / static_cast<core::f32>(wet.cellCount());
            }
            kind = chooseCaveKind(context);
        }

        switch (kind)
        {
        case CaveKind::Bsp: builder.dungeon(recipe.rooms); break;
        case CaveKind::Dla: builder.dlaCaves(recipe.aggregation); break;
        case CaveKind::Layered: builder.caveSystem(recipe.caveSystem); break;
        case CaveKind::Auto:
        case CaveKind::Cellular: builder.caves(recipe.caves); break;
        }
    }

    if (recipe.placeSettlement)
    {
        SettlementParams settlement = recipe.settlement;
        settlement.minHeight += lift;
        builder.settlement(settlement);
    }

    if (recipe.raiseBuildings)
        builder.buildings(recipe.buildings);

    if (recipe.growRoads)
    {
        RoadParams roads = recipe.roads;
        roads.minHeight += lift;
        builder.roads(roads);
    }

    if (recipe.roadsideLevels != 0u && recipe.roadsidePattern[0] != '\0')
        builder.roadside(recipe.roadsidePattern, recipe.roadsideLevels);

    if (recipe.checkPlayability)
        builder.validate(recipe.gate);

    return lift;
}

WorldRecipeResult bakeWorld(ecs::Registry &registry, const WorldRecipe &recipe)
{
    WorldRecipeResult result{};

    WorldBuilder builder{recipe.seed};
    applyRecipe(builder, recipe);

    const BuiltWorldStats stats =
        recipe.materializeGround ? builder.materialize(registry) : builder.materializeProps(registry);

    result.entityCount = stats.terrainEntities + stats.propEntities;
    result.heightSignature = stats.heightSignature;
    result.biomeSignature = stats.biomeSignature;
    result.riverCells = stats.riverCells;
    result.roadCells = stats.roadCells;
    result.lakeCells = stats.lakeCells;
    result.dungeonFloor = stats.dungeonFloor;
    result.settlementPlots = stats.settlementPlots;

    if (recipe.checkPlayability)
    {
        const LevelQuality &quality = builder.lastQuality();
        result.gateReachable = quality.goalReachable ? 1u : 0u;
        result.gateVisited = quality.reachableCells;
        result.gatePathLength = quality.pathLength;
    }
    else
    {
        result.gateReachable = 1u;
    }

    result.stateSignature = foldWorldState(registry);

    const bool gateOk = !recipe.checkPlayability || builder.gatePassed();
    result.ok = (result.entityCount > 0u && result.stateSignature != kFnv1aOffsetBasis && gateOk) ? 1u : 0u;
    return result;
}

} // namespace lpl::procgen
