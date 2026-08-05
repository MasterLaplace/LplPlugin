/**
 * @file WorldRecipe.cpp
 * @brief Implementation of recipe baking and the authoritative state fold.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

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

    // normalize maps min and max onto the two bounds linearly, so asking for the same
    // span translated is an exact shift, not a rescale.
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
        // Archetype membership, never a null pointer test: readComponent returns
        // a valid pointer for every component id a chunk was built with, so a
        // null check would answer a different question than "does this partition
        // store positions".
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

core::f32 applyRecipe(WorldBuilder &builder, const WorldRecipe &recipe)
{
    // The pass order lives here, in one place, rather than at every call site.
    // It is not a style preference: two callers ordering erosion and rivers
    // differently would produce two different worlds from the same recipe, and
    // the parity gate would be comparing worlds that were never meant to match.
    //
    // It is a free function rather than a step inside bakeWorld because a caller
    // may want the GRIDS without the entities — a viewer draws a heightfield, it
    // does not need one entity per ground cell. Before this existed the in-kernel
    // viewer built its own pipeline by hand, which is exactly how a project ends
    // up with two generators and a gate that exercises the one nothing else runs.
    builder.cellSize(recipe.cellSize);
    builder.terrain(recipe.width, recipe.depth, recipe.terrain);

    if (recipe.normalizeTerrain)
        builder.normalize(recipe.heightLow, recipe.heightHigh);

    // Terracing before erosion, because erosion is what softens the steps back into
    // something that looks cut rather than stamped. The other way round gives sharp
    // benches on a smooth hill, which reads as a rendering artefact.
    if (recipe.terraceSteps != 0u)
        builder.terraces(recipe.terraceSteps);

    if (recipe.erodeTerrain)
    {
        builder.erodeThermal(recipe.thermal);
        builder.erodeHydraulic(recipe.hydraulic);
    }

    if (recipe.carveRivers)
        builder.rivers(recipe.rivers);

    // The world is put in the frame the physics expects HERE — after erosion and the
    // rivers, because the lowest cell is not knowable before them, and before a single
    // entity exists, because moving the ground under things already standing on it
    // means chasing a list that only ever grows. See WorldRecipe::groundClearance.
    const core::f32 lift = liftGround(builder, recipe.groundClearance);

    // Provinces before the climate block: the districting is a partition of the
    // SURFACE, so it belongs with the terrain that is now final, and a settlement laid
    // out later can be anchored on it.
    if (recipe.partitionRegions)
        builder.regions(recipe.provinces);

    if (recipe.classifyBiomes)
    {
        BiomeParams biomes = recipe.biomes;
        ClimateParams axes = recipe.axes;
        // Every absolute threshold travels with the lift, or the classifier reads the
        // shifted field against thresholds tuned for the old one and calls a plain a
        // summit.
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

    // One underground, chosen by the recipe. Four generators existed and only the
    // cellular one could be named, so the other three were reachable exclusively by
    // hand-written builder calls — a world nothing could save, bake or replay.
    if (recipe.carveCaves)
    {
        switch (recipe.caveKind)
        {
        case CaveKind::Bsp: builder.dungeon(recipe.rooms); break;
        case CaveKind::Dla: builder.dlaCaves(recipe.aggregation); break;
        case CaveKind::Layered: builder.caveSystem(recipe.caveSystem); break;
        case CaveKind::Cellular: builder.caves(recipe.caves); break;
        }
    }

    if (recipe.placeSettlement)
    {
        SettlementParams settlement = recipe.settlement;
        settlement.minHeight += lift;
        builder.settlement(settlement);
    }

    // The grammar needs the plots the settlement laid out, so it cannot run before
    // one exists. Raising a town is what turns a painted footprint into a silhouette.
    if (recipe.raiseBuildings)
        builder.buildings(recipe.buildings);

    // After the settlement, never before: the road field is anchored on the
    // districts the town laid out, so growing first would steer the network by
    // a town that does not exist yet.
    if (recipe.growRoads)
    {
        RoadParams roads = recipe.roads;
        roads.minHeight += lift;
        builder.roads(roads);
    }

    // And the verges after the roads they decorate.
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
        // A recipe that declines the check is trivially "not failing" it, so the
        // ok flag below stays a statement about the world, not about the gate.
        result.gateReachable = 1u;
    }

    result.stateSignature = foldWorldState(registry);

    const bool gateOk = !recipe.checkPlayability || builder.gatePassed();
    result.ok = (result.entityCount > 0u && result.stateSignature != kFnv1aOffsetBasis && gateOk) ? 1u : 0u;
    return result;
}

} // namespace lpl::procgen
