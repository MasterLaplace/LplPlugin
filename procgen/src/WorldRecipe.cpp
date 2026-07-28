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

WorldRecipeResult bakeWorld(ecs::Registry &registry, const WorldRecipe &recipe)
{
    WorldRecipeResult result{};

    // The pass order lives here, in one place, rather than at every call site.
    // It is not a style preference: two callers ordering erosion and rivers
    // differently would produce two different worlds from the same recipe, and
    // the parity gate would be comparing worlds that were never meant to match.
    WorldBuilder builder{recipe.seed};
    builder.cellSize(recipe.cellSize);
    builder.terrain(recipe.width, recipe.depth, recipe.terrain);

    if (recipe.normalizeTerrain)
        builder.normalize(recipe.heightLow, recipe.heightHigh);

    if (recipe.erodeTerrain)
    {
        builder.erodeThermal(recipe.thermal);
        builder.erodeHydraulic(recipe.hydraulic);
    }

    if (recipe.carveRivers)
        builder.rivers(recipe.rivers);

    if (recipe.classifyBiomes)
    {
        builder.climate(recipe.climate);
        builder.climateAxes(recipe.axes);
        builder.biomes(recipe.biomes);
    }

    const core::u32 rules = recipe.scatterCount < kMaxScatterRules ? recipe.scatterCount : kMaxScatterRules;
    for (core::u32 i = 0u; i < rules; ++i)
        builder.scatter(recipe.scatter[i]);

    if (recipe.carveCaves)
        builder.caves(recipe.caves);

    if (recipe.placeSettlement)
        builder.settlement(recipe.settlement);

    // After the settlement, never before: the road field is anchored on the
    // districts the town laid out, so growing first would steer the network by
    // a town that does not exist yet.
    if (recipe.growRoads)
        builder.roads(recipe.roads);

    if (recipe.checkPlayability)
        builder.validate(recipe.gate);

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
    result.ok =
        (result.entityCount > 0u && result.stateSignature != kFnv1aOffsetBasis && gateOk) ? 1u : 0u;
    return result;
}

} // namespace lpl::procgen
