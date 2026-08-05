/**
 * @file WorldSnapshot.hpp
 * @brief Building a whole world from a seed, and keeping only what a game reads.
 *
 * "Build a world from a recipe" was written inside a sample, and it should never
 * have been: it is the single most reusable thing this module does. @ref bakeWorld
 * already runs the passes; what was missing is the step after — a WorldBuilder
 * holds the intermediate grid of every pass, and on a 4 MiB heap those are worth
 * freeing the moment the few a game actually reads have been copied out.
 *
 * So this is that: one call in, one plain aggregate out, the builder gone by the
 * time it returns. A game that wants the intermediates still has @ref WorldBuilder;
 * a game that wants a world has this.
 *
 * The walkability mask is part of the snapshot for a reason learned the hard way:
 * three separate notions of "blocked" — one for the pheromone field, one for the
 * herd, one for the spawn — is how an animal ends up standing in a lake that the
 * scent flows around. One mask, three readers.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_WORLD_SNAPSHOT_HPP
#    define LPL_PROCGEN_WORLD_SNAPSHOT_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>
#    include <lpl/procgen/Settlement.hpp>
#    include <lpl/procgen/WorldBuilder.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

namespace lpl::procgen {

/**
 * @struct WalkabilityRule
 * @brief What makes a cell impassable.
 */
struct WalkabilityRule {
    core::f32 seaLevel{-1.0f}; ///< Below this, a cell is drowned.
    core::f32 maxSlope{2.4f};  ///< Above this, a cell is too steep to stand on.
};

/**
 * @struct WorldSnapshot
 * @brief Everything a game reads from a built world, and nothing it does not.
 */
struct WorldSnapshot {
    Heightfield height{};
    BiomeMap biomes{};
    Grid<math::Fixed32> moisture{};
    Grid<core::u8> rivers{};
    SettlementMap settlement{};
    Grid<core::u8> roads{};
    Grid<core::u8> blocked{}; ///< The one walkability mask; see the file comment.

    core::u32 width{0u};
    core::u32 depth{0u};
    math::Fixed32 lowest{};
    math::Fixed32 highest{};
    core::u32 biomeCounts[static_cast<core::u32>(BiomeId::Count)]{};

    BuiltWorldStats stats{};
    bool gatePassed{false};
};

/**
 * @brief Copies out of @p builder everything a game reads, and derives the rest.
 *
 * Separate from @ref buildSnapshot because materialisation is the caller's
 * business — a viewer wants the grids and no entities at all — and because a
 * second aggregate builds on this one: see @ref WorldAtlas, which is this plus the
 * grids only an instrument looks at. Two copy-out routines would drift the day one
 * of them learned about a new pass.
 *
 * Leaves @c stats alone: whoever materialised knows it, and this function did not.
 *
 * @param builder A builder every wanted pass has already run on.
 * @param rule    What counts as impassable.
 * @param out     Destination.
 */
inline void captureSnapshot(const WorldBuilder &builder, const WalkabilityRule &rule, WorldSnapshot &out)
{
    out.gatePassed = builder.gatePassed();

    out.height = builder.heightfield();
    out.biomes = builder.biomeMap();
    out.moisture = builder.moisture();
    out.rivers = riverMask(builder.drainage(), 0.02f);
    out.settlement = builder.settlementMap();
    out.roads = builder.roadMap();

    out.width = out.height.width();
    out.depth = out.height.depth();
    (void) heightRange(out.height, out.lowest, out.highest);
    countBiomes(out.biomes, out.biomeCounts);

    out.blocked = Grid<core::u8>{out.width, out.depth, 0u};
    for (core::u32 z = 0u; z < out.depth; ++z)
        for (core::u32 x = 0u; x < out.width; ++x)
        {
            const bool drowned = out.height.at(x, z).toFloat() < rule.seaLevel;
            const bool steep = slopeAt(out.height, x, z).toFloat() > rule.maxSlope;
            out.blocked.at(x, z) = (drowned || steep) ? 1u : 0u;
        }
}

/**
 * @brief Bakes @p recipe and copies out what a game reads.
 *
 * @param recipe    The world to build. Taken by value: the seeds are derived here.
 * @param registry  Where the scatter's props become entities, or nullptr to skip
 *                  materialising them (a viewer that only draws a heightfield).
 * @param outPropIds Receives the created entity ids, so a caller can retire the
 *                   previous world's props before the next one is scattered — the
 *                   registry has no bulk clear by design.
 * @param rule      What counts as impassable.
 */
[[nodiscard]] inline WorldSnapshot buildSnapshot(WorldRecipe recipe, ecs::Registry *registry,
                                                 lpl::pmr::vector<ecs::EntityId> *outPropIds,
                                                 const WalkabilityRule &rule = WalkabilityRule{})
{
    WorldSnapshot out;

    // The builder is a local on purpose: it holds every pass's intermediate
    // grid, and those are freed as it goes out of scope.
    WorldBuilder builder{recipe.seed};
    // The rule reads ABSOLUTE heights, and a recipe with a ground clearance moves the
    // world after its own thresholds were written. Shifting the rule by the applied
    // lift is what stops "below sea level" from meaning a different altitude than the
    // classifier used — the failure looks like walkable water, and it is silent.
    const core::f32 lift = applyRecipe(builder, recipe);
    WalkabilityRule lifted = rule;
    lifted.seaLevel += lift;

    if (registry != nullptr)
        out.stats = builder.materializeProps(*registry, outPropIds);

    captureSnapshot(builder, lifted, out);
    return out;
}

/**
 * @brief Cells where vegetation grows, thinned deterministically.
 *
 * The rule a game states is "one plant per wooded cell, thinned so the map is not
 * a solid canopy", and the thinning has to be reproducible or a reload grows a
 * different forest. The predicate decides which biomes count, so a game can say
 * "conifers on taiga only" without this function knowing what a conifer is.
 *
 * @param oneIn   Keep roughly one candidate cell in this many; 1 keeps them all.
 * @param emit    Called with (cellX, cellZ) for each plant.
 */
template <typename IsWooded, typename Emit>
core::u32 scatterVegetation(const WorldSnapshot &snapshot, core::u32 seed, core::u32 oneIn, IsWooded &&isWooded,
                            Emit &&emit)
{
    math::Random thin{seed};
    core::u32 planted = 0u;
    const core::u32 divisor = oneIn == 0u ? 1u : oneIn;
    for (core::u32 z = 0u; z < snapshot.depth; ++z)
        for (core::u32 x = 0u; x < snapshot.width; ++x)
        {
            if (!isWooded(snapshot.biomes.at(x, z)) || snapshot.blocked.at(x, z) != 0u)
                continue;
            if (divisor > 1u && thin.below(divisor) != 0u)
                continue;
            emit(static_cast<core::i32>(x), static_cast<core::i32>(z));
            ++planted;
        }
    return planted;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLD_SNAPSHOT_HPP
