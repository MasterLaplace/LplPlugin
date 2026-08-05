/**
 * @file WorldAtlas.hpp
 * @brief Everything an instrument looks at, which is more than a game reads.
 *
 * @ref WorldSnapshot is deliberately narrow: what a game reads and nothing it
 * does not, because on a 4 MiB heap every pass's intermediate grid is worth
 * freeing the moment the few useful ones are copied out. That is the right answer
 * for ring 0 and the wrong one for a viewer, whose entire job is to show the
 * intermediates — the drainage that decided where the rivers went, the provinces
 * the districts were laid on, the six climate axes the classifier read.
 *
 * So an atlas IS a snapshot, extended. Not a second aggregate that happens to
 * overlap it: `WorldAtlas : WorldSnapshot` means the fields a game reads are
 * declared once, a function that takes a `const WorldSnapshot &` accepts an atlas
 * for free, and a pass that grows the snapshot grows both. The map viewer had its
 * own parallel version of this struct — its own names for @c stats counters, its
 * own copies of @c DrainageNetwork fields, its own libm logarithm where
 * @ref fixedLog2 already existed — and a second one was about to be written for
 * the editor. That is the duplication this file exists to refuse.
 *
 * There is no atlas in ring 0 and there should never be: nothing here is
 * authoritative, and the kernel viewer draws a heightfield, not an instrument
 * panel. It is header-only for that reason among others — a target that does not
 * include it pays nothing.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_WORLD_ATLAS_HPP
#    define LPL_PROCGEN_WORLD_ATLAS_HPP

#    include <lpl/procgen/WorldSnapshot.hpp>

namespace lpl::procgen {

/**
 * @struct WorldAtlas
 * @brief A snapshot plus the grids only a diagnostic view reads.
 *
 * Every member is a copy of a @ref WorldBuilder output, under the builder's own
 * name. Nothing is flattened out of @c stats or out of @ref DrainageNetwork: two
 * fields answering one question is how a viewer ends up reporting a stale count
 * beside a live grid.
 */
struct WorldAtlas : WorldSnapshot {
    DrainageNetwork drainage{}; ///< Flow directions, accumulation, the filled surface.
    VoronoiDiagram regions{};   ///< Surface provinces, when partitioned.
    ClimateField climate{};     ///< The six axes, when shaped.
    DungeonMap dungeon{};       ///< The flat underground layer, when carved.
    CaveSystem caveSystem{};    ///< The layered underground, when dug.
    VoxelVolume townVolume{};   ///< The town raised by the shape grammar.
    VoxelVolume roadsideVolume{};              ///< Roadside modules, when decorated.
    lpl::pmr::vector<BuildingPlot> plots{};    ///< Building footprints the settlement laid out.

    /**
     * @brief The height a viewer draws water at.
     *
     * Not a builder output: it is the threshold the *caller* classified biomes
     * with, and a world lifted to clear the physics floor carries a lifted sea
     * with it. Left for the caller to set for exactly that reason — deriving it
     * here would silently disagree with the biome map whenever the terrain moved.
     */
    core::f32 seaLevel{0.0f};
};

/**
 * @brief Copies out of @p builder everything an instrument looks at.
 *
 * Takes a configured builder rather than a recipe because the two consumers
 * configure differently and both are legitimate: the editor drives a
 * @ref WorldRecipe through @ref applyRecipe, while the map viewer sets up passes
 * a recipe cannot yet name (provinces, terraces, the three alternative cave
 * generators) and measures the terrain before deciding how far to lift it. One
 * copy-out, two configurations — the alternative is two copy-outs, and then only
 * one of them learns about the next pass.
 *
 * @param builder A builder every wanted pass has already run on.
 * @param rule    What counts as impassable.
 * @param out     Destination; @c stats and @c seaLevel stay the caller's.
 */
inline void captureAtlas(const WorldBuilder &builder, const WalkabilityRule &rule, WorldAtlas &out)
{
    captureSnapshot(builder, rule, out);

    out.drainage = builder.drainage();
    out.regions = builder.regionMap();
    out.climate = builder.climateField();
    out.dungeon = builder.dungeonMap();
    out.caveSystem = builder.caves();
    out.townVolume = builder.townVolume();
    out.roadsideVolume = builder.roadsideVolume();
    out.plots = builder.plots();
}

/**
 * @brief Runs @p recipe and copies out everything an instrument looks at.
 *
 * The same pipeline @ref buildSnapshot runs, kept as one call so an editor cannot
 * end up showing a world its own Generate button did not build. Generation is
 * deterministic, so a recipe re-run for its pictures is the same world down to the
 * bit — that is what makes a diagnostic re-derivation honest rather than a second
 * source of truth.
 *
 * @param recipe     The world to build.
 * @param registry   Where scattered props become entities, or nullptr for grids only.
 * @param outPropIds Receives the created entity ids, or nullptr.
 * @param rule       What counts as impassable.
 */
[[nodiscard]] inline WorldAtlas buildAtlas(WorldRecipe recipe, ecs::Registry *registry,
                                           lpl::pmr::vector<ecs::EntityId> *outPropIds,
                                           const WalkabilityRule &rule = WalkabilityRule{})
{
    WorldAtlas out;

    WorldBuilder builder{recipe.seed};
    // See buildSnapshot: the rule and the reported sea level both travel with the lift,
    // or every absolute height an instrument reads is off by it.
    const core::f32 lift = applyRecipe(builder, recipe);
    WalkabilityRule lifted = rule;
    lifted.seaLevel += lift;

    out.stats = registry != nullptr ? builder.materializeProps(*registry, outPropIds) : builder.bakeGrids();
    captureAtlas(builder, lifted, out);
    out.seaLevel = recipe.biomes.seaLevel + lift;
    return out;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLD_ATLAS_HPP
