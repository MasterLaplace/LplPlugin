/**
 * @file EndlessPlan.hpp
 * @brief The same world, walked through instead of looked at from above.
 *
 * A @ref WorldRecipe describes a bounded map. Streaming the same world endlessly
 * needs @ref ChunkParams and a @ref ChunkTerrainRule, and those used to be written by
 * hand next to the recipe — two descriptions of one world, in one file, free to drift
 * apart with nothing to say they had. They are not the same numbers, either, and that
 * is the interesting part: a recipe's amplitude is tuned for a 64-cell map seen from
 * above, where sixteen metres peak-to-trough is a mountain range. Walk through the
 * same terrain at eye height and it is a lawn.
 *
 * So the relationship between the two is a real piece of knowledge — how you turn a
 * map into a place — and it belongs here, with a test, rather than in a sample's
 * constant block where nothing could exercise it.
 *
 * What is NOT derived: the streaming budget (radii, release ratio, per-tick
 * allowance). That is @ref StreamingParams' business and a property of the host's
 * memory, not of the world. Copying it in here would be the duplication this file
 * exists to remove.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_ENDLESSPLAN_HPP
#    define LPL_PROCGEN_ENDLESSPLAN_HPP

#    include <lpl/procgen/ChunkTerrain.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

namespace lpl::procgen {

/**
 * @struct WalkScale
 * @brief How much bigger the world has to be when you are inside it.
 *
 * Every default here was measured against a body two metres tall walking through
 * trees six point eight metres tall — that is the frame these numbers mean something
 * in. Raising @ref reliefScale alone makes mountains without making plains, which is
 * why the shaping terms come with it: a walker needs somewhere flat to walk.
 */
struct WalkScale {
    core::f32 reliefScale{2.8f};      ///< Multiplies the recipe's amplitude.
    core::f32 reliefFrequency{0.26f}; ///< Multiplies its frequency: fewer, wider landforms.
    core::f32 landLift{4.0f};         ///< Raises the base so the walker starts above water.

    /// Shaping. A map reads fine as raw noise; a place to walk needs plains.
    core::f32 plainsFlatten{0.80f};
    core::f32 plainsWidth{0.55f};
    core::f32 mountainThreshold{0.58f};
    core::f32 mountainGain{5.0f};

    /**
     * @brief Where bare rock starts, and where snow does, in the WALKED frame.
     *
     * Absolute metres, so they have to follow the relief they describe. At the flat
     * map's eight and eleven, a world whose mountains rise five times faster comes out
     * as bare rock under snow everywhere, including the plains.
     */
    core::f32 rockLine{42.0f};
    core::f32 snowLine{62.0f};

    core::u32 erosionIterations{6u}; ///< Thermal passes per chunk; also sets the apron.

    /**
     * @brief One wooded cell in N grows a plant.
     *
     * Fourteen, not three. This is what actually decides how many trees exist — not
     * the recipe's scatter densities, which govern a different pass. One in three was
     * measured on terrain where most ground was rock, water or too steep to plant on,
     * so it only ever applied to a fraction of the world; flattening made nearly all
     * of it plantable at once and the same rule became a wall of trunks.
     */
    core::u32 vegetationOneIn{14u};
};

/**
 * @struct EndlessPlan
 * @brief What a streamer needs: the terrain parameters and the content rule.
 */
struct EndlessPlan {
    ChunkParams chunk{};
    ChunkTerrainRule rule{};
};

/**
 * @brief Derives the endless form of @p recipe.
 *
 * The seed, the noise kind and the octave structure are the RECIPE's, so changing the
 * world a document describes changes the world you walk through — which was true only
 * by coincidence while the two were written separately.
 *
 * Sea level likewise comes from the recipe's biome thresholds rather than a number
 * chosen here. Where the sea is has to be one answer: a second one classifies a cell
 * as land, draws water over it and refuses to let anything walk on it, all at once.
 *
 * @param recipe    The bounded world to walk through.
 * @param chunkSize Cells along each edge of a chunk.
 * @param scale     How much bigger it has to be. Defaults are the measured ones.
 */
[[nodiscard]] inline EndlessPlan endlessPlanFromRecipe(const WorldRecipe &recipe, core::u32 chunkSize,
                                                       const WalkScale &scale = WalkScale{})
{
    EndlessPlan plan;

    plan.chunk.size = chunkSize;
    plan.chunk.worldSeed = recipe.seed;
    plan.chunk.noise = recipe.terrain;
    plan.chunk.noise.seed = recipe.seed;
    plan.chunk.noise.amplitude = recipe.terrain.amplitude * scale.reliefScale;
    plan.chunk.noise.frequency = recipe.terrain.frequency * scale.reliefFrequency;
    plan.chunk.noise.baseHeight = recipe.terrain.baseHeight + scale.landLift;
    plan.chunk.noise.plainsFlatten = scale.plainsFlatten;
    plan.chunk.noise.plainsWidth = scale.plainsWidth;
    plan.chunk.noise.mountainThreshold = scale.mountainThreshold;
    plan.chunk.noise.mountainGain = scale.mountainGain;

    plan.rule.erosionIterations = scale.erosionIterations;
    plan.rule.seaLevel = recipe.biomes.seaLevel;
    plan.rule.beachBand = recipe.biomes.beachHeight;
    plan.rule.rockLine = scale.rockLine;
    plan.rule.snowLine = scale.snowLine;
    plan.rule.vegetationOneIn = scale.vegetationOneIn;

    return plan;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_ENDLESSPLAN_HPP
