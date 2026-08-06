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
#    include <lpl/procgen/Landmark.hpp>
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
    core::f32 reliefScale{2.8f}; ///< Multiplies the recipe's amplitude.
    /**
     * @brief Multiplies the recipe's frequency: fewer, WIDER landforms.
     *
     * ⚠ The number that decides whether a mountain is a massif or a spire, and it was set
     * for spires. The measure that matters is not the peak — it is how far you walk while
     * staying high. Mean run length above twenty metres, over a 480-cell traverse:
     *
     *   | reliefFrequency | run above 20 m | p99  |  max |
     *   |-----------------|----------------|------|------|
     *   |      0.26       |   8.9 cells    | 40.5 | 88.3 |
     *   |      0.16       |  12.9 cells    | 38.5 | 84.1 |
     *   |      0.10       |  16.2 cells    | 33.4 | 74.6 |
     *   |      0.06       |  33.3 cells    | 43.5 | 75.1 |
     *
     * Nine cells is a pillar you walk around; thirty-three is a range you walk over. And it
     * is this knob and not @ref mountainGain: at a fixed frequency, halving the gain moves
     * the width from 8.9 to 8.1 while flattening the summit by half.
     */
    core::f32 reliefFrequency{0.06f};
    /**
     * @brief Raises or lowers the base, i.e. how much of the world is SEA.
     *
     * ⚠ This was the wrong knob, and the record of getting that wrong is worth keeping.
     * The walked world had 0.0% sea, so it was lowered six metres to give it a coast — and
     * that turned 97% forest into 0% forest, because the climate classifier was deciding on
     * a single axis and the lift moved it across a knife edge. What the world was actually
     * missing was RELIEF: with 99% of it under two metres, there was no spread for a sea
     * level to cut through.
     *
     * With the shaping fixed the lift barely has to move. Measured over 81 chunks, at
     * amplitude 33.6:
     *
     *   | landLift | ocean | beach | grass | forest | rock | p90  |  p99 |  max |
     *   |----------|-------|-------|-------|--------|------|------|------|------|
     *   |   0.0    | 26.0% |  4.2% | 39.5% |  22.3% | 0.4% |  7.8 | 24.7 | 57.5 |
     *   |  +2.0    | 15.7% |  3.3% | 46.5% |  27.0% | 0.5% |  9.9 | 26.7 | 59.5 |
     *   |  +3.0    | 11.9% |  2.9% | 48.8% |  28.2% | 0.6% | 10.9 | 27.7 | 60.5 |
     *   |  +4.0    |  8.7% |  2.2% | 51.7% |  29.2% | 0.7% | 11.9 | 28.7 | 61.5 |
     *
     * +3.0: about a fifth of the world is water once the lakes are counted, four fifths is
     * land with forests and plains on it, and there is bare rock and snow at the top of it.
     *
     * What the old comment worried about — "so the walker starts above water" — is not this
     * field's job and never was: @ref samples::TerrainWorld::spawnBody already searches
     * outwards for a cell above the waterline with a slope it will not slide off.
     */
    core::f32 landLift{3.0f};

    /**
     * @brief Shaping. A map reads fine as raw noise; a place to walk needs plains.
     *
     * ⚠ The old values made a PANCAKE. Measured over 49 chunks: 99% of the world was below
     * 2.0 metres and the tallest thing in it was 34 — not a flat world with mountains, a
     * flat world with three spikes. There was nowhere to climb because there was almost
     * nothing above the plain.
     *
     * The distribution, at landLift −1.5 and amplitude 33.6:
     *
     *   | flatten | mThreshold |  p50 |  p90 |  p99 |  max |
     *   |---------|------------|------|------|------|------|
     *   |  0.80   |    0.58    | −1.5 |  0.6 |  2.0 | 34.1 |
     *   |  0.55    |   0.40    | −1.5 |  3.3 | 25.5 | 66.5 |
     *   |  0.30   |    0.40    | −1.5 |  6.0 | 28.9 | 69.8 |
     *   |  0.30   |    0.25    | −1.5 | 15.7 | 50.5 | 91.5 |
     *
     * 0.30 / 0.40: half the world at the waterline, ninety per cent under six metres of
     * rolling ground, one per cent above twenty-nine, and peaks near seventy. The last row
     * is a mountain range with valleys in it rather than a land with mountains in it.
     *
     * @note plainsWidth looks inert in that table and is not: it bounds the LOW side of the
     *       lowland band, so it shapes the sea floor, which no percentile above reaches.
     */
    core::f32 plainsFlatten{0.30f};
    core::f32 plainsWidth{0.55f};
    core::f32 mountainThreshold{0.40f};
    core::f32 mountainGain{5.0f};

    /**
     * @brief Where bare rock starts, and where snow does, as MULTIPLES of the amplitude.
     *
     * Absolute metres was the trap, and it was live: 42 and 62 against a world whose
     * relief measured 21 metres from end to end. Not one cell of it could ever be bare
     * rock, and not one could ever be snow — so a walker had no summit to reach, only
     * grass going up. Expressed against the amplitude the plan actually produces, they
     * follow whatever relief the scale is set to.
     *
     * 0.90 and 1.45 of 33.6 metres put rock at 30 and snow at 49, against a measured
     * ninety-ninth percentile of 29 and peaks near 70: rock is the top one per cent of the
     * world and snow is the handful of summits above it.
     */
    core::f32 rockLine{0.90f};
    core::f32 snowLine{1.45f};

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

    /**
     * @brief Share of cells that should carry a river.
     *
     * A SHARE and not a count, because a count is a number calibrated at one scale and this
     * plan changes the scale. The river threshold is derived from this by
     * @ref calibrateRiverThreshold, over a window fixed by the parameters — so it is one
     * answer for the whole world rather than one per chunk.
     *
     * Two per cent, and it is a judgement rather than a measurement: it is roughly what the
     * bounded map produces (1.5%), and the bounded map is the world these recipes were
     * written for. What it replaces was not a judgement at all — an absolute 6 that marked
     * 13.2% once the walk scale smoothed the terrain.
     */
    core::f32 targetRiverShare{0.02f};

    /**
     * @brief How deep a river cuts its bed, and how deep the water in it stands.
     *
     * The bed is lowered BELOW the surrounding land, which is the difference between a river
     * and a sheet of water lying on a hillside — the second is what the endless path drew,
     * because it marked a mask and never touched the heightfield. The bounded path has always
     * carved: `carveRivers` takes its heightfield by non-const reference.
     *
     * The fill is a FRACTION of the depth, and it is under one on purpose: a channel whose
     * water reaches the brim overflows the moment the terrain beside it dips, and terrain
     * dips. At one the surface is flush with the ground it was carved out of, which reads as
     * a wet stripe painted on the hillside rather than as a river with banks. This comment
     * said all of that while the value was 1.0 — the prose was right and the number was not.
     */
    core::f32 riverDepth{1.6f};
    core::f32 riverFill{0.72f};

    /**
     * @brief Whether the walked world has cave mouths and villages in it.
     *
     * On here and off in @ref ChunkTerrainRule, and the asymmetry is deliberate: a rule is
     * a low-level description that must not grow features nobody asked for, and a WALKED
     * world is a place a person is standing in — a place with no doors and no houses in it
     * is a landscape, which is the note this world was sent back with.
     */
    bool caveMouths{true};
    bool villages{true};

    /**
     * @brief Whether the mouths lead anywhere.
     *
     * A separate switch from @ref caveMouths and a separate cost: a shelf is a disc of
     * comparisons and a warren is a cellular automaton per floor plus a reachability
     * flood — 1.4 ms against the 0.28 ms a whole chunk takes, on the one chunk in ten
     * that owns a mouth. Off gives a world with cave mouths carved into its hillsides
     * and nothing behind them, which is a legitimate thing for a host to ask for and
     * exactly what this world had before.
     */
    bool caveWarrens{true};

    /**
     * @brief How far the gallery behind a mouth reaches, in cells from the site.
     *
     * Bounded by RESIDENCY rather than by taste, and that is the constraint to keep in
     * mind if it is ever raised: a body inside a warren must have the chunk that owns
     * it loaded, and that chunk is at most `generateRadius * chunkSize` cells away.
     * Twenty against a radius of two and a chunk of twenty-four leaves a factor of two
     * in hand.
     */
    core::u32 warrenHalfSpan{20u};

    /**
     * @brief What SHARE of candidate sites should qualify, by relief.
     *
     * ⚠ These were absolute metres, and they broke twice in one afternoon for the same
     * reason the river threshold did: an absolute threshold against a distribution that
     * moves. A village tolerance of 2.4 m admitted two sites in three and a half thousand
     * chunks; 6.0 m then gave one village per forty-nine chunks, and NONE at all once the
     * terrain shaping was fixed and the world grew hills. The cave rule went the other way
     * — from one mouth per four hundred chunks to one per three.
     *
     * A share cannot do that. @ref calibrateLandmarkRelief turns it into the metre value
     * that produces it on the terrain that actually exists, over a window fixed by the
     * parameters — so the density survives any future change to the relief.
     *
     * A village takes the FLATTEST sites and a cave mouth the STEEPEST, which is the same
     * rule read from either end.
     */
    core::f32 villageFlattestShare{0.22f};
    core::f32 caveSteepestShare{0.13f};

    /**
     * @brief How deep a mouth's shelf is cut, in metres. HUMAN scale, deliberately.
     *
     * ⚠ This was a fraction of the calibrated relief, and that was a category error with a
     * very visible result. Calibration decides WHERE a mouth goes — which hillside is steep
     * enough — and it rightly grows with the world: on a range that reaches sixty metres
     * the steepest tenth of sites has sixteen metres of relief. Deriving the mouth's SIZE
     * from that made the shelf eleven and a half metres deep and its dark face seventeen
     * metres tall: a black slab standing on a mountain, which is what the screenshot showed
     * and why no cave was recognisable as one.
     *
     * A doorway is a doorway. Calibrate the placement, fix the size.
     */
    core::f32 caveMouthDrop{2.0f};
};

/**
 * @struct EndlessPlan
 * @brief What a streamer needs: the terrain parameters and the content rule.
 */
struct EndlessPlan {
    ChunkParams chunk{};
    ChunkTerrainRule rule{};

    /**
     * @brief How rivers are decided, derived and CALIBRATED.
     *
     * Was missing, and its absence was the whole bug: a caller built the terrain from the
     * plan and the rivers from a default-constructed set, so the two disagreed about where
     * the sea was and the threshold was the one calibrated for a different scale.
     */
    EndlessRiverParams rivers{};

    /**
     * @brief How far a river's water stands above its carved bed, in world units.
     *
     * The product of the rule's depth and fill, derived here so the renderer and the
     * generator cannot hold two answers. Multiplying it out at each call site is how a
     * bed and its surface come to disagree by a factor nobody can find.
     */
    core::f32 riverSurfaceRise{0.0f};
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
    // Against the amplitude the plan actually produces, so a change of scale carries them.
    plan.rule.rockLine = plan.chunk.noise.amplitude * scale.rockLine;
    plan.rule.snowLine = plan.chunk.noise.amplitude * scale.snowLine;
    plan.rule.vegetationOneIn = scale.vegetationOneIn;
    // Cooling is a rate PER METRE, so a world whose relief is nearly three times taller
    // cools nearly three times faster over the same landform — and comes out as tundra
    // wherever it is not sea. The rock and snow lines are already scaled here for exactly
    // this reason; this is the third member of that family and it was left behind.
    plan.rule.altitudeCooling = ChunkTerrainRule{}.altitudeCooling / scale.reliefScale;
    plan.rule.riverDepth = scale.riverDepth;
    plan.rule.riverFill = scale.riverFill;

    // The river parameters belong to the PLAN, and their absence was a live defect: nothing
    // derived them, so a caller got a default-constructed set whose sea level was -1.0 while
    // the classifier's was -4.0. Where the sea is has to be one answer — the same argument
    // the rule's own comment makes, applied to the half that was left out of it.
    plan.rivers.seaLevel = plan.rule.seaLevel;
    plan.rivers.riverThreshold = calibrateRiverThreshold(plan.chunk, plan.rivers, scale.targetRiverShare);

    plan.riverSurfaceRise = plan.rule.riverDepth * plan.rule.riverFill;

    // Landmarks, in the walked frame. Every threshold is derived from something the scale
    // already states rather than restated: a village stays below the bare rock line
    // because that is where soil stops, and both kinds keep clear of the water the recipe
    // put there.
    // Calibrated, not chosen. See WalkScale::villageFlattestShare for the two times an
    // absolute metre value broke this, in opposite directions.
    plan.rule.carveCaveMouths = scale.caveMouths;
    plan.rule.caveMouths = caveMouthDefaults();
    plan.rule.caveMouths.minRelief = calibrateLandmarkRelief(plan.chunk, plan.rule.caveMouths, LandmarkKind::CaveMouth,
                                                             plan.rule.seaLevel, 1.0f - scale.caveSteepestShare);
    plan.rule.caveMouthDrop = scale.caveMouthDrop;

    plan.rule.raiseVillages = scale.villages;
    plan.rule.villages = settlementDefaults();
    plan.rule.villages.maxHeight = plan.rule.rockLine;
    plan.rule.villages.maxRelief = calibrateLandmarkRelief(plan.chunk, plan.rule.villages, LandmarkKind::Settlement,
                                                           plan.rule.seaLevel, scale.villageFlattestShare);

    // The cave BEHIND the mouth. Its geometry is derived rather than restated, for the
    // same reason the river surface is: the shelf depth, the sea and the settlement
    // lattice all exist above, and a second copy of any of them is a warren that
    // disagrees with the ground it is under.
    plan.rule.buildWarrens = scale.caveMouths && scale.caveWarrens;
    plan.rule.warren = caveWarrenDefaults();
    plan.rule.warren.halfSpan = scale.warrenHalfSpan;
    plan.rule.warren.seaLevel = plan.rule.seaLevel;
    plan.rule.warren.villages = plan.rule.villages;
    // Straight from the document. Nothing else in the streamed path reads it, which is
    // why a `.lplscene` that named `bsp` used to get cellular caves everywhere: the
    // recipe's word reached the BOUNDED builder and nothing else.
    plan.rule.warren.kind = recipe.caveKind;

    return plan;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_ENDLESSPLAN_HPP
