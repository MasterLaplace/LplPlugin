#ifndef LPL_PROCGEN_ENDLESSPLAN_INL
#define LPL_PROCGEN_ENDLESSPLAN_INL

namespace lpl::procgen {

inline EndlessPlan endlessPlanFromRecipe(const WorldRecipe &recipe, core::u32 chunkSize, const WalkScale &scale)
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
    plan.rule.rockLine = plan.chunk.noise.amplitude * scale.rockLine;
    plan.rule.snowLine = plan.chunk.noise.amplitude * scale.snowLine;
    plan.rule.vegetationOneIn = scale.vegetationOneIn;
    plan.rule.altitudeCooling = ChunkTerrainRule{}.altitudeCooling / scale.reliefScale;
    plan.rule.riverDepth = scale.riverDepth;
    plan.rule.riverFill = scale.riverFill;

    plan.rivers.seaLevel = plan.rule.seaLevel;
    plan.rivers.riverThreshold = calibrateRiverThreshold(plan.chunk, plan.rivers, scale.targetRiverShare);

    plan.riverSurfaceRise = plan.rule.riverDepth * plan.rule.riverFill;

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

    plan.rule.buildWarrens = scale.caveMouths && scale.caveWarrens;
    plan.rule.warren = caveWarrenDefaults();
    plan.rule.warren.halfSpan = scale.warrenHalfSpan;
    plan.rule.warren.seaLevel = plan.rule.seaLevel;
    plan.rule.warren.villages = plan.rule.villages;
    plan.rule.warren.kind = recipe.caveKind;

    return plan;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_ENDLESSPLAN_INL
