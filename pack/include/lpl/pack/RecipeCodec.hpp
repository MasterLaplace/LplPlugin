/**
 * @file RecipeCodec.hpp
 * @brief Translates between the wire recipe and the engine recipe.
 *
 * The one place that knows both layouts, kept apart from GamePack.hpp so the
 * container format stays usable without pulling procgen in. Every field is
 * copied by name: a reordering on either side is then a compile error rather
 * than a silently reinterpreted world.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PACK_RECIPECODEC_HPP
#    define LPL_PACK_RECIPECODEC_HPP

#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/pack/GamePack.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

static_assert(lpl::pack::kWireScatterRules == lpl::procgen::kMaxScatterRules,
              "wire and engine must agree on how many scatter rules a recipe carries");

static_assert(lpl::pack::kWireRoadsidePattern == lpl::procgen::kMaxRoadsidePattern,
              "wire and engine must agree on how long a roadside grammar may be");

static_assert(sizeof(lpl::pack::LivingV1::species) / sizeof(lpl::pack::LivingSpeciesV1) ==
                  lpl::ecology::kMaxLivingSpecies,
              "wire and engine must agree on how many species a living recipe carries");

namespace lpl::pack {

namespace detail {

/**
 * @brief Reads one wire scatter rule.
 *
 * A free function because a rule is converted in both directions and a field-by-field
 * copy written twice is a field silently dropped in one of them the day a rule grows a
 * knob.
 */
inline void readScatterRule(const ScatterV1 &from, procgen::ScatterRule &to) noexcept
{
    to.biome = static_cast<procgen::BiomeId>(from.biome);
    to.density = from.density;
    to.halfExtent = from.halfExtent;
    to.maxSlope = from.maxSlope;
    to.minMoisture = from.minMoisture;
    to.maxMoisture = from.maxMoisture;
    to.moistureAffinity = from.moistureAffinity;
    to.tag = from.tag;
    to.treeLine = from.treeLine;
    to.altitudeFalloff = from.altitudeFalloff;
    to.maxRiverDistance = from.maxRiverDistance;
    to.endemicShare = from.endemicShare;
    to.collidable = (from.flags & kScatterFlagCollidable) != 0u;
}

/// Flattens one scatter rule into its wire form. See @ref readScatterRule.
inline void writeScatterRule(const procgen::ScatterRule &from, ScatterV1 &to) noexcept
{
    to.biome = static_cast<core::u32>(from.biome);
    to.density = from.density;
    to.halfExtent = from.halfExtent;
    to.maxSlope = from.maxSlope;
    to.minMoisture = from.minMoisture;
    to.maxMoisture = from.maxMoisture;
    to.moistureAffinity = from.moistureAffinity;
    to.tag = from.tag;
    to.treeLine = from.treeLine;
    to.altitudeFalloff = from.altitudeFalloff;
    to.maxRiverDistance = from.maxRiverDistance;
    to.endemicShare = from.endemicShare;
    to.flags = from.collidable ? kScatterFlagCollidable : 0u;
}

} // namespace detail

/**
 * @brief Expands a wire recipe into the engine's in-memory recipe.
 * @param wire The decoded section payload.
 * @return The recipe procgen::bakeWorld consumes.
 */
[[nodiscard]] inline procgen::WorldRecipe toEngineRecipe(const RecipeV1 &wire) noexcept
{
    procgen::WorldRecipe recipe{};

    recipe.seed = wire.seed;
    recipe.width = wire.width;
    recipe.depth = wire.depth;
    recipe.cellSize = wire.cellSize;

    recipe.terrain.seed = wire.noiseSeed;
    recipe.terrain.frequency = wire.noiseFrequency;
    recipe.terrain.amplitude = wire.noiseAmplitude;
    recipe.terrain.octaves = wire.noiseOctaves;
    recipe.terrain.baseHeight = wire.noiseBaseHeight;
    recipe.terrain.lacunarity = wire.noiseLacunarity;
    recipe.terrain.persistence = wire.noisePersistence;
    recipe.terrain.warpStrength = wire.noiseWarpStrength;
    recipe.terrain.kind = static_cast<procgen::NoiseKind>(wire.noiseKind);
    recipe.heightLow = wire.heightLow;
    recipe.heightHigh = wire.heightHigh;
    recipe.groundClearance = wire.groundClearance;

    recipe.thermal.iterations = wire.thermalIterations;
    recipe.thermal.talus = wire.thermalTalus;
    recipe.thermal.carryFraction = wire.thermalCarryFraction;

    recipe.hydraulic.iterations = wire.hydraulicIterations;
    recipe.hydraulic.rainAmount = wire.hydraulicRainAmount;
    recipe.hydraulic.solubility = wire.hydraulicSolubility;
    recipe.hydraulic.evaporation = wire.hydraulicEvaporation;
    recipe.hydraulic.sedimentCapacity = wire.hydraulicSedimentCapacity;
    recipe.hydraulic.deposition = wire.hydraulicDeposition;
    recipe.hydraulic.minSlope = wire.hydraulicMinSlope;

    recipe.rivers.density = wire.riverDensity;
    recipe.rivers.carveDepth = wire.riverCarveDepth;
    recipe.rivers.smoothing = wire.riverSmoothing;

    recipe.climate.rainfallWeight = wire.climateRainfallWeight;
    recipe.climate.flowWeight = wire.climateFlowWeight;
    recipe.climate.altitudeWeight = wire.climateAltitudeWeight;
    recipe.climate.coastWeight = wire.climateCoastWeight;
    recipe.climate.seaLevel = wire.climateSeaLevel;
    recipe.climate.coastReach = wire.climateCoastReach;
    recipe.climate.rainShadow = wire.climateRainShadow;
    recipe.climate.windDirection = wire.climateWindDirection;
    recipe.climate.smoothing = wire.climateSmoothing;
    recipe.climate.rainfallSeed = wire.climateRainfallSeed;
    recipe.climate.rainfallBelts = wire.climateRainfallBelts;
    recipe.climate.rainfallOctaves = wire.climateRainfallOctaves;

    recipe.biomes.seaLevel = wire.biomeSeaLevel;
    recipe.biomes.beachHeight = wire.biomeBeachHeight;
    recipe.biomes.mountainHeight = wire.biomeMountainHeight;
    recipe.biomes.snowHeight = wire.biomeSnowHeight;
    recipe.biomes.snowlineWarmth = wire.biomeSnowlineWarmth;

    recipe.axes.coldLatitude = wire.axisColdLatitude;
    recipe.axes.lapseRate = wire.axisLapseRate;
    recipe.axes.coastReach = wire.axisCoastReach;
    recipe.axes.weirdnessSeed = wire.axisWeirdnessSeed;
    recipe.axes.weirdnessBelts = wire.axisWeirdnessBelts;
    recipe.axes.weirdnessOctaves = wire.axisWeirdnessOctaves;
    recipe.axes.surfaceDepth = wire.axisSurfaceDepth;

    recipe.provinces.width = wire.provinceWidth;
    recipe.provinces.depth = wire.provinceDepth;
    recipe.provinces.seed = wire.provinceSeed;
    recipe.provinces.cellSize = wire.provinceCellSize;
    recipe.provinces.jitter = wire.provinceJitter;
    recipe.provinces.warpStrength = wire.provinceWarpStrength;
    recipe.provinces.metric = static_cast<procgen::DistanceMetric>(wire.provinceMetric);

    recipe.terraceSteps = wire.terraceSteps;
    // Clamped rather than trusted: a byte from disk naming a generator that does not
    // exist would otherwise index a switch that has no case for it.
    //
    // ⚠ The bound is the enum's LAST value and must stay that way. It was written out
    // as `Layered` and went stale the moment `Auto` was added: a cartridge that said
    // "auto" baked a 4, the decoder clamped it back to Cellular, and the document's
    // word was silently discarded on the way in. Nothing failed — the world just was
    // not the one the document asked for, which is the quietest way a format can lie.
    recipe.caveKind = wire.caveKind <= static_cast<core::u32>(procgen::CaveKind::Auto) ?
                          static_cast<procgen::CaveKind>(wire.caveKind) :
                          procgen::CaveKind::Cellular;

    recipe.rooms.width = wire.roomsWidth;
    recipe.rooms.depth = wire.roomsDepth;
    recipe.rooms.seed = wire.roomsSeed;
    recipe.rooms.maxDepth = wire.roomsMaxDepth;
    recipe.rooms.minLeafSize = wire.roomsMinLeafSize;
    recipe.rooms.roomPadding = wire.roomsPadding;
    recipe.rooms.corridorWidth = wire.roomsCorridorWidth;

    recipe.aggregation.width = wire.dlaWidth;
    recipe.aggregation.depth = wire.dlaDepth;
    recipe.aggregation.seed = wire.dlaSeed;
    recipe.aggregation.particles = wire.dlaParticles;
    recipe.aggregation.maxStepsPerParticle = wire.dlaMaxStepsPerParticle;
    recipe.aggregation.spawnMargin = wire.dlaSpawnMargin;
    recipe.aggregation.thickness = wire.dlaThickness;

    recipe.caveSystem.width = wire.systemWidth;
    recipe.caveSystem.depth = wire.systemDepth;
    recipe.caveSystem.seed = wire.systemSeed;
    recipe.caveSystem.layers = wire.systemLayers;
    recipe.caveSystem.levelsPerLayer = wire.systemLevelsPerLayer;
    recipe.caveSystem.topFill = wire.systemTopFill;
    recipe.caveSystem.deepFill = wire.systemDeepFill;
    recipe.caveSystem.automatonSteps = wire.systemAutomatonSteps;
    recipe.caveSystem.minChamberSize = wire.systemMinChamberSize;
    recipe.caveSystem.shaftsPerPair = wire.systemShaftsPerPair;
    recipe.caveSystem.entrances = wire.systemEntrances;
    recipe.caveSystem.entranceMaxSlope = wire.systemEntranceMaxSlope;

    recipe.buildings.seed = wire.buildingSeed;
    recipe.buildings.minFloors = wire.buildingMinFloors;
    recipe.buildings.maxFloors = wire.buildingMaxFloors;
    recipe.buildings.baseHeight = wire.buildingBaseHeight;
    recipe.buildings.floorHeight = wire.buildingFloorHeight;
    recipe.buildings.roofHeight = wire.buildingRoofHeight;
    recipe.buildings.inset = wire.buildingInset;
    recipe.buildings.roofTaper = wire.buildingRoofTaper;
    recipe.buildings.baseMaterial = static_cast<core::u8>(wire.buildingBaseMaterial);
    recipe.buildings.wallMaterial = static_cast<core::u8>(wire.buildingWallMaterial);
    recipe.buildings.roofMaterial = static_cast<core::u8>(wire.buildingRoofMaterial);
    recipe.buildings.hollow = (wire.flags & kRecipeFlagBuildingsHollow) != 0u;

    recipe.roadsideLevels = wire.roadsideLevels;
    for (core::u32 i = 0u; i < kWireRoadsidePattern; ++i)
        recipe.roadsidePattern[i] = wire.roadsidePattern[i];
    // A pattern that arrived unterminated is a truncated grammar, not a shorter one.
    recipe.roadsidePattern[kWireRoadsidePattern - 1u] = '\0';

    recipe.partitionRegions = (wire.flags & kRecipeFlagPartitionRegions) != 0u;
    recipe.raiseBuildings = (wire.flags & kRecipeFlagRaiseBuildings) != 0u;

    recipe.caves.width = wire.caveWidth;
    recipe.caves.depth = wire.caveDepth;
    recipe.caves.seed = wire.caveSeed;
    recipe.caves.fillProbability = wire.caveFillProbability;
    recipe.caves.steps = wire.caveSteps;
    recipe.caves.birthLimit = wire.caveBirthLimit;
    recipe.caves.survivalLimit = wire.caveSurvivalLimit;
    recipe.caves.minRegionSize = wire.caveMinRegionSize;

    recipe.settlement.seed = wire.settlementSeed;
    recipe.settlement.districtSize = wire.settlementDistrictSize;
    recipe.settlement.roadWidth = wire.settlementRoadWidth;
    recipe.settlement.plazaRadius = wire.settlementPlazaRadius;
    recipe.settlement.minPlot = wire.settlementMinPlot;
    recipe.settlement.maxPlot = wire.settlementMaxPlot;
    recipe.settlement.plotDensity = wire.settlementPlotDensity;
    recipe.settlement.maxSlope = wire.settlementMaxSlope;
    recipe.settlement.minHeight = wire.settlementMinHeight;

    recipe.roads.seed = wire.roadSeed;
    recipe.roads.iterations = wire.roadIterations;
    recipe.roads.stepLength = wire.roadStepLength;
    recipe.roads.conform = wire.roadConform;
    recipe.roads.maxSlope = wire.roadMaxSlope;
    recipe.roads.minHeight = wire.roadMinHeight;
    recipe.roads.gridDistricts = wire.roadGridDistricts;
    recipe.roads.arterials = (wire.flags & kRecipeFlagRoadArterials) != 0u;

    recipe.gate.minPathLength = wire.gateMinPathLength;
    recipe.gate.minWalkableCells = wire.gateMinWalkableCells;
    recipe.gate.maxDeadEndRatio = wire.gateMaxDeadEndRatio;
    recipe.gate.requireGoalReachable = (wire.flags & kRecipeFlagGateRequireGoal) != 0u;
    recipe.gate.requireFullyConnected = (wire.flags & kRecipeFlagGateRequireConnected) != 0u;

    // A count from untrusted bytes indexes an array, so it is clamped here
    // rather than trusted: a cartridge is input, not a promise.
    recipe.scatterCount = wire.scatterCount < kWireScatterRules ? wire.scatterCount : kWireScatterRules;
    for (core::u32 i = 0u; i < recipe.scatterCount; ++i)
        detail::readScatterRule(wire.scatter[i], recipe.scatter[i]);

    recipe.normalizeTerrain = (wire.flags & kRecipeFlagNormalizeTerrain) != 0u;
    recipe.erodeTerrain = (wire.flags & kRecipeFlagErodeTerrain) != 0u;
    recipe.carveRivers = (wire.flags & kRecipeFlagCarveRivers) != 0u;
    recipe.classifyBiomes = (wire.flags & kRecipeFlagClassifyBiomes) != 0u;
    recipe.carveCaves = (wire.flags & kRecipeFlagCarveCaves) != 0u;
    recipe.placeSettlement = (wire.flags & kRecipeFlagPlaceSettlement) != 0u;
    recipe.materializeGround = (wire.flags & kRecipeFlagMaterializeGround) != 0u;
    recipe.growRoads = (wire.flags & kRecipeFlagGrowRoads) != 0u;
    recipe.checkPlayability = (wire.flags & kRecipeFlagCheckPlayability) != 0u;

    return recipe;
}

/**
 * @brief Flattens an engine recipe into its wire form, for baking.
 * @param recipe The in-memory recipe.
 * @return The payload to write into a WorldRecipe section.
 */
[[nodiscard]] inline RecipeV1 toWireRecipe(const procgen::WorldRecipe &recipe) noexcept
{
    RecipeV1 wire{};

    wire.seed = recipe.seed;
    wire.width = recipe.width;
    wire.depth = recipe.depth;
    wire.cellSize = recipe.cellSize;

    wire.noiseSeed = recipe.terrain.seed;
    wire.noiseFrequency = recipe.terrain.frequency;
    wire.noiseAmplitude = recipe.terrain.amplitude;
    wire.noiseOctaves = recipe.terrain.octaves;
    wire.noiseBaseHeight = recipe.terrain.baseHeight;
    wire.noiseLacunarity = recipe.terrain.lacunarity;
    wire.noisePersistence = recipe.terrain.persistence;
    wire.noiseWarpStrength = recipe.terrain.warpStrength;
    wire.noiseKind = static_cast<core::u32>(recipe.terrain.kind);
    wire.heightLow = recipe.heightLow;
    wire.heightHigh = recipe.heightHigh;
    wire.groundClearance = recipe.groundClearance;

    wire.thermalIterations = recipe.thermal.iterations;
    wire.thermalTalus = recipe.thermal.talus;
    wire.thermalCarryFraction = recipe.thermal.carryFraction;

    wire.hydraulicIterations = recipe.hydraulic.iterations;
    wire.hydraulicRainAmount = recipe.hydraulic.rainAmount;
    wire.hydraulicSolubility = recipe.hydraulic.solubility;
    wire.hydraulicEvaporation = recipe.hydraulic.evaporation;
    wire.hydraulicSedimentCapacity = recipe.hydraulic.sedimentCapacity;
    wire.hydraulicDeposition = recipe.hydraulic.deposition;
    wire.hydraulicMinSlope = recipe.hydraulic.minSlope;

    wire.riverDensity = recipe.rivers.density;
    wire.riverCarveDepth = recipe.rivers.carveDepth;
    wire.riverSmoothing = recipe.rivers.smoothing;

    wire.climateRainfallWeight = recipe.climate.rainfallWeight;
    wire.climateFlowWeight = recipe.climate.flowWeight;
    wire.climateAltitudeWeight = recipe.climate.altitudeWeight;
    wire.climateCoastWeight = recipe.climate.coastWeight;
    wire.climateSeaLevel = recipe.climate.seaLevel;
    wire.climateCoastReach = recipe.climate.coastReach;
    wire.climateRainShadow = recipe.climate.rainShadow;
    wire.climateWindDirection = recipe.climate.windDirection;
    wire.climateSmoothing = recipe.climate.smoothing;
    wire.climateRainfallSeed = recipe.climate.rainfallSeed;
    wire.climateRainfallBelts = recipe.climate.rainfallBelts;
    wire.climateRainfallOctaves = recipe.climate.rainfallOctaves;

    wire.biomeSeaLevel = recipe.biomes.seaLevel;
    wire.biomeBeachHeight = recipe.biomes.beachHeight;
    wire.biomeMountainHeight = recipe.biomes.mountainHeight;
    wire.biomeSnowHeight = recipe.biomes.snowHeight;
    wire.axisColdLatitude = recipe.axes.coldLatitude;
    wire.axisLapseRate = recipe.axes.lapseRate;
    wire.axisCoastReach = recipe.axes.coastReach;
    wire.axisWeirdnessSeed = recipe.axes.weirdnessSeed;
    wire.axisWeirdnessBelts = recipe.axes.weirdnessBelts;
    wire.axisWeirdnessOctaves = recipe.axes.weirdnessOctaves;
    wire.axisSurfaceDepth = recipe.axes.surfaceDepth;
    wire.biomeSnowlineWarmth = recipe.biomes.snowlineWarmth;

    wire.provinceWidth = recipe.provinces.width;
    wire.provinceDepth = recipe.provinces.depth;
    wire.provinceSeed = recipe.provinces.seed;
    wire.provinceCellSize = recipe.provinces.cellSize;
    wire.provinceJitter = recipe.provinces.jitter;
    wire.provinceWarpStrength = recipe.provinces.warpStrength;
    wire.provinceMetric = static_cast<core::u32>(recipe.provinces.metric);

    wire.terraceSteps = recipe.terraceSteps;
    wire.caveKind = static_cast<core::u32>(recipe.caveKind);

    wire.roomsWidth = recipe.rooms.width;
    wire.roomsDepth = recipe.rooms.depth;
    wire.roomsSeed = recipe.rooms.seed;
    wire.roomsMaxDepth = recipe.rooms.maxDepth;
    wire.roomsMinLeafSize = recipe.rooms.minLeafSize;
    wire.roomsPadding = recipe.rooms.roomPadding;
    wire.roomsCorridorWidth = recipe.rooms.corridorWidth;

    wire.dlaWidth = recipe.aggregation.width;
    wire.dlaDepth = recipe.aggregation.depth;
    wire.dlaSeed = recipe.aggregation.seed;
    wire.dlaParticles = recipe.aggregation.particles;
    wire.dlaMaxStepsPerParticle = recipe.aggregation.maxStepsPerParticle;
    wire.dlaSpawnMargin = recipe.aggregation.spawnMargin;
    wire.dlaThickness = recipe.aggregation.thickness;

    wire.systemWidth = recipe.caveSystem.width;
    wire.systemDepth = recipe.caveSystem.depth;
    wire.systemSeed = recipe.caveSystem.seed;
    wire.systemLayers = recipe.caveSystem.layers;
    wire.systemLevelsPerLayer = recipe.caveSystem.levelsPerLayer;
    wire.systemTopFill = recipe.caveSystem.topFill;
    wire.systemDeepFill = recipe.caveSystem.deepFill;
    wire.systemAutomatonSteps = recipe.caveSystem.automatonSteps;
    wire.systemMinChamberSize = recipe.caveSystem.minChamberSize;
    wire.systemShaftsPerPair = recipe.caveSystem.shaftsPerPair;
    wire.systemEntrances = recipe.caveSystem.entrances;
    wire.systemEntranceMaxSlope = recipe.caveSystem.entranceMaxSlope;

    wire.buildingSeed = recipe.buildings.seed;
    wire.buildingMinFloors = recipe.buildings.minFloors;
    wire.buildingMaxFloors = recipe.buildings.maxFloors;
    wire.buildingBaseHeight = recipe.buildings.baseHeight;
    wire.buildingFloorHeight = recipe.buildings.floorHeight;
    wire.buildingRoofHeight = recipe.buildings.roofHeight;
    wire.buildingInset = recipe.buildings.inset;
    wire.buildingRoofTaper = recipe.buildings.roofTaper;
    wire.buildingBaseMaterial = recipe.buildings.baseMaterial;
    wire.buildingWallMaterial = recipe.buildings.wallMaterial;
    wire.buildingRoofMaterial = recipe.buildings.roofMaterial;

    wire.roadsideLevels = recipe.roadsideLevels;
    for (core::u32 i = 0u; i < kWireRoadsidePattern; ++i)
        wire.roadsidePattern[i] = recipe.roadsidePattern[i];
    wire.roadsidePattern[kWireRoadsidePattern - 1u] = '\0';

    wire.caveWidth = recipe.caves.width;
    wire.caveDepth = recipe.caves.depth;
    wire.caveSeed = recipe.caves.seed;
    wire.caveFillProbability = recipe.caves.fillProbability;
    wire.caveSteps = recipe.caves.steps;
    wire.caveBirthLimit = recipe.caves.birthLimit;
    wire.caveSurvivalLimit = recipe.caves.survivalLimit;
    wire.caveMinRegionSize = recipe.caves.minRegionSize;

    wire.settlementSeed = recipe.settlement.seed;
    wire.settlementDistrictSize = recipe.settlement.districtSize;
    wire.settlementRoadWidth = recipe.settlement.roadWidth;
    wire.settlementPlazaRadius = recipe.settlement.plazaRadius;
    wire.settlementMinPlot = recipe.settlement.minPlot;
    wire.settlementMaxPlot = recipe.settlement.maxPlot;
    wire.settlementPlotDensity = recipe.settlement.plotDensity;
    wire.settlementMaxSlope = recipe.settlement.maxSlope;
    wire.settlementMinHeight = recipe.settlement.minHeight;

    wire.roadSeed = recipe.roads.seed;
    wire.roadIterations = recipe.roads.iterations;
    wire.roadStepLength = recipe.roads.stepLength;
    wire.roadConform = recipe.roads.conform;
    wire.roadMaxSlope = recipe.roads.maxSlope;
    wire.roadMinHeight = recipe.roads.minHeight;
    wire.roadGridDistricts = recipe.roads.gridDistricts;

    wire.gateMinPathLength = recipe.gate.minPathLength;
    wire.gateMinWalkableCells = recipe.gate.minWalkableCells;
    wire.gateMaxDeadEndRatio = recipe.gate.maxDeadEndRatio;

    wire.scatterCount = recipe.scatterCount < kWireScatterRules ? recipe.scatterCount : kWireScatterRules;
    for (core::u32 i = 0u; i < wire.scatterCount; ++i)
    {
        const procgen::ScatterRule &from = recipe.scatter[i];
        ScatterV1 &to = wire.scatter[i];
        to.biome = static_cast<core::u32>(from.biome);
        to.density = from.density;
        to.halfExtent = from.halfExtent;
        to.maxSlope = from.maxSlope;
        to.minMoisture = from.minMoisture;
        to.maxMoisture = from.maxMoisture;
        to.moistureAffinity = from.moistureAffinity;
        to.tag = from.tag;
        to.treeLine = from.treeLine;
        to.altitudeFalloff = from.altitudeFalloff;
        to.maxRiverDistance = from.maxRiverDistance;
        to.endemicShare = from.endemicShare;
        to.flags = from.collidable ? kScatterFlagCollidable : 0u;
    }

    wire.flags = 0u;
    if (recipe.normalizeTerrain)
        wire.flags |= kRecipeFlagNormalizeTerrain;
    if (recipe.erodeTerrain)
        wire.flags |= kRecipeFlagErodeTerrain;
    if (recipe.carveRivers)
        wire.flags |= kRecipeFlagCarveRivers;
    if (recipe.classifyBiomes)
        wire.flags |= kRecipeFlagClassifyBiomes;
    if (recipe.carveCaves)
        wire.flags |= kRecipeFlagCarveCaves;
    if (recipe.placeSettlement)
        wire.flags |= kRecipeFlagPlaceSettlement;
    if (recipe.materializeGround)
        wire.flags |= kRecipeFlagMaterializeGround;
    if (recipe.growRoads)
        wire.flags |= kRecipeFlagGrowRoads;
    if (recipe.roads.arterials)
        wire.flags |= kRecipeFlagRoadArterials;
    if (recipe.checkPlayability)
        wire.flags |= kRecipeFlagCheckPlayability;
    if (recipe.gate.requireGoalReachable)
        wire.flags |= kRecipeFlagGateRequireGoal;
    if (recipe.gate.requireFullyConnected)
        wire.flags |= kRecipeFlagGateRequireConnected;
    if (recipe.partitionRegions)
        wire.flags |= kRecipeFlagPartitionRegions;
    if (recipe.raiseBuildings)
        wire.flags |= kRecipeFlagRaiseBuildings;
    if (recipe.buildings.hollow)
        wire.flags |= kRecipeFlagBuildingsHollow;

    return wire;
}

/**
 * @brief Wire living recipe to engine living recipe.
 *
 * Field by field and by NAME, like its world counterpart: a rename on either
 * side is then a compile error rather than a silently reinterpreted ecosystem.
 * Fixed32 crosses as its raw Q16.16 word — the value is the bits.
 *
 * @param wire Decoded living section.
 * @return The engine recipe it describes.
 */
[[nodiscard]] inline ecology::LivingRecipe toEngineLiving(const LivingV1 &wire) noexcept
{
    ecology::LivingRecipe recipe{};

    recipe.seed = wire.seed;
    recipe.ticks = wire.ticks;
    recipe.stepSeconds = math::Fixed32::fromRaw(wire.stepSeconds);

    recipe.width = wire.width;
    recipe.depth = wire.depth;
    recipe.channels = wire.channels;

    recipe.rooms = wire.rooms;
    recipe.creatures = wire.creatures;
    recipe.ants = wire.ants;
    recipe.boids = wire.boids;
    recipe.genomes = wire.genomes;
    recipe.packMembers = wire.packMembers;
    recipe.regrowthTicks = wire.regrowthTicks;
    recipe.headPerBody = wire.headPerBody;

    recipe.stigmergy.evaporation = wire.evaporation;
    recipe.stigmergy.diffusion = wire.diffusion;
    recipe.stigmergy.maximum = wire.maximum;
    recipe.stigmergy.floor = wire.floorValue;

    recipe.foraging.explore16 = wire.explore16;
    recipe.foraging.depositQuality = math::Fixed32::fromRaw(wire.depositQuality);
    recipe.foraging.channel = wire.trailChannel;

    recipe.flock.separationRadius = math::Fixed32::fromRaw(wire.separationRadius);
    recipe.flock.neighbourRadius = math::Fixed32::fromRaw(wire.neighbourRadius);
    recipe.flock.separationWeight = wire.separationWeight;
    recipe.flock.alignmentWeight = wire.alignmentWeight;
    recipe.flock.cohesionWeight = wire.cohesionWeight;
    recipe.flock.maxSpeed = wire.maxSpeed;

    recipe.budget.maxRealisedRooms = wire.maxRealisedRooms;
    recipe.budget.changeWeight = wire.changeWeight;
    recipe.budget.adjacentBonus = wire.adjacentBonus;
    recipe.budget.predictedBonus = wire.predictedBonus;
    recipe.budget.unlikelyPenalty = wire.unlikelyPenalty;

    recipe.heredity.mutationChance16 = wire.mutationChance16;
    recipe.heredity.mutationAmplitude = wire.mutationAmplitude;
    recipe.heredity.collapseShare16 = wire.collapseShare16;
    recipe.heredity.meltdownChance16 = wire.meltdownChance16;
    recipe.heredity.meltdownAmplitude = wire.meltdownAmplitude;
    recipe.heredity.anomalySigma = wire.anomalySigma;

    recipe.packs.maxSize = wire.packMaxSize;
    recipe.packs.minSize = wire.packMinSize;
    recipe.packs.dissolutionChance16 = wire.dissolutionChance16;
    recipe.packs.adoptStrays = wire.adoptStrays != 0u;

    const core::u32 count =
        wire.speciesCount < ecology::kMaxLivingSpecies ? wire.speciesCount : ecology::kMaxLivingSpecies;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const LivingSpeciesV1 &from = wire.species[i];
        ecology::LivingSpecies &to = recipe.species[i];
        to.params.level = static_cast<ecology::TrophicLevel>(from.level);
        to.params.growth = math::Fixed32::fromRaw(from.growth);
        to.params.mortality = math::Fixed32::fromRaw(from.mortality);
        to.params.predation = math::Fixed32::fromRaw(from.predation);
        to.params.conversion = math::Fixed32::fromRaw(from.conversion);
        to.params.capacity = math::Fixed32::fromRaw(from.capacity);
        to.params.refuge = math::Fixed32::fromRaw(from.refuge);
        to.initial = math::Fixed32::fromRaw(from.initial);
        to.preyIndex = from.preyIndex;
    }
    recipe.speciesCount = count;

    return recipe;
}

/**
 * @brief Engine living recipe to wire living recipe.
 * @param recipe Engine recipe.
 * @return Its wire form.
 */
[[nodiscard]] inline LivingV1 toWireLiving(const ecology::LivingRecipe &recipe) noexcept
{
    LivingV1 wire{};

    wire.seed = recipe.seed;
    wire.ticks = recipe.ticks;
    wire.stepSeconds = recipe.stepSeconds.raw();

    wire.width = recipe.width;
    wire.depth = recipe.depth;
    wire.channels = recipe.channels;

    wire.rooms = recipe.rooms;
    wire.creatures = recipe.creatures;
    wire.ants = recipe.ants;
    wire.boids = recipe.boids;
    wire.genomes = recipe.genomes;
    wire.packMembers = recipe.packMembers;
    wire.regrowthTicks = recipe.regrowthTicks;
    wire.headPerBody = recipe.headPerBody;

    wire.evaporation = recipe.stigmergy.evaporation;
    wire.diffusion = recipe.stigmergy.diffusion;
    wire.maximum = recipe.stigmergy.maximum;
    wire.floorValue = recipe.stigmergy.floor;

    wire.explore16 = recipe.foraging.explore16;
    wire.depositQuality = recipe.foraging.depositQuality.raw();
    wire.trailChannel = recipe.foraging.channel;

    wire.separationRadius = recipe.flock.separationRadius.raw();
    wire.neighbourRadius = recipe.flock.neighbourRadius.raw();
    wire.separationWeight = recipe.flock.separationWeight;
    wire.alignmentWeight = recipe.flock.alignmentWeight;
    wire.cohesionWeight = recipe.flock.cohesionWeight;
    wire.maxSpeed = recipe.flock.maxSpeed;

    wire.maxRealisedRooms = recipe.budget.maxRealisedRooms;
    wire.changeWeight = recipe.budget.changeWeight;
    wire.adjacentBonus = recipe.budget.adjacentBonus;
    wire.predictedBonus = recipe.budget.predictedBonus;
    wire.unlikelyPenalty = recipe.budget.unlikelyPenalty;

    wire.mutationChance16 = recipe.heredity.mutationChance16;
    wire.mutationAmplitude = recipe.heredity.mutationAmplitude;
    wire.collapseShare16 = recipe.heredity.collapseShare16;
    wire.meltdownChance16 = recipe.heredity.meltdownChance16;
    wire.meltdownAmplitude = recipe.heredity.meltdownAmplitude;
    wire.anomalySigma = recipe.heredity.anomalySigma;

    wire.packMaxSize = recipe.packs.maxSize;
    wire.packMinSize = recipe.packs.minSize;
    wire.dissolutionChance16 = recipe.packs.dissolutionChance16;
    wire.adoptStrays = recipe.packs.adoptStrays ? 1u : 0u;

    const core::u32 count =
        recipe.speciesCount < ecology::kMaxLivingSpecies ? recipe.speciesCount : ecology::kMaxLivingSpecies;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const ecology::LivingSpecies &from = recipe.species[i];
        LivingSpeciesV1 &to = wire.species[i];
        to.level = static_cast<core::u32>(from.params.level);
        to.growth = from.params.growth.raw();
        to.mortality = from.params.mortality.raw();
        to.predation = from.params.predation.raw();
        to.conversion = from.params.conversion.raw();
        to.capacity = from.params.capacity.raw();
        to.refuge = from.params.refuge.raw();
        to.initial = from.initial.raw();
        to.preyIndex = from.preyIndex;
    }
    wire.speciesCount = count;

    return wire;
}

} // namespace lpl::pack

#endif // LPL_PACK_RECIPECODEC_HPP
