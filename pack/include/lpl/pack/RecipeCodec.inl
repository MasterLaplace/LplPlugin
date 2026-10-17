#ifndef LPL_PACK_RECIPECODEC_INL
#define LPL_PACK_RECIPECODEC_INL

namespace lpl::pack {

namespace detail {

constexpr bool isKnown(procgen::NoiseKind kind) noexcept
{
    switch (kind)
    {
    case procgen::NoiseKind::Fbm:
    case procgen::NoiseKind::Ridged:
    case procgen::NoiseKind::Billow: return true;
    }
    return false;
}

constexpr bool isKnown(procgen::DistanceMetric metric) noexcept
{
    switch (metric)
    {
    case procgen::DistanceMetric::Euclidean:
    case procgen::DistanceMetric::Manhattan:
    case procgen::DistanceMetric::Chebyshev: return true;
    }
    return false;
}

constexpr bool isKnown(procgen::CaveKind kind) noexcept
{
    switch (kind)
    {
    case procgen::CaveKind::Cellular:
    case procgen::CaveKind::Bsp:
    case procgen::CaveKind::Dla:
    case procgen::CaveKind::Layered:
    case procgen::CaveKind::Auto: return true;
    }
    return false;
}

constexpr bool isKnown(procgen::BiomeId biome) noexcept
{
    switch (biome)
    {
    case procgen::BiomeId::Ocean:
    case procgen::BiomeId::Beach:
    case procgen::BiomeId::Snow:
    case procgen::BiomeId::Tundra:
    case procgen::BiomeId::Taiga:
    case procgen::BiomeId::Rock:
    case procgen::BiomeId::Desert:
    case procgen::BiomeId::Savanna:
    case procgen::BiomeId::Grassland:
    case procgen::BiomeId::Forest:
    case procgen::BiomeId::Rainforest:
    case procgen::BiomeId::Marsh:
    case procgen::BiomeId::Lake: return true;
    case procgen::BiomeId::Count: return false;
    }
    return false;
}

constexpr bool isKnown(ecology::TrophicLevel level) noexcept
{
    switch (level)
    {
    case ecology::TrophicLevel::Producer:
    case ecology::TrophicLevel::Primary:
    case ecology::TrophicLevel::Secondary:
    case ecology::TrophicLevel::Apex: return true;
    }
    return false;
}

template <typename Enum>
requires std::is_scoped_enum_v<Enum>
constexpr bool readEnum(const char *field, core::u32 value, Enum &out, WireRefusal &outRefusal) noexcept
{
    const Enum candidate = static_cast<Enum>(value);

    if (static_cast<core::u32>(candidate) != value || !isKnown(candidate))
    {
        outRefusal = WireRefusal{field, value};
        return false;
    }
    out = candidate;
    return true;
}

template <typename Section, core::usize Count>
constexpr bool withinCounts(const Section &section, const CountBound<Section> (&bounds)[Count],
                            WireRefusal &outRefusal) noexcept
{
    for (const CountBound<Section> &bound : bounds)
    {
        if (section.*bound.member > bound.most)
        {
            outRefusal = WireRefusal{bound.field, section.*bound.member};
            return false;
        }
    }
    return true;
}

template <typename Section, core::usize Count>
constexpr bool withinReals(const Section &section, const RealBound<Section> (&bounds)[Count],
                           WireRefusal &outRefusal) noexcept
{
    for (const RealBound<Section> &bound : bounds)
    {
        const core::f32 value = section.*bound.member;
        const core::u32 bits = std::bit_cast<core::u32>(value);

        if (!isFinite(bits) || value < bound.least || value > bound.most)
        {
            outRefusal = WireRefusal{bound.field, bits};
            return false;
        }
    }
    return true;
}

template <typename Section, core::usize Count>
constexpr bool withinRaws(const Section &section, const RawBound<Section> (&bounds)[Count],
                          WireRefusal &outRefusal) noexcept
{
    for (const RawBound<Section> &bound : bounds)
    {
        const core::i32 raw = section.*bound.member;

        if (raw < bound.least || raw > bound.most)
        {
            outRefusal = WireRefusal{bound.field, static_cast<core::u32>(raw)};
            return false;
        }
    }
    return true;
}

constexpr bool withinProducts(const RecipeV1 &wire, WireRefusal &outRefusal) noexcept
{
    const core::u64 layers = wire.systemLayers < procgen::kMaxCaveLayers ? wire.systemLayers : procgen::kMaxCaveLayers;
    const core::u64 caveLevels = layers * wire.systemLevelsPerLayer;
    const core::u32 floors =
        wire.buildingMaxFloors > wire.buildingMinFloors ? wire.buildingMaxFloors : wire.buildingMinFloors;
    const core::u64 storeys = floors > 1u ? floors : 1u;
    const core::u64 storeyHeight = wire.buildingFloorHeight > 1u ? wire.buildingFloorHeight : 1u;
    const core::u64 buildingLevels =
        static_cast<core::u64>(wire.buildingBaseHeight) + storeys * storeyHeight + wire.buildingRoofHeight;
    const core::u64 aggregationSteps = static_cast<core::u64>(wire.dlaParticles) * wire.dlaMaxStepsPerParticle;

    if (caveLevels > kMaxVolumeLevels)
    {
        outRefusal = WireRefusal{"systemLayers * systemLevelsPerLayer", saturated(caveLevels)};
        return false;
    }
    if (buildingLevels > kMaxVolumeLevels)
    {
        outRefusal = WireRefusal{"buildingBaseHeight + buildingMaxFloors * buildingFloorHeight + buildingRoofHeight",
                                 saturated(buildingLevels)};
        return false;
    }
    if (aggregationSteps > kMaxAggregationSteps)
    {
        outRefusal = WireRefusal{"dlaParticles * dlaMaxStepsPerParticle", saturated(aggregationSteps)};
        return false;
    }
    return true;
}

inline bool readScatterRule(const ScatterV1 &from, procgen::ScatterRule &to, WireRefusal &outRefusal) noexcept
{
    if (!withinReals(from, kScatterReals, outRefusal) || !readEnum("scatter.biome", from.biome, to.biome, outRefusal))
        return false;
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
    return true;
}

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

inline bool readPreyIndex(core::u32 value, core::u32 speciesCount, core::u32 &out, WireRefusal &outRefusal) noexcept
{
    if (value != ecology::Species::kNoPrey && value >= speciesCount)
    {
        outRefusal = WireRefusal{"species.preyIndex", value};
        return false;
    }
    out = value;
    return true;
}

} // namespace detail

inline bool toEngineRecipe(const RecipeV1 &wire, procgen::WorldRecipe &outRecipe, WireRefusal &outRefusal) noexcept
{
    procgen::WorldRecipe recipe{};

    if (!detail::withinCounts(wire, detail::kRecipeCounts, outRefusal) ||
        !detail::withinReals(wire, detail::kRecipeReals, outRefusal) || !detail::withinProducts(wire, outRefusal))
        return false;

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
    if (!detail::readEnum("noiseKind", wire.noiseKind, recipe.terrain.kind, outRefusal))
        return false;
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
    if (!detail::readEnum("provinceMetric", wire.provinceMetric, recipe.provinces.metric, outRefusal))
        return false;

    recipe.terraceSteps = wire.terraceSteps;
    if (!detail::readEnum("caveKind", wire.caveKind, recipe.caveKind, outRefusal))
        return false;

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

    recipe.scatterCount = wire.scatterCount < kWireScatterRules ? wire.scatterCount : kWireScatterRules;
    for (core::u32 i = 0u; i < recipe.scatterCount; ++i)
        if (!detail::readScatterRule(wire.scatter[i], recipe.scatter[i], outRefusal))
            return false;

    recipe.normalizeTerrain = (wire.flags & kRecipeFlagNormalizeTerrain) != 0u;
    recipe.erodeTerrain = (wire.flags & kRecipeFlagErodeTerrain) != 0u;
    recipe.carveRivers = (wire.flags & kRecipeFlagCarveRivers) != 0u;
    recipe.classifyBiomes = (wire.flags & kRecipeFlagClassifyBiomes) != 0u;
    recipe.carveCaves = (wire.flags & kRecipeFlagCarveCaves) != 0u;
    recipe.placeSettlement = (wire.flags & kRecipeFlagPlaceSettlement) != 0u;
    recipe.materializeGround = (wire.flags & kRecipeFlagMaterializeGround) != 0u;
    recipe.growRoads = (wire.flags & kRecipeFlagGrowRoads) != 0u;
    recipe.checkPlayability = (wire.flags & kRecipeFlagCheckPlayability) != 0u;

    outRecipe = recipe;
    return true;
}

inline RecipeV1 toWireRecipe(const procgen::WorldRecipe &recipe) noexcept
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

inline bool toEngineLiving(const LivingV1 &wire, ecology::LivingRecipe &outLiving, WireRefusal &outRefusal) noexcept
{
    ecology::LivingRecipe recipe{};

    if (!detail::withinCounts(wire, detail::kLivingCounts, outRefusal) ||
        !detail::withinReals(wire, detail::kLivingReals, outRefusal))
        return false;

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
        if (!detail::withinRaws(from, detail::kSpeciesRaws, outRefusal) ||
            !detail::readEnum("species.level", from.level, to.params.level, outRefusal))
            return false;
        to.params.growth = math::Fixed32::fromRaw(from.growth);
        to.params.mortality = math::Fixed32::fromRaw(from.mortality);
        to.params.predation = math::Fixed32::fromRaw(from.predation);
        to.params.conversion = math::Fixed32::fromRaw(from.conversion);
        to.params.capacity = math::Fixed32::fromRaw(from.capacity);
        to.params.refuge = math::Fixed32::fromRaw(from.refuge);
        to.initial = math::Fixed32::fromRaw(from.initial);
        if (!detail::readPreyIndex(from.preyIndex, count, to.preyIndex, outRefusal))
            return false;
    }
    recipe.speciesCount = count;

    outLiving = recipe;
    return true;
}

inline LivingV1 toWireLiving(const ecology::LivingRecipe &recipe) noexcept
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

#endif // LPL_PACK_RECIPECODEC_INL
