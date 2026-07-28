/**
 * @file GamePackBaker.cpp
 * @brief Implementation of the `.lplscene` -> `.lplpak` oven.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/editor/GamePackBaker.hpp>

#include <lpl/editor/GameDocument.hpp>
#include <lpl/editor/Json.hpp>
#include <lpl/pack/GamePack.hpp>
#include <lpl/pack/RecipeCodec.hpp>

#include <cstdio>
#include <cstring>

namespace lpl::editor {

namespace {

/// Reads an unsigned field, keeping @p fallback when absent or not a number.
core::u32 readU32(const detail::JVal &object, const char *key, core::u32 fallback)
{
    return static_cast<core::u32>(object.numOr(key, static_cast<double>(fallback)));
}

/// Reads a float field, keeping @p fallback when absent or not a number.
core::f32 readF32(const detail::JVal &object, const char *key, core::f32 fallback)
{
    return static_cast<core::f32>(object.numOr(key, static_cast<double>(fallback)));
}

/// Reads a boolean field; JSON true/false, keeping @p fallback when absent.
bool readBool(const detail::JVal &object, const char *key, bool fallback)
{
    const detail::JVal *value = object.find(key);
    return (value != nullptr && value->t == detail::JVal::T::Bool) ? value->b : fallback;
}

/// Reads a noise kind by name ("fbm", "ridged", "billow"), else @p fallback.
procgen::NoiseKind readNoiseKind(const detail::JVal &object, const char *key, procgen::NoiseKind fallback)
{
    const detail::JVal *value = object.find(key);
    if (value == nullptr || value->t != detail::JVal::T::Str)
        return fallback;
    if (value->str == "fbm")
        return procgen::NoiseKind::Fbm;
    if (value->str == "ridged")
        return procgen::NoiseKind::Ridged;
    if (value->str == "billow")
        return procgen::NoiseKind::Billow;
    return fallback;
}

/// The name @ref readNoiseKind accepts for @p kind.
const char *noiseKindName(procgen::NoiseKind kind)
{
    switch (kind)
    {
    case procgen::NoiseKind::Ridged: return "ridged";
    case procgen::NoiseKind::Billow: return "billow";
    case procgen::NoiseKind::Fbm: break;
    }
    return "fbm";
}

/// Appends the raw bytes of a trivially-copyable value to @p out.
template <typename T> void appendPod(std::vector<core::u8> &out, const T &value)
{
    const auto *bytes = reinterpret_cast<const core::u8 *>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

/// Appends `"key":value` to a JSON object under construction.
void appendU32(std::string &out, const char *key, core::u32 value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "\"%s\":%u", key, value);
    out += buffer;
}

/**
 * @brief Appends `"key":value` for a float.
 *
 * %.9g round-trips a float exactly through decimal, so re-reading an emitted
 * document reproduces the same bits — a recipe that drifted on save/load would
 * break client/server agreement in the least visible way possible.
 */
void appendF32(std::string &out, const char *key, core::f32 value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "\"%s\":%.9g", key, static_cast<double>(value));
    out += buffer;
}

/// Appends `"key":true|false`.
void appendBool(std::string &out, const char *key, bool value)
{
    out += '"';
    out += key;
    out += "\":";
    out += value ? "true" : "false";
}

/// Reads a Fixed32 field written as a decimal number in the document.
math::Fixed32 readFixed(const detail::JVal &object, const char *key, math::Fixed32 fallback)
{
    const detail::JVal *value = object.find(key);
    if (value == nullptr || value->t != detail::JVal::T::Num)
        return fallback;
    return math::Fixed32::fromFloat(static_cast<core::f32>(value->num));
}

/// Reads a trophic level by name, else @p fallback.
ecology::TrophicLevel readTrophicLevel(const detail::JVal &object, const char *key, ecology::TrophicLevel fallback)
{
    const detail::JVal *value = object.find(key);
    if (value == nullptr || value->t != detail::JVal::T::Str)
        return fallback;
    if (value->str == "producer")
        return ecology::TrophicLevel::Producer;
    if (value->str == "primary" || value->str == "herbivore")
        return ecology::TrophicLevel::Primary;
    if (value->str == "secondary" || value->str == "predator")
        return ecology::TrophicLevel::Secondary;
    if (value->str == "apex")
        return ecology::TrophicLevel::Apex;
    return fallback;
}

const char *trophicLevelName(ecology::TrophicLevel level)
{
    switch (level)
    {
    case ecology::TrophicLevel::Producer: return "producer";
    case ecology::TrophicLevel::Primary: return "primary";
    case ecology::TrophicLevel::Secondary: return "secondary";
    case ecology::TrophicLevel::Apex: return "apex";
    }
    return "primary";
}

} // namespace

bool parseSceneLiving(const detail::JVal &scene, ecology::LivingRecipe &outLiving)
{
    const detail::JVal *living = scene.find("living");
    if (living == nullptr || living->t != detail::JVal::T::Obj)
        return false;

    // Start from the engine defaults, so a document states only what it changes —
    // the same rule the procedural block follows, and the reason a default that
    // moves moves both sides together instead of silently splitting them.
    ecology::LivingRecipe recipe{};

    recipe.seed = readU32(*living, "seed", recipe.seed);
    recipe.ticks = readU32(*living, "ticks", recipe.ticks);
    recipe.width = readU32(*living, "width", recipe.width);
    recipe.depth = readU32(*living, "depth", recipe.depth);
    recipe.channels = readU32(*living, "channels", recipe.channels);
    recipe.rooms = readU32(*living, "rooms", recipe.rooms);
    recipe.creatures = readU32(*living, "creatures", recipe.creatures);
    recipe.ants = readU32(*living, "ants", recipe.ants);
    recipe.boids = readU32(*living, "boids", recipe.boids);
    recipe.genomes = readU32(*living, "genomes", recipe.genomes);
    recipe.packMembers = readU32(*living, "packMembers", recipe.packMembers);
    recipe.regrowthTicks = readU32(*living, "regrowthTicks", recipe.regrowthTicks);
    recipe.headPerBody = readU32(*living, "headPerBody", recipe.headPerBody);

    if (const detail::JVal *field = living->find("stigmergy"); field != nullptr && field->t == detail::JVal::T::Obj)
    {
        recipe.stigmergy.evaporation = readF32(*field, "evaporation", recipe.stigmergy.evaporation);
        recipe.stigmergy.diffusion = readF32(*field, "diffusion", recipe.stigmergy.diffusion);
        recipe.stigmergy.maximum = readF32(*field, "maximum", recipe.stigmergy.maximum);
        recipe.stigmergy.floor = readF32(*field, "floor", recipe.stigmergy.floor);
    }

    if (const detail::JVal *flock = living->find("flock"); flock != nullptr && flock->t == detail::JVal::T::Obj)
    {
        recipe.flock.separationRadius = readFixed(*flock, "separationRadius", recipe.flock.separationRadius);
        recipe.flock.neighbourRadius = readFixed(*flock, "neighbourRadius", recipe.flock.neighbourRadius);
        recipe.flock.separationWeight = readF32(*flock, "separation", recipe.flock.separationWeight);
        recipe.flock.alignmentWeight = readF32(*flock, "alignment", recipe.flock.alignmentWeight);
        recipe.flock.cohesionWeight = readF32(*flock, "cohesion", recipe.flock.cohesionWeight);
        recipe.flock.maxSpeed = readF32(*flock, "maxSpeed", recipe.flock.maxSpeed);
    }

    if (const detail::JVal *heredity = living->find("heredity");
        heredity != nullptr && heredity->t == detail::JVal::T::Obj)
    {
        recipe.heredity.mutationChance16 = readU32(*heredity, "mutationChance16", recipe.heredity.mutationChance16);
        recipe.heredity.mutationAmplitude = readF32(*heredity, "mutationAmplitude", recipe.heredity.mutationAmplitude);
        recipe.heredity.collapseShare16 = readU32(*heredity, "collapseShare16", recipe.heredity.collapseShare16);
        recipe.heredity.meltdownChance16 = readU32(*heredity, "meltdownChance16", recipe.heredity.meltdownChance16);
        recipe.heredity.meltdownAmplitude = readF32(*heredity, "meltdownAmplitude", recipe.heredity.meltdownAmplitude);
        recipe.heredity.anomalySigma = readF32(*heredity, "anomalySigma", recipe.heredity.anomalySigma);
    }

    if (const detail::JVal *packs = living->find("packs"); packs != nullptr && packs->t == detail::JVal::T::Obj)
    {
        recipe.packs.maxSize = readU32(*packs, "maxSize", recipe.packs.maxSize);
        recipe.packs.minSize = readU32(*packs, "minSize", recipe.packs.minSize);
        recipe.packs.dissolutionChance16 = readU32(*packs, "dissolutionChance16", recipe.packs.dissolutionChance16);
        recipe.packs.adoptStrays = readBool(*packs, "adoptStrays", recipe.packs.adoptStrays);
    }

    if (const detail::JVal *budget = living->find("budget"); budget != nullptr && budget->t == detail::JVal::T::Obj)
    {
        recipe.budget.maxRealisedRooms = readU32(*budget, "maxRealisedRooms", recipe.budget.maxRealisedRooms);
        recipe.budget.changeWeight = readU32(*budget, "changeWeight", recipe.budget.changeWeight);
        recipe.budget.adjacentBonus = readU32(*budget, "adjacentBonus", recipe.budget.adjacentBonus);
        recipe.budget.predictedBonus = readU32(*budget, "predictedBonus", recipe.budget.predictedBonus);
        recipe.budget.unlikelyPenalty = readU32(*budget, "unlikelyPenalty", recipe.budget.unlikelyPenalty);
    }

    // The food web, as an ordered table: "eats" is an INDEX into it, so a species
    // can only eat something declared before it and a cycle is unwritable.
    if (const detail::JVal *species = living->find("species"); species != nullptr && species->t == detail::JVal::T::Arr)
    {
        core::u32 count = 0u;
        for (const detail::JVal &entry : species->arr)
        {
            if (count >= ecology::kMaxLivingSpecies || entry.t != detail::JVal::T::Obj)
                break;
            ecology::LivingSpecies &slot = recipe.species[count];
            slot.params.level = readTrophicLevel(entry, "level", slot.params.level);
            slot.params.growth = readFixed(entry, "growth", slot.params.growth);
            slot.params.mortality = readFixed(entry, "mortality", slot.params.mortality);
            slot.params.predation = readFixed(entry, "predation", slot.params.predation);
            slot.params.conversion = readFixed(entry, "conversion", slot.params.conversion);
            slot.params.capacity = readFixed(entry, "capacity", slot.params.capacity);
            slot.params.refuge = readFixed(entry, "refuge", slot.params.refuge);
            slot.initial = readFixed(entry, "initial", slot.initial);

            const detail::JVal *eats = entry.find("eats");
            slot.preyIndex = (eats != nullptr && eats->t == detail::JVal::T::Num) ? static_cast<core::u32>(eats->num) :
                                                                                    ecology::Species::kNoPrey;
            ++count;
        }
        recipe.speciesCount = count;
    }

    outLiving = recipe;
    return true;
}

core::ExpectedVoid parseSceneRecipe(std::string_view document, procgen::WorldRecipe &outRecipe)
{
    detail::Parser parser{document, 0, true};
    const detail::JVal root = parser.value();
    if (!parser.ok || root.t != detail::JVal::T::Obj)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"malformed .lplscene root"});

    const detail::JVal *format = root.find("format");
    if (format == nullptr || format->t != detail::JVal::T::Str || format->str != "lplscene/1")
        return core::makeError(core::ErrorCode::kNotSupported, lpl::pmr::string{"unsupported .lplscene format"});

    const detail::JVal *procedural = root.find("procedural");
    if (procedural == nullptr || procedural->t != detail::JVal::T::Obj)
        return core::makeError(core::ErrorCode::kNotFound,
                               lpl::pmr::string{"document carries no \"procedural\" block"});

    // Start from the engine defaults so a document only states what it changes.
    procgen::WorldRecipe recipe{};

    recipe.seed = readU32(*procedural, "seed", recipe.seed);
    recipe.width = readU32(*procedural, "width", recipe.width);
    recipe.depth = readU32(*procedural, "depth", recipe.depth);
    recipe.cellSize = readF32(*procedural, "cellSize", recipe.cellSize);
    recipe.materializeGround = readBool(*procedural, "materializeGround", recipe.materializeGround);

    if (const detail::JVal *terrain = procedural->find("terrain");
        terrain != nullptr && terrain->t == detail::JVal::T::Obj)
    {
        recipe.terrain.seed = readU32(*terrain, "seed", recipe.terrain.seed);
        recipe.terrain.frequency = readF32(*terrain, "frequency", recipe.terrain.frequency);
        recipe.terrain.amplitude = readF32(*terrain, "amplitude", recipe.terrain.amplitude);
        recipe.terrain.octaves = readU32(*terrain, "octaves", recipe.terrain.octaves);
        recipe.terrain.baseHeight = readF32(*terrain, "baseHeight", recipe.terrain.baseHeight);
        recipe.terrain.lacunarity = readF32(*terrain, "lacunarity", recipe.terrain.lacunarity);
        recipe.terrain.persistence = readF32(*terrain, "persistence", recipe.terrain.persistence);
        recipe.terrain.warpStrength = readF32(*terrain, "warpStrength", recipe.terrain.warpStrength);
        recipe.terrain.kind = readNoiseKind(*terrain, "kind", recipe.terrain.kind);
        recipe.heightLow = readF32(*terrain, "low", recipe.heightLow);
        recipe.heightHigh = readF32(*terrain, "high", recipe.heightHigh);
        recipe.normalizeTerrain = readBool(*terrain, "normalize", recipe.normalizeTerrain);
    }

    if (const detail::JVal *erosion = procedural->find("erosion");
        erosion != nullptr && erosion->t == detail::JVal::T::Obj)
    {
        recipe.erodeTerrain = readBool(*erosion, "enabled", recipe.erodeTerrain);
        recipe.thermal.iterations = readU32(*erosion, "thermalIterations", recipe.thermal.iterations);
        recipe.thermal.talus = readF32(*erosion, "talus", recipe.thermal.talus);
        recipe.thermal.carryFraction = readF32(*erosion, "carryFraction", recipe.thermal.carryFraction);
        recipe.hydraulic.iterations = readU32(*erosion, "hydraulicIterations", recipe.hydraulic.iterations);
        recipe.hydraulic.rainAmount = readF32(*erosion, "rainAmount", recipe.hydraulic.rainAmount);
        recipe.hydraulic.solubility = readF32(*erosion, "solubility", recipe.hydraulic.solubility);
        recipe.hydraulic.evaporation = readF32(*erosion, "evaporation", recipe.hydraulic.evaporation);
        recipe.hydraulic.sedimentCapacity = readF32(*erosion, "sedimentCapacity", recipe.hydraulic.sedimentCapacity);
        recipe.hydraulic.deposition = readF32(*erosion, "deposition", recipe.hydraulic.deposition);
        recipe.hydraulic.minSlope = readF32(*erosion, "minSlope", recipe.hydraulic.minSlope);
    }

    if (const detail::JVal *rivers = procedural->find("rivers"); rivers != nullptr && rivers->t == detail::JVal::T::Obj)
    {
        recipe.carveRivers = readBool(*rivers, "enabled", recipe.carveRivers);
        recipe.rivers.density = readF32(*rivers, "density", recipe.rivers.density);
        recipe.rivers.carveDepth = readF32(*rivers, "carveDepth", recipe.rivers.carveDepth);
        recipe.rivers.smoothing = readU32(*rivers, "smoothing", recipe.rivers.smoothing);
    }

    if (const detail::JVal *climate = procedural->find("climate");
        climate != nullptr && climate->t == detail::JVal::T::Obj)
    {
        recipe.classifyBiomes = readBool(*climate, "enabled", recipe.classifyBiomes);
        recipe.climate.rainfallWeight = readF32(*climate, "rainfallWeight", recipe.climate.rainfallWeight);
        recipe.climate.flowWeight = readF32(*climate, "flowWeight", recipe.climate.flowWeight);
        recipe.climate.altitudeWeight = readF32(*climate, "altitudeWeight", recipe.climate.altitudeWeight);
        recipe.climate.coastWeight = readF32(*climate, "coastWeight", recipe.climate.coastWeight);
        recipe.climate.seaLevel = readF32(*climate, "seaLevel", recipe.climate.seaLevel);
        recipe.climate.coastReach = readF32(*climate, "coastReach", recipe.climate.coastReach);
        recipe.climate.rainShadow = readF32(*climate, "rainShadow", recipe.climate.rainShadow);
        recipe.climate.windDirection = readU32(*climate, "windDirection", recipe.climate.windDirection);
        recipe.climate.smoothing = readU32(*climate, "smoothing", recipe.climate.smoothing);
        recipe.climate.rainfallSeed = readU32(*climate, "rainfallSeed", recipe.climate.rainfallSeed);
        recipe.climate.rainfallBelts = readF32(*climate, "rainfallBelts", recipe.climate.rainfallBelts);
        recipe.climate.rainfallOctaves = readU32(*climate, "rainfallOctaves", recipe.climate.rainfallOctaves);
    }

    if (const detail::JVal *biomes = procedural->find("biomes"); biomes != nullptr && biomes->t == detail::JVal::T::Obj)
    {
        recipe.biomes.seaLevel = readF32(*biomes, "seaLevel", recipe.biomes.seaLevel);
        recipe.biomes.beachHeight = readF32(*biomes, "beachHeight", recipe.biomes.beachHeight);
        recipe.biomes.mountainHeight = readF32(*biomes, "mountainHeight", recipe.biomes.mountainHeight);
        recipe.biomes.snowHeight = readF32(*biomes, "snowHeight", recipe.biomes.snowHeight);
        recipe.biomes.snowlineWarmth = readF32(*biomes, "snowlineWarmth", recipe.biomes.snowlineWarmth);
    }

    if (const detail::JVal *axes = procedural->find("climateAxes"); axes != nullptr && axes->t == detail::JVal::T::Obj)
    {
        recipe.axes.coldLatitude = readF32(*axes, "coldLatitude", recipe.axes.coldLatitude);
        recipe.axes.lapseRate = readF32(*axes, "lapseRate", recipe.axes.lapseRate);
        recipe.axes.coastReach = readF32(*axes, "coastReach", recipe.axes.coastReach);
        recipe.axes.weirdnessSeed = readU32(*axes, "weirdnessSeed", recipe.axes.weirdnessSeed);
        recipe.axes.weirdnessBelts = readF32(*axes, "weirdnessBelts", recipe.axes.weirdnessBelts);
        recipe.axes.weirdnessOctaves = readU32(*axes, "weirdnessOctaves", recipe.axes.weirdnessOctaves);
        recipe.axes.surfaceDepth = readF32(*axes, "surfaceDepth", recipe.axes.surfaceDepth);
    }

    if (const detail::JVal *caves = procedural->find("caves"); caves != nullptr && caves->t == detail::JVal::T::Obj)
    {
        recipe.carveCaves = readBool(*caves, "enabled", recipe.carveCaves);
        recipe.caves.width = readU32(*caves, "width", recipe.caves.width);
        recipe.caves.depth = readU32(*caves, "depth", recipe.caves.depth);
        recipe.caves.seed = readU32(*caves, "seed", recipe.caves.seed);
        recipe.caves.fillProbability = readF32(*caves, "fillProbability", recipe.caves.fillProbability);
        recipe.caves.steps = readU32(*caves, "steps", recipe.caves.steps);
        recipe.caves.birthLimit = readU32(*caves, "birthLimit", recipe.caves.birthLimit);
        recipe.caves.survivalLimit = readU32(*caves, "survivalLimit", recipe.caves.survivalLimit);
        recipe.caves.minRegionSize = readU32(*caves, "minRegionSize", recipe.caves.minRegionSize);
    }

    if (const detail::JVal *town = procedural->find("settlement"); town != nullptr && town->t == detail::JVal::T::Obj)
    {
        recipe.placeSettlement = readBool(*town, "enabled", recipe.placeSettlement);
        recipe.settlement.seed = readU32(*town, "seed", recipe.settlement.seed);
        recipe.settlement.districtSize = readU32(*town, "districtSize", recipe.settlement.districtSize);
        recipe.settlement.roadWidth = readU32(*town, "roadWidth", recipe.settlement.roadWidth);
        recipe.settlement.plazaRadius = readU32(*town, "plazaRadius", recipe.settlement.plazaRadius);
        recipe.settlement.minPlot = readU32(*town, "minPlot", recipe.settlement.minPlot);
        recipe.settlement.maxPlot = readU32(*town, "maxPlot", recipe.settlement.maxPlot);
        recipe.settlement.plotDensity = readF32(*town, "plotDensity", recipe.settlement.plotDensity);
        recipe.settlement.maxSlope = readF32(*town, "maxSlope", recipe.settlement.maxSlope);
        recipe.settlement.minHeight = readF32(*town, "minHeight", recipe.settlement.minHeight);
    }

    if (const detail::JVal *roads = procedural->find("roads"); roads != nullptr && roads->t == detail::JVal::T::Obj)
    {
        recipe.growRoads = readBool(*roads, "enabled", recipe.growRoads);
        recipe.roads.seed = readU32(*roads, "seed", recipe.roads.seed);
        recipe.roads.iterations = readU32(*roads, "iterations", recipe.roads.iterations);
        recipe.roads.stepLength = readU32(*roads, "stepLength", recipe.roads.stepLength);
        recipe.roads.conform = readF32(*roads, "conform", recipe.roads.conform);
        recipe.roads.maxSlope = readF32(*roads, "maxSlope", recipe.roads.maxSlope);
        recipe.roads.minHeight = readF32(*roads, "minHeight", recipe.roads.minHeight);
        recipe.roads.gridDistricts = readU32(*roads, "gridDistricts", recipe.roads.gridDistricts);
        recipe.roads.arterials = readBool(*roads, "arterials", recipe.roads.arterials);
    }

    if (const detail::JVal *gate = procedural->find("gate"); gate != nullptr && gate->t == detail::JVal::T::Obj)
    {
        recipe.checkPlayability = readBool(*gate, "enabled", recipe.checkPlayability);
        recipe.gate.minPathLength = readU32(*gate, "minPathLength", recipe.gate.minPathLength);
        recipe.gate.minWalkableCells = readU32(*gate, "minWalkableCells", recipe.gate.minWalkableCells);
        recipe.gate.maxDeadEndRatio = readU32(*gate, "maxDeadEndRatio", recipe.gate.maxDeadEndRatio);
        recipe.gate.requireGoalReachable = readBool(*gate, "requireGoal", recipe.gate.requireGoalReachable);
        recipe.gate.requireFullyConnected = readBool(*gate, "requireConnected", recipe.gate.requireFullyConnected);
    }

    if (const detail::JVal *scatter = procedural->find("scatter");
        scatter != nullptr && scatter->t == detail::JVal::T::Arr)
    {
        recipe.scatterCount = 0u;
        for (const detail::JVal &rule : scatter->arr)
        {
            if (rule.t != detail::JVal::T::Obj || recipe.scatterCount >= procgen::kMaxScatterRules)
                break;
            procgen::ScatterRule &to = recipe.scatter[recipe.scatterCount];
            if (const detail::JVal *biome = rule.find("biome"); biome != nullptr && biome->t == detail::JVal::T::Str)
            {
                const procgen::BiomeId resolved = procgen::biomeIdByName(biome->str.c_str());
                if (resolved == procgen::BiomeId::Count)
                    return core::makeError(core::ErrorCode::kDeserializationFailed,
                                           lpl::pmr::string{"unknown biome in scatter rule"});
                to.biome = resolved;
            }
            to.density = readF32(rule, "density", to.density);
            to.halfExtent = readF32(rule, "halfExtent", to.halfExtent);
            to.maxSlope = readF32(rule, "maxSlope", to.maxSlope);
            to.minMoisture = readF32(rule, "minMoisture", to.minMoisture);
            to.maxMoisture = readF32(rule, "maxMoisture", to.maxMoisture);
            to.moistureAffinity = readF32(rule, "moistureAffinity", to.moistureAffinity);
            to.treeLine = readF32(rule, "treeLine", to.treeLine);
            to.altitudeFalloff = readF32(rule, "altitudeFalloff", to.altitudeFalloff);
            to.maxRiverDistance = readU32(rule, "maxRiverDistance", to.maxRiverDistance);
            to.endemicShare = readF32(rule, "endemicShare", to.endemicShare);
            to.tag = readU32(rule, "tag", to.tag);
            to.collidable = readBool(rule, "collidable", to.collidable);
            ++recipe.scatterCount;
        }
    }

    outRecipe = recipe;
    return {};
}

std::string emitSceneRecipe(const procgen::WorldRecipe &recipe)
{
    std::string out = "{";
    appendU32(out, "seed", recipe.seed);
    out += ',';
    appendU32(out, "width", recipe.width);
    out += ',';
    appendU32(out, "depth", recipe.depth);
    out += ',';
    appendF32(out, "cellSize", recipe.cellSize);
    out += ',';
    appendBool(out, "materializeGround", recipe.materializeGround);

    out += ",\"terrain\":{";
    appendBool(out, "normalize", recipe.normalizeTerrain);
    out += ',';
    appendU32(out, "seed", recipe.terrain.seed);
    out += ',';
    appendF32(out, "frequency", recipe.terrain.frequency);
    out += ',';
    appendF32(out, "amplitude", recipe.terrain.amplitude);
    out += ',';
    appendU32(out, "octaves", recipe.terrain.octaves);
    out += ',';
    appendF32(out, "baseHeight", recipe.terrain.baseHeight);
    out += ',';
    appendF32(out, "lacunarity", recipe.terrain.lacunarity);
    out += ',';
    appendF32(out, "persistence", recipe.terrain.persistence);
    out += ',';
    appendF32(out, "warpStrength", recipe.terrain.warpStrength);
    out += ",\"kind\":\"";
    out += noiseKindName(recipe.terrain.kind);
    out += "\",";
    appendF32(out, "low", recipe.heightLow);
    out += ',';
    appendF32(out, "high", recipe.heightHigh);
    out += '}';

    out += ",\"erosion\":{";
    appendBool(out, "enabled", recipe.erodeTerrain);
    out += ',';
    appendU32(out, "thermalIterations", recipe.thermal.iterations);
    out += ',';
    appendF32(out, "talus", recipe.thermal.talus);
    out += ',';
    appendF32(out, "carryFraction", recipe.thermal.carryFraction);
    out += ',';
    appendU32(out, "hydraulicIterations", recipe.hydraulic.iterations);
    out += ',';
    appendF32(out, "rainAmount", recipe.hydraulic.rainAmount);
    out += ',';
    appendF32(out, "solubility", recipe.hydraulic.solubility);
    out += ',';
    appendF32(out, "evaporation", recipe.hydraulic.evaporation);
    out += ',';
    appendF32(out, "sedimentCapacity", recipe.hydraulic.sedimentCapacity);
    out += ',';
    appendF32(out, "deposition", recipe.hydraulic.deposition);
    out += ',';
    appendF32(out, "minSlope", recipe.hydraulic.minSlope);
    out += '}';

    out += ",\"rivers\":{";
    appendBool(out, "enabled", recipe.carveRivers);
    out += ',';
    appendF32(out, "density", recipe.rivers.density);
    out += ',';
    appendF32(out, "carveDepth", recipe.rivers.carveDepth);
    out += ',';
    appendU32(out, "smoothing", recipe.rivers.smoothing);
    out += '}';

    out += ",\"climate\":{";
    appendBool(out, "enabled", recipe.classifyBiomes);
    out += ',';
    appendF32(out, "rainfallWeight", recipe.climate.rainfallWeight);
    out += ',';
    appendF32(out, "flowWeight", recipe.climate.flowWeight);
    out += ',';
    appendF32(out, "altitudeWeight", recipe.climate.altitudeWeight);
    out += ',';
    appendF32(out, "coastWeight", recipe.climate.coastWeight);
    out += ',';
    appendF32(out, "seaLevel", recipe.climate.seaLevel);
    out += ',';
    appendF32(out, "coastReach", recipe.climate.coastReach);
    out += ',';
    appendF32(out, "rainShadow", recipe.climate.rainShadow);
    out += ',';
    appendU32(out, "windDirection", recipe.climate.windDirection);
    out += ',';
    appendU32(out, "smoothing", recipe.climate.smoothing);
    out += ',';
    appendU32(out, "rainfallSeed", recipe.climate.rainfallSeed);
    out += ',';
    appendF32(out, "rainfallBelts", recipe.climate.rainfallBelts);
    out += ',';
    appendU32(out, "rainfallOctaves", recipe.climate.rainfallOctaves);
    out += '}';

    out += ",\"biomes\":{";
    appendF32(out, "seaLevel", recipe.biomes.seaLevel);
    out += ',';
    appendF32(out, "beachHeight", recipe.biomes.beachHeight);
    out += ',';
    appendF32(out, "mountainHeight", recipe.biomes.mountainHeight);
    out += ',';
    appendF32(out, "snowHeight", recipe.biomes.snowHeight);
    out += ',';
    appendF32(out, "snowlineWarmth", recipe.biomes.snowlineWarmth);
    out += '}';

    out += ",\"climateAxes\":{";
    appendF32(out, "coldLatitude", recipe.axes.coldLatitude);
    out += ',';
    appendF32(out, "lapseRate", recipe.axes.lapseRate);
    out += ',';
    appendF32(out, "coastReach", recipe.axes.coastReach);
    out += ',';
    appendU32(out, "weirdnessSeed", recipe.axes.weirdnessSeed);
    out += ',';
    appendF32(out, "weirdnessBelts", recipe.axes.weirdnessBelts);
    out += ',';
    appendU32(out, "weirdnessOctaves", recipe.axes.weirdnessOctaves);
    out += ',';
    appendF32(out, "surfaceDepth", recipe.axes.surfaceDepth);
    out += '}';

    out += ",\"caves\":{";
    appendBool(out, "enabled", recipe.carveCaves);
    out += ',';
    appendU32(out, "width", recipe.caves.width);
    out += ',';
    appendU32(out, "depth", recipe.caves.depth);
    out += ',';
    appendU32(out, "seed", recipe.caves.seed);
    out += ',';
    appendF32(out, "fillProbability", recipe.caves.fillProbability);
    out += ',';
    appendU32(out, "steps", recipe.caves.steps);
    out += ',';
    appendU32(out, "birthLimit", recipe.caves.birthLimit);
    out += ',';
    appendU32(out, "survivalLimit", recipe.caves.survivalLimit);
    out += ',';
    appendU32(out, "minRegionSize", recipe.caves.minRegionSize);
    out += '}';

    out += ",\"settlement\":{";
    appendBool(out, "enabled", recipe.placeSettlement);
    out += ',';
    appendU32(out, "seed", recipe.settlement.seed);
    out += ',';
    appendU32(out, "districtSize", recipe.settlement.districtSize);
    out += ',';
    appendU32(out, "roadWidth", recipe.settlement.roadWidth);
    out += ',';
    appendU32(out, "plazaRadius", recipe.settlement.plazaRadius);
    out += ',';
    appendU32(out, "minPlot", recipe.settlement.minPlot);
    out += ',';
    appendU32(out, "maxPlot", recipe.settlement.maxPlot);
    out += ',';
    appendF32(out, "plotDensity", recipe.settlement.plotDensity);
    out += ',';
    appendF32(out, "maxSlope", recipe.settlement.maxSlope);
    out += ',';
    appendF32(out, "minHeight", recipe.settlement.minHeight);
    out += '}';

    out += ",\"roads\":{";
    appendBool(out, "enabled", recipe.growRoads);
    out += ',';
    appendU32(out, "seed", recipe.roads.seed);
    out += ',';
    appendU32(out, "iterations", recipe.roads.iterations);
    out += ',';
    appendU32(out, "stepLength", recipe.roads.stepLength);
    out += ',';
    appendF32(out, "conform", recipe.roads.conform);
    out += ',';
    appendF32(out, "maxSlope", recipe.roads.maxSlope);
    out += ',';
    appendF32(out, "minHeight", recipe.roads.minHeight);
    out += ',';
    appendU32(out, "gridDistricts", recipe.roads.gridDistricts);
    out += ',';
    appendBool(out, "arterials", recipe.roads.arterials);
    out += '}';

    out += ",\"gate\":{";
    appendBool(out, "enabled", recipe.checkPlayability);
    out += ',';
    appendU32(out, "minPathLength", recipe.gate.minPathLength);
    out += ',';
    appendU32(out, "minWalkableCells", recipe.gate.minWalkableCells);
    out += ',';
    appendU32(out, "maxDeadEndRatio", recipe.gate.maxDeadEndRatio);
    out += ',';
    appendBool(out, "requireGoal", recipe.gate.requireGoalReachable);
    out += ',';
    appendBool(out, "requireConnected", recipe.gate.requireFullyConnected);
    out += '}';

    out += ",\"scatter\":[";
    for (core::u32 i = 0u; i < recipe.scatterCount && i < procgen::kMaxScatterRules; ++i)
    {
        const procgen::ScatterRule &rule = recipe.scatter[i];
        if (i != 0u)
            out += ',';
        out += "{\"biome\":\"";
        out += procgen::biomeName(rule.biome);
        out += "\",";
        appendF32(out, "density", rule.density);
        out += ',';
        appendF32(out, "halfExtent", rule.halfExtent);
        out += ',';
        appendF32(out, "maxSlope", rule.maxSlope);
        out += ',';
        appendF32(out, "minMoisture", rule.minMoisture);
        out += ',';
        appendF32(out, "maxMoisture", rule.maxMoisture);
        out += ',';
        appendF32(out, "moistureAffinity", rule.moistureAffinity);
        out += ',';
        appendF32(out, "treeLine", rule.treeLine);
        out += ',';
        appendF32(out, "altitudeFalloff", rule.altitudeFalloff);
        out += ',';
        appendU32(out, "maxRiverDistance", rule.maxRiverDistance);
        out += ',';
        appendF32(out, "endemicShare", rule.endemicShare);
        out += ',';
        appendU32(out, "tag", rule.tag);
        out += ',';
        appendBool(out, "collidable", rule.collidable);
        out += '}';
    }
    out += "]}";
    return out;
}

std::string emitSceneLiving(const ecology::LivingRecipe &living)
{
    // Every field, not only the ones that differ from the defaults: a document
    // that emits a subset is not a round trip, it is a lossy save that looks like
    // one until a default moves under it.
    std::string out = "{";
    appendU32(out, "seed", living.seed);
    out += ',';
    appendU32(out, "ticks", living.ticks);
    out += ',';
    appendU32(out, "width", living.width);
    out += ',';
    appendU32(out, "depth", living.depth);
    out += ',';
    appendU32(out, "channels", living.channels);
    out += ',';
    appendU32(out, "rooms", living.rooms);
    out += ',';
    appendU32(out, "creatures", living.creatures);
    out += ',';
    appendU32(out, "ants", living.ants);
    out += ',';
    appendU32(out, "boids", living.boids);
    out += ',';
    appendU32(out, "genomes", living.genomes);
    out += ',';
    appendU32(out, "packMembers", living.packMembers);
    out += ',';
    appendU32(out, "regrowthTicks", living.regrowthTicks);
    out += ',';
    appendU32(out, "headPerBody", living.headPerBody);

    out += ",\"stigmergy\":{";
    appendF32(out, "evaporation", living.stigmergy.evaporation);
    out += ',';
    appendF32(out, "diffusion", living.stigmergy.diffusion);
    out += ',';
    appendF32(out, "maximum", living.stigmergy.maximum);
    out += ',';
    appendF32(out, "floor", living.stigmergy.floor);
    out += '}';

    out += ",\"flock\":{";
    appendF32(out, "separationRadius", living.flock.separationRadius.toFloat());
    out += ',';
    appendF32(out, "neighbourRadius", living.flock.neighbourRadius.toFloat());
    out += ',';
    appendF32(out, "separation", living.flock.separationWeight);
    out += ',';
    appendF32(out, "alignment", living.flock.alignmentWeight);
    out += ',';
    appendF32(out, "cohesion", living.flock.cohesionWeight);
    out += ',';
    appendF32(out, "maxSpeed", living.flock.maxSpeed);
    out += '}';

    out += ",\"heredity\":{";
    appendU32(out, "mutationChance16", living.heredity.mutationChance16);
    out += ',';
    appendF32(out, "mutationAmplitude", living.heredity.mutationAmplitude);
    out += ',';
    appendU32(out, "collapseShare16", living.heredity.collapseShare16);
    out += ',';
    appendU32(out, "meltdownChance16", living.heredity.meltdownChance16);
    out += ',';
    appendF32(out, "meltdownAmplitude", living.heredity.meltdownAmplitude);
    out += ',';
    appendF32(out, "anomalySigma", living.heredity.anomalySigma);
    out += '}';

    out += ",\"packs\":{";
    appendU32(out, "maxSize", living.packs.maxSize);
    out += ',';
    appendU32(out, "minSize", living.packs.minSize);
    out += ',';
    appendU32(out, "dissolutionChance16", living.packs.dissolutionChance16);
    out += ',';
    appendBool(out, "adoptStrays", living.packs.adoptStrays);
    out += '}';

    out += ",\"budget\":{";
    appendU32(out, "maxRealisedRooms", living.budget.maxRealisedRooms);
    out += ',';
    appendU32(out, "changeWeight", living.budget.changeWeight);
    out += ',';
    appendU32(out, "adjacentBonus", living.budget.adjacentBonus);
    out += ',';
    appendU32(out, "predictedBonus", living.budget.predictedBonus);
    out += ',';
    appendU32(out, "unlikelyPenalty", living.budget.unlikelyPenalty);
    out += '}';

    out += ",\"species\":[";
    const core::u32 count =
        living.speciesCount < ecology::kMaxLivingSpecies ? living.speciesCount : ecology::kMaxLivingSpecies;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const ecology::LivingSpecies &species = living.species[i];
        if (i != 0u)
            out += ',';
        out += "{\"level\":\"";
        out += trophicLevelName(species.params.level);
        out += "\",";
        appendF32(out, "growth", species.params.growth.toFloat());
        out += ',';
        appendF32(out, "mortality", species.params.mortality.toFloat());
        out += ',';
        appendF32(out, "predation", species.params.predation.toFloat());
        out += ',';
        appendF32(out, "conversion", species.params.conversion.toFloat());
        out += ',';
        appendF32(out, "capacity", species.params.capacity.toFloat());
        out += ',';
        appendF32(out, "refuge", species.params.refuge.toFloat());
        out += ',';
        appendF32(out, "initial", species.initial.toFloat());
        if (species.preyIndex != ecology::Species::kNoPrey)
        {
            out += ',';
            appendU32(out, "eats", species.preyIndex);
        }
        out += '}';
    }
    out += "]}";
    return out;
}

std::vector<core::u8> bakeGamePack(const procgen::WorldRecipe &recipe) { return bakeGamePack(recipe, nullptr); }

std::vector<core::u8> bakeGamePack(const procgen::WorldRecipe &recipe, const ecology::LivingRecipe *living)
{
    const pack::RecipeV1 wire = pack::toWireRecipe(recipe);

    // One section or two. A world with nothing declared living on it stays a
    // one-section pack, byte for byte what it was before this existed — which is
    // what keeps every cartridge baked so far valid, and the parity gate's own
    // image unchanged.
    const core::u32 sectionCount = living != nullptr ? 2u : 1u;
    constexpr core::u32 kHeaderBytes = static_cast<core::u32>(sizeof(pack::Header));
    const core::u32 tableBytes = sectionCount * static_cast<core::u32>(sizeof(pack::SectionEntry));
    const core::u32 recipeOffset = kHeaderBytes + tableBytes;
    const core::u32 livingOffset = recipeOffset + static_cast<core::u32>(sizeof(pack::RecipeV1));
    const core::u32 totalSize =
        livingOffset + (living != nullptr ? static_cast<core::u32>(sizeof(pack::LivingV1)) : 0u);

    // Build the content first: the header carries a hash over everything that
    // follows it, so it can only be finalised once the content exists.
    std::vector<core::u8> content;
    content.reserve(totalSize - kHeaderBytes);

    pack::SectionEntry entry{};
    entry.type = static_cast<core::u32>(pack::SectionType::WorldRecipe);
    entry.offset = recipeOffset;
    entry.size = static_cast<core::u32>(sizeof(pack::RecipeV1));
    entry.reserved = 0u;
    appendPod(content, entry);

    if (living != nullptr)
    {
        pack::SectionEntry livingEntry{};
        livingEntry.type = static_cast<core::u32>(pack::SectionType::LivingRecipe);
        livingEntry.offset = livingOffset;
        livingEntry.size = static_cast<core::u32>(sizeof(pack::LivingV1));
        livingEntry.reserved = 0u;
        appendPod(content, livingEntry);
    }

    // Payloads in table order, so an offset is always ahead of the entry naming it.
    appendPod(content, wire);
    if (living != nullptr)
        appendPod(content, pack::toWireLiving(*living));

    pack::Header header{};
    std::memcpy(header.magic, "LPLPAK\0\0", pack::kMagicSize);
    header.formatVersion = pack::kFormatVersion;
    header.totalSize = totalSize;
    header.sectionCount = sectionCount;
    header.contentHash = pack::hashBytes(content.data(), static_cast<core::u32>(content.size()));
    header.reserved0 = 0u;
    header.reserved1 = 0u;

    std::vector<core::u8> image;
    image.reserve(totalSize);
    appendPod(image, header);
    image.insert(image.end(), content.begin(), content.end());
    return image;
}

core::Expected<std::vector<core::u8>> bakeSceneDocument(std::string_view document)
{
    // A four-stage document holds its recipe inside the scene it starts on, so
    // go through the document model first. Falling back to the flat form keeps
    // every document written before that stage existed bakeable unchanged.
    if (const auto game = parseGameDocument(document); game.has_value())
    {
        const SceneDescription *scene = game->startScene();
        if (scene != nullptr && scene->hasRecipe)
            return bakeGamePack(scene->recipe, scene->hasLiving ? &scene->living : nullptr);
    }

    procgen::WorldRecipe recipe{};
    if (auto parsed = parseSceneRecipe(document, recipe); !parsed)
        return std::unexpected(parsed.error());
    return bakeGamePack(recipe);
}

} // namespace lpl::editor
