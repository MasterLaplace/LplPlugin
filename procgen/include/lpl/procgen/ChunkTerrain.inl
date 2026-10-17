#ifndef LPL_PROCGEN_CHUNK_TERRAIN_INL
#define LPL_PROCGEN_CHUNK_TERRAIN_INL

namespace lpl::procgen {

template <typename EmitPlant>
ChunkTerrain generateChunkTerrain(const ChunkParams &params, const EndlessRiverParams &riverParams, ChunkCoord coord,
                                  const ChunkTerrainRule &rule, EmitPlant &&emitPlant)
{
    const core::u32 size = params.size;
    ChunkTerrain out;
    out.height = generateErodedChunkTerrain(params, coord, rule.erosionIterations);
    out.biomes = BiomeMap{size, size, BiomeId::Grassland};
    out.rivers = markChunkRivers(params, riverParams, coord);

    if (rule.riverDepth > 0.0f)
    {
        const math::Fixed32 depth = math::Fixed32::fromFloat(rule.riverDepth);
        for (core::u32 z = 0u; z < size; ++z)
            for (core::u32 x = 0u; x < size; ++x)
                if (out.rivers.at(x, z) != 0u)
                    out.height.at(x, z) = out.height.at(x, z) - depth;
    }

    if (rule.carveCaveMouths)
    {
        const core::i32 originX = coord.x * static_cast<core::i32>(size);
        const core::i32 originZ = coord.z * static_cast<core::i32>(size);
        forEachLandmarkNear(params, rule.caveMouths, LandmarkKind::CaveMouth, rule.seaLevel, coord,
                            [&](const LandmarkSite &site) {
                                const CaveAdit adit = planCaveAdit(params, site, rule.warren, rule.caveMouthDrop);
                                for (core::u32 z = 0u; z < size; ++z)
                                    for (core::u32 x = 0u; x < size; ++x)
                                    {
                                        core::f32 floor = 0.0f;
                                        if (!caveMouthFloorAt(site, adit, originX + static_cast<core::i32>(x),
                                                              originZ + static_cast<core::i32>(z), floor))
                                            continue;
                                        const math::Fixed32 level = math::Fixed32::fromFloat(floor);
                                        if (out.height.at(x, z) > level)
                                            out.height.at(x, z) = level;
                                    }
                                if (!chunkOwnsLandmark(params, site, coord))
                                    return;
                                out.caveMouths.push_back(site);
                                if (rule.buildWarrens)
                                {
                                    CaveWarren warren = buildCaveWarren(params, site, rule.warren, rule.caveMouthDrop);
                                    if (warren.valid)
                                        out.warrens.push_back(static_cast<CaveWarren &&>(warren));
                                }
                            });
    }

    if (rule.raiseVillages)
    {
        const core::i32 originX = coord.x * static_cast<core::i32>(size);
        const core::i32 originZ = coord.z * static_cast<core::i32>(size);
        forEachLandmarkNear(
            params, rule.villages, LandmarkKind::Settlement, rule.seaLevel, coord, [&](const LandmarkSite &site) {
                const core::i32 radius = static_cast<core::i32>(site.radius);
                const core::f32 pad = site.height;
                const core::f32 feather = 3.0f;
                for (core::u32 z = 0u; z < size; ++z)
                    for (core::u32 x = 0u; x < size; ++x)
                    {
                        const core::i32 dx = originX + static_cast<core::i32>(x) - site.cellX;
                        const core::i32 dz = originZ + static_cast<core::i32>(z) - site.cellZ;
                        const core::i32 spanX = dx < 0 ? -dx : dx;
                        const core::i32 spanZ = dz < 0 ? -dz : dz;
                        const core::i32 furthest = spanX > spanZ ? spanX : spanZ;
                        if (furthest > radius)
                            continue;
                        const core::f32 inside = static_cast<core::f32>(radius - furthest);
                        core::f32 weight = inside / feather;
                        weight = weight > 1.0f ? 1.0f : weight;
                        const core::f32 natural = out.height.at(x, z).toFloat();
                        out.height.at(x, z) = math::Fixed32::fromFloat(natural + (pad - natural) * weight);
                    }

                if (!chunkOwnsLandmark(params, site, coord))
                    return;
                const VillagePlan plan = planVillage(params, site);
                forEachVillageBuilding(plan,
                                       [&out](const LandmarkBuilding &building) { out.buildings.push_back(building); });
            });
    }

    out.flow = markChunkRiverFlow(params, riverParams, coord, out.rivers);

    out.lowest = out.height.empty() ? 0.0f : out.height.at(0u, 0u).toFloat();
    for (core::u32 z = 0u; z < size; ++z)
        for (core::u32 x = 0u; x < size; ++x)
        {
            const core::f32 h = out.height.at(x, z).toFloat();
            if (h < out.lowest)
                out.lowest = h;
            if (h < rule.seaLevel)
            {
                if (!out.hasSea)
                {
                    out.seaMinX = x;
                    out.seaMaxX = x;
                    out.seaMinZ = z;
                    out.seaMaxZ = z;
                    out.hasSea = true;
                }
                else
                {
                    if (x < out.seaMinX)
                        out.seaMinX = x;
                    if (x > out.seaMaxX)
                        out.seaMaxX = x;
                    if (z < out.seaMinZ)
                        out.seaMinZ = z;
                    if (z > out.seaMaxZ)
                        out.seaMaxZ = z;
                }
            }
            if (!out.rivers.empty() && out.rivers.at(x, z) != 0u)
                out.hasRiver = true;
        }

    const core::i32 originX = coord.x * static_cast<core::i32>(size);
    const core::i32 originZ = coord.z * static_cast<core::i32>(size);

    NoiseParams climateLayer{};
    climateLayer.amplitude = 1.0f;
    climateLayer.octaves = 3u;

    NoiseParams moistureLayer = climateLayer;
    moistureLayer.seed = params.worldSeed ^ 0x3A15E7u;
    moistureLayer.frequency = params.noise.frequency * 0.6f;

    NoiseParams continentLayer = climateLayer;
    continentLayer.seed = params.worldSeed ^ 0xC0A57Du;
    continentLayer.frequency = params.noise.frequency * 0.25f;

    NoiseParams weirdnessLayer = climateLayer;
    weirdnessLayer.seed = params.worldSeed ^ 0x7E12D0u;
    weirdnessLayer.frequency = params.noise.frequency * 0.9f;

    math::Random thin{chunkSeed(params, coord)};
    const auto clamp01 = [](core::f32 v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };

    for (core::u32 z = 0u; z < size; ++z)
        for (core::u32 x = 0u; x < size; ++x)
        {
            const core::i32 worldX = originX + static_cast<core::i32>(x);
            const core::i32 worldZ = originZ + static_cast<core::i32>(z);
            const core::f32 height = out.height.at(x, z).toFloat();

            ClimateVector climate{};
            const core::f32 latitude = static_cast<core::f32>(worldZ) * rule.latitudeScale;
            const core::f32 warmth = rule.baseWarmth - (latitude < 0.0f ? -latitude : latitude) -
                                     (height > rule.seaLevel ? (height - rule.seaLevel) * rule.altitudeCooling : 0.0f);
            climate[ClimateAxis::Temperature] = math::Fixed32::fromFloat(clamp01(warmth));

            const core::f32 wet = 0.5f + sampleNoiseAt(worldX, worldZ, moistureLayer).toFloat() * 0.5f;
            climate[ClimateAxis::Moisture] = math::Fixed32::fromFloat(clamp01(wet));

            const core::f32 land = 0.5f + sampleNoiseAt(worldX, worldZ, continentLayer).toFloat() * 0.5f;
            climate[ClimateAxis::Continentalness] = math::Fixed32::fromFloat(clamp01(land));
            const core::f32 east = out.height.at(x + 1u < size ? x + 1u : x, z).toFloat();
            const core::f32 south = out.height.at(x, z + 1u < size ? z + 1u : z).toFloat();
            const core::f32 slope =
                ((east > height ? east - height : height - east) + (south > height ? south - height : height - south)) *
                0.5f;
            climate[ClimateAxis::Erosion] =
                math::Fixed32::fromFloat(clamp01(0.5f + slope / (params.noise.amplitude * 0.25f)));
            climate[ClimateAxis::Depth] =
                math::Fixed32::fromFloat(height < rule.seaLevel ? clamp01((rule.seaLevel - height) / 16.0f) : 0.0f);
            climate[ClimateAxis::Weirdness] = math::Fixed32::fromFloat(
                clamp01(0.5f + sampleNoiseAt(worldX, worldZ, weirdnessLayer).toFloat() * 0.5f));

            math::Fixed32 distance{};
            BiomeId biome = nearestBiomeProfile(climate, distance);

            if (height < rule.seaLevel)
                biome = BiomeId::Ocean;
            else if (height < rule.seaLevel + rule.beachBand)
                biome = BiomeId::Beach;
            else if (height > rule.snowLine)
                biome = BiomeId::Snow;
            else if (height > rule.rockLine)
                biome = BiomeId::Rock;
            if (!out.rivers.empty() && out.rivers.at(x, z) != 0u && height >= rule.seaLevel)
                biome = BiomeId::Lake;
            out.biomes.at(x, z) = biome;

            const bool wooded = biome == BiomeId::Forest || biome == BiomeId::Taiga || biome == BiomeId::Rainforest;
            if (wooded && (rule.vegetationOneIn <= 1u || thin.below(rule.vegetationOneIn) == 0u))
                emitPlant(worldX, worldZ);
        }

    return out;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CHUNK_TERRAIN_INL
