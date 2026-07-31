/**
 * @file ChunkTerrain.hpp
 * @brief One chunk of an endless world: eroded relief, drainage, climate, biomes.
 *
 * The generation of a single chunk, as a function of its coordinates and nothing
 * else. That "nothing else" is the whole point: every value is sampled at ABSOLUTE
 * world coordinates, so two chunks generated at different times, on different
 * machines, in a different order, agree along their shared border by construction
 * rather than by a stitching pass.
 *
 * Three things in here are the answers to questions a chunked world asks and a
 * bounded one does not:
 *
 *  - erosion needs an APRON. Thermal relaxation propagates one cell per iteration,
 *    so a chunk generated with N iterations must be computed on a field N + 1 cells
 *    wider on every side and then cropped, or the chunked assembly stops being the
 *    unchunked computation. The host test measures that equality rather than
 *    asserting it.
 *  - drainage has no edges to flow off, so it is posed as a BOUNDED question: a cell
 *    carries water when enough terrain within a radius drains through it. The verdict
 *    depends only on the position, so neighbours agree.
 *  - temperature has to fall with latitude, and an endless world has no equator
 *    unless one is chosen. Here it is z = 0; without it, a world is one biome band
 *    repeated to infinity.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_CHUNK_TERRAIN_HPP
#    define LPL_PROCGEN_CHUNK_TERRAIN_HPP

#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Climate.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Random.hpp>
#    include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

/**
 * @struct ChunkTerrainRule
 * @brief Where the sea, the snow line and the rock line are, and how wet the world is.
 *
 * Altitude and water have the last word over the climate axes, exactly as they do in
 * the bounded classifier: no noise axis knows where the sea is.
 */
struct ChunkTerrainRule {
    core::u32 erosionIterations{6u}; ///< Thermal passes; also sets the apron.
    core::f32 seaLevel{-1.0f};
    core::f32 beachBand{0.8f};    ///< Height above the sea that is still shore.
    core::f32 rockLine{8.0f};     ///< Above this, bare rock.
    core::f32 snowLine{11.0f};    ///< Above this, snow.
    core::f32 latitudeScale{0.0016f}; ///< How fast it cools away from z = 0.
    core::f32 baseWarmth{0.72f};
    core::f32 altitudeCooling{0.03f};
    core::u32 vegetationOneIn{3u}; ///< One wooded cell in N grows a plant.
};

/**
 * @struct ChunkTerrain
 * @brief What one chunk's terrain is.
 */
struct ChunkTerrain {
    Heightfield height{};
    BiomeMap biomes{};
    Grid<core::u8> rivers{};
    core::f32 lowest{0.0f}; ///< Lowest cell: whether this chunk has water at all.
};

/**
 * @brief Generates one chunk's relief, drainage and biomes.
 *
 * @param emitPlant Called with (worldCellX, worldCellZ) for each plant grown. The
 *                  thinning is seeded from the chunk, so the same chunk grows the
 *                  same forest every time it streams in — which is what stops a
 *                  wood from rearranging itself when a walker turns around.
 */
template <typename EmitPlant>
[[nodiscard]] ChunkTerrain generateChunkTerrain(const ChunkParams &params, const EndlessRiverParams &riverParams,
                                                ChunkCoord coord, const ChunkTerrainRule &rule, EmitPlant &&emitPlant)
{
    const core::u32 size = params.size;
    ChunkTerrain out;
    out.height = generateErodedChunkTerrain(params, coord, rule.erosionIterations);
    out.biomes = BiomeMap{size, size, BiomeId::Grassland};
    out.rivers = markChunkRivers(params, riverParams, coord);

    out.lowest = out.height.empty() ? 0.0f : out.height.at(0u, 0u).toFloat();
    for (core::u32 z = 0u; z < size; ++z)
        for (core::u32 x = 0u; x < size; ++x)
        {
            const core::f32 h = out.height.at(x, z).toFloat();
            if (h < out.lowest)
                out.lowest = h;
        }

    const core::i32 originX = coord.x * static_cast<core::i32>(size);
    const core::i32 originZ = coord.z * static_cast<core::i32>(size);

    // Two extra noise layers, both at absolute coordinates: moisture at a little
    // under the terrain's frequency, continentalness at a quarter of it. Deriving
    // them from the terrain's own parameters is what keeps a wet world wet at every
    // scale instead of only at the one the layer was tuned at.
    NoiseParams moistureLayer = params.noise;
    moistureLayer.seed = params.worldSeed ^ 0x3A15E7u;
    moistureLayer.frequency = params.noise.frequency * 0.6f;
    moistureLayer.amplitude = 1.0f;
    moistureLayer.octaves = 3u;

    NoiseParams continentLayer = moistureLayer;
    continentLayer.seed = params.worldSeed ^ 0xC0A57Du;
    continentLayer.frequency = params.noise.frequency * 0.25f;

    Random thin{chunkSeed(params, coord)};
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
                                     (height > 0.0f ? height * rule.altitudeCooling : 0.0f);
            climate[ClimateAxis::Temperature] = math::Fixed32::fromFloat(clamp01(warmth));

            const core::f32 wet = 0.5f + sampleNoiseAt(worldX, worldZ, moistureLayer).toFloat() * 0.5f;
            climate[ClimateAxis::Moisture] = math::Fixed32::fromFloat(clamp01(wet));

            const core::f32 land = 0.5f + sampleNoiseAt(worldX, worldZ, continentLayer).toFloat() * 0.5f;
            climate[ClimateAxis::Continentalness] = math::Fixed32::fromFloat(clamp01(land));
            climate[ClimateAxis::Erosion] = math::Fixed32::half();
            climate[ClimateAxis::Depth] = math::Fixed32::zero();
            climate[ClimateAxis::Weirdness] = math::Fixed32::half();

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
                biome = BiomeId::Lake; // drawn as running water
            out.biomes.at(x, z) = biome;

            const bool wooded = biome == BiomeId::Forest || biome == BiomeId::Taiga || biome == BiomeId::Rainforest;
            if (wooded && (rule.vegetationOneIn <= 1u || thin.below(rule.vegetationOneIn) == 0u))
                emitPlant(worldX, worldZ);
        }

    return out;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CHUNK_TERRAIN_HPP
