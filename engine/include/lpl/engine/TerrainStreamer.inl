/**
 * @file TerrainStreamer.inl
 * @brief Out-of-line definitions for engine::TerrainStreamer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_TERRAIN_STREAMER_INL
#    define LPL_ENGINE_TERRAIN_STREAMER_INL

namespace lpl::engine {

inline void TerrainStreamer::configure(const procgen::ChunkParams &chunkParams,
                                       const procgen::EndlessRiverParams &riverParams,
                                       const procgen::StreamingParams &streamParams, core::u32 maxResident,
                                       const procgen::ChunkTerrainRule &rule)
{
    _chunkParams = chunkParams;
    _riverParams = riverParams;
    _rule = rule;
    _residency.configure(chunkParams, streamParams, maxResident);
    _shadowCursor = 0u;
}

/** @brief Ground height at a world cell — the resident field first, then the noise. */
inline core::f32 TerrainStreamer::groundAt(core::i32 worldX, core::i32 worldZ) const
{
    return _residency.groundAt(worldX, worldZ, [](const TerrainChunk &chunk, core::u32 x, core::u32 z) {
        return chunk.height.empty() ? 0.0f : chunk.height.at(x, z).toFloat();
    });
}

inline TerrainChunk *TerrainStreamer::nextShadowChunk() noexcept
{
    if (_residency.empty())
        return nullptr;
    _shadowCursor = (_shadowCursor + 1u) % _residency.size();
    return &_residency.at(_shadowCursor);
}

inline TerrainChunk TerrainStreamer::buildChunk(procgen::ChunkCoord coord) const
{
    TerrainChunk chunk;
    chunk.coord = coord;

    const procgen::ChunkTerrain terrain = procgen::generateChunkTerrain(_chunkParams, _riverParams, coord, _rule,
                                                                        [&chunk](core::i32 worldX, core::i32 worldZ) {
                                                                            ecology::PlantCell plant;
                                                                            plant.cellX = worldX;
                                                                            plant.cellZ = worldZ;
                                                                            chunk.plants.push_back(plant);
                                                                        });

    chunk.height = terrain.height;
    chunk.biomes = terrain.biomes;
    chunk.rivers = terrain.rivers;
    chunk.lowest = terrain.lowest;
    chunk.shade = procgen::Grid<core::u8>{_chunkParams.size, _chunkParams.size, 0u};
    return chunk;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_STREAMER_INL
