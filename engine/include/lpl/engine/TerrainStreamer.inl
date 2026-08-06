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

inline math::Fixed32 TerrainStreamer::groundHeightAt(core::i32 worldX, core::i32 worldZ) const
{
    return _residency.groundFixedAt(worldX, worldZ, [](const TerrainChunk &chunk, core::u32 x, core::u32 z) {
        return chunk.height.empty() ? math::Fixed32{} : chunk.height.at(x, z);
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
    chunk.flow = terrain.flow;
    chunk.lowest = terrain.lowest;
    chunk.seaMinX = terrain.seaMinX;
    chunk.seaMaxX = terrain.seaMaxX;
    chunk.seaMinZ = terrain.seaMinZ;
    chunk.seaMaxZ = terrain.seaMaxZ;
    chunk.hasSea = terrain.hasSea;
    chunk.hasRiver = terrain.hasRiver;
    chunk.caveMouths = terrain.caveMouths;
    chunk.buildings = terrain.buildings;
    // MOVED, not copied. A warren carries a voxel volume and a cover mask — about
    // twelve kibibytes each — and copying them would double the peak of the one
    // allocation in this function that is not bounded by the chunk size.
    for (core::u32 i = 0u; i < terrain.warrens.size(); ++i)
        chunk.warrens.push_back(
            static_cast<procgen::CaveWarren &&>(const_cast<procgen::CaveWarren &>(terrain.warrens[i])));
    chunk.shade = procgen::Grid<core::u8>{_chunkParams.size, _chunkParams.size, 0u};
    return chunk;
}

inline const procgen::CaveWarren *TerrainStreamer::warrenAt(core::i32 worldX, core::i32 worldZ) const noexcept
{
    // A linear scan over the resident set, and it stays one: a warren is owned WHOLE
    // by one chunk and spills into its neighbours, so "the chunk holding this cell"
    // is not the chunk holding the warren, and an index keyed on the cell would have
    // to be rebuilt every time a chunk streamed. Measured against the alternative it
    // is a few dozen rectangle tests per query.
    for (core::u32 i = 0u; i < _residency.size(); ++i)
    {
        const TerrainChunk &chunk = _residency.at(i);
        for (core::u32 w = 0u; w < chunk.warrens.size(); ++w)
            if (chunk.warrens[w].isCavernous(worldX, worldZ))
                return &chunk.warrens[w];
    }
    return nullptr;
}

inline procgen::VerticalSpan TerrainStreamer::spanAt(core::i32 worldX, core::i32 worldZ, math::Fixed32 y) const
{
    const math::Fixed32 ground = groundHeightAt(worldX, worldZ);
    const procgen::CaveWarren *warren = warrenAt(worldX, worldZ);
    if (warren == nullptr)
        return procgen::VerticalSpan{ground, procgen::openSky(), false};
    return procgen::caveWarrenSpanAt(*warren, worldX, worldZ, y, ground);
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_STREAMER_INL
