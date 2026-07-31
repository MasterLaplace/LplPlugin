/**
 * @file TerrainStreamer.hpp
 * @brief The chunks of an endless world: generated around a walker, released behind.
 *
 * Binds the three pieces that a streamed world needs at once and that are useless
 * apart: procgen::ChunkResidency (what is loaded), procgen::generateChunkTerrain
 * (what a chunk is made of) and the payload a world keeps per chunk — its shadow
 * mask and its plants.
 *
 * The one thing it insists on is @ref groundAt. A streamed world is DRAWN from the
 * resident field, so anything standing on the ground must read the same field; the
 * raw noise differs from it by exactly the material erosion moved, which put a herd
 * in mid-air for a day. Outside residency the noise still answers, because a creature
 * walking toward the horizon needs an answer there too.
 *
 * A composable object: a world HAS a streamer. Nothing here knows what a biome looks
 * like, how the world is drawn, or what walks on it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_TERRAIN_STREAMER_HPP
#    define LPL_ENGINE_TERRAIN_STREAMER_HPP

#    include <lpl/ecology/Vegetation.hpp>
#    include <lpl/procgen/ChunkResidency.hpp>
#    include <lpl/procgen/ChunkTerrain.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::engine {

/**
 * @struct TerrainChunk
 * @brief One resident chunk: its terrain, and what a world keeps on it.
 */
struct TerrainChunk {
    procgen::ChunkCoord coord{};
    procgen::Heightfield height{};
    procgen::BiomeMap biomes{};
    procgen::Grid<core::u8> rivers{};
    procgen::Grid<core::u8> shade{}; ///< 0 lit, 255 shadowed; refreshed as the sun moves.
    lpl::pmr::vector<ecology::PlantCell> plants;
    core::f32 lowest{0.0f}; ///< Lowest cell: whether this chunk has water at all.
};

/**
 * @class TerrainStreamer
 * @brief Residency, generation and the one answer to "how high is the ground".
 */
class TerrainStreamer {
public:
    void configure(const procgen::ChunkParams &chunkParams, const procgen::EndlessRiverParams &riverParams,
                   const procgen::StreamingParams &streamParams, core::u32 maxResident,
                   const procgen::ChunkTerrainRule &rule);

    void clear() { _residency.clear(); }

    [[nodiscard]] core::u32 size() const noexcept { return _residency.size(); }
    [[nodiscard]] bool empty() const noexcept { return _residency.empty(); }
    [[nodiscard]] TerrainChunk &at(core::u32 index) noexcept { return _residency.at(index); }
    [[nodiscard]] const TerrainChunk &at(core::u32 index) const noexcept { return _residency.at(index); }
    [[nodiscard]] core::u32 generatedCount() const noexcept { return _residency.generatedCount(); }
    [[nodiscard]] core::u32 releasedCount() const noexcept { return _residency.releasedCount(); }
    [[nodiscard]] core::u32 maxResident() const noexcept { return _residency.maxResident(); }
    [[nodiscard]] const procgen::ChunkParams &chunkParams() const noexcept { return _chunkParams; }
    [[nodiscard]] core::u32 chunkSize() const noexcept { return _chunkParams.size; }

    /**
     * @brief Generates and releases chunks around a focus, within the tick budget.
     *
     * @param onGenerated Called with each fresh chunk, for whatever a world does once
     *                    per new chunk — a shadow mask, an entity spawn.
     */
    template <typename OnGenerated>
    void update(core::f32 focusCellX, core::f32 focusCellZ, core::f32 headingX, core::f32 headingZ,
                OnGenerated &&onGenerated)
    {
        _residency.stream(
            focusCellX, focusCellZ, headingX, headingZ,
            [this](procgen::ChunkCoord coord) { return buildChunk(coord); }, onGenerated);
    }

    /** @brief Ground height at a world cell — the resident field first, then the noise. */
    [[nodiscard]] core::f32 groundAt(core::i32 worldX, core::i32 worldZ) const;

    /** @brief The chunk holding a world cell, or nullptr. */
    [[nodiscard]] const TerrainChunk *chunkAt(core::i32 worldX, core::i32 worldZ) const noexcept
    {
        return _residency.findByCell(worldX, worldZ);
    }

    /**
     * @brief The next chunk whose shadow mask is due, round-robin.
     *
     * Amortised rather than global: the sun moves a day in four minutes, so a mask a
     * second stale is invisible, and refreshing every resident chunk on the tick the
     * sun moves is a spike with no visible benefit.
     */
    [[nodiscard]] TerrainChunk *nextShadowChunk() noexcept;

private:
    [[nodiscard]] TerrainChunk buildChunk(procgen::ChunkCoord coord) const;

    procgen::ChunkResidency<TerrainChunk> _residency;
    procgen::ChunkParams _chunkParams{};
    procgen::EndlessRiverParams _riverParams{};
    procgen::ChunkTerrainRule _rule{};
    core::u32 _shadowCursor{0u};
};

} // namespace lpl::engine

// Out-of-line definitions: the streamer is consumed header-only, the freestanding
// kernel included, so they live in a .inl rather than a .cpp that neither kernel
// build path lists.
#    include <lpl/engine/TerrainStreamer.inl>

#endif // LPL_ENGINE_TERRAIN_STREAMER_HPP
