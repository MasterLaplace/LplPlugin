/**
 * @file Chunking.hpp
 * @brief Generating a world larger than memory, one piece at a time.
 *
 * An unbounded world cannot be generated as one grid, and generating it in
 * pieces is only safe if a piece depends on nothing but its own coordinates.
 * That is the whole discipline here: every value is a pure function of the
 * WORLD coordinate and the world seed, never of a running state or of what a
 * neighbouring chunk happened to produce.
 *
 * The payoff is exactly what a networked game needs. A server does not ship
 * terrain, it ships a seed; every client rebuilds the same world locally, and
 * two players standing in opposite corners see the same ground without a single
 * message about it. It also means chunks can be generated in any order, dropped,
 * and regenerated identically.
 *
 * Coordinates stay integral for the same reason. Far from the origin a float
 * loses the precision to distinguish adjacent cells, and a world that quietly
 * degrades at its edges is worse than one with a hard boundary.
 *
 * Heightfields are seamless for free — the noise is sampled at world
 * coordinates, so neighbouring chunks agree on their shared edge by
 * construction. Constraint-based passes are not: a WFC solve knows nothing of
 * its neighbour, so @ref borderConstraintsFrom lifts a neighbour's edge into
 * the pinned cells of the next solve.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_CHUNKING_HPP
#    define LPL_PROCGEN_CHUNKING_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/WaveFunctionCollapse.hpp>

namespace lpl::procgen {

/**
 * @struct ChunkCoord
 * @brief Which chunk, in chunk units (not cells).
 */
struct ChunkCoord {
    core::i32 x{0}; ///< Chunk column; may be negative.
    core::i32 z{0}; ///< Chunk row; may be negative.

    [[nodiscard]] constexpr bool operator==(const ChunkCoord &other) const noexcept
    {
        return x == other.x && z == other.z;
    }
};

/**
 * @struct ChunkParams
 * @brief Chunk size and the world seed every chunk derives from.
 */
struct ChunkParams {
    core::u32 size{32u};      ///< Cells per chunk edge.
    core::u32 worldSeed{1337u}; ///< The one seed the whole world comes from.
    NoiseParams noise{};      ///< Terrain sampling; its own seed is ignored.
};

/**
 * @brief The seed a chunk's discrete passes should use.
 *
 * Derived from the world seed and the chunk coordinates, so it is reproducible
 * from the coordinates alone — no chunk has to have been generated before.
 *
 * @param params World parameters.
 * @param coord  Chunk to seed.
 * @return A seed unique to that chunk.
 */
[[nodiscard]] core::u32 chunkSeed(const ChunkParams &params, ChunkCoord coord);

/**
 * @brief Generates one chunk's terrain.
 *
 * Sampled at world coordinates rather than chunk-local ones, which is what makes
 * the edges match: the last column of one chunk and the first of the next are
 * the same world position evaluated twice.
 *
 * @param params World parameters.
 * @param coord  Chunk to generate.
 * @return The chunk's heightfield, of size `params.size` squared.
 */
[[nodiscard]] Heightfield generateChunkTerrain(const ChunkParams &params, ChunkCoord coord);

/**
 * @brief Height at one world cell, without generating its chunk.
 * @param params  World parameters.
 * @param worldX  World column.
 * @param worldZ  World row.
 * @return The elevation there.
 */
[[nodiscard]] math::Fixed32 sampleWorldHeight(const ChunkParams &params, core::i32 worldX, core::i32 worldZ);

/**
 * @brief Checks that two neighbouring chunks agree along their shared edge.
 * @param params World parameters.
 * @param a      First chunk.
 * @param b      Second chunk; must be 4-adjacent to @p a.
 * @return Number of mismatched cells along the seam (0 when seamless).
 */
[[nodiscard]] core::u32 countSeamMismatches(const ChunkParams &params, ChunkCoord a, ChunkCoord b);

/**
 * @brief Builds a WFC preset that pins a chunk's edge to its neighbour's.
 *
 * The mechanism that keeps constraint-solved chunks seamless: the already
 * generated neighbour's touching row or column becomes immovable input to the
 * next solve, so the solver has to agree with what is already there rather than
 * being told about it afterwards.
 *
 * @param size          Chunk edge in cells.
 * @param neighbour     The solved neighbour.
 * @param neighbourSide Which side of the NEW chunk the neighbour lies on
 *                      (index into kNeighbor4X/Z).
 * @return A preset grid, pinned along one edge and free elsewhere.
 */
[[nodiscard]] TileGrid borderConstraintsFrom(core::u32 size, const TileGrid &neighbour, core::u32 neighbourSide);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CHUNKING_HPP
