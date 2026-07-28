/**
 * @file WaveFunctionCollapse.hpp
 * @brief Tiled WFC: fill a grid so that every neighbour pair is legal.
 *
 * The author states adjacency rules — "grass may sit next to sand, sand next to
 * water, grass never next to water" — and the solver finds an arrangement that
 * never violates one. It is constraint propagation, not randomness with a
 * filter: each cell holds the SET of tiles still possible, collapsing a cell
 * removes options from its neighbours, and that removal cascades.
 *
 * Three decisions make this reproducible, which classic WFC implementations
 * usually are not:
 *
 *  - **Entropy ties break by index.** The lowest-entropy cell is the one the
 *    solver collapses next, and there are usually many tied for lowest. Picking
 *    among them randomly (the usual trick to avoid visual artefacts) would make
 *    the result depend on how many numbers earlier passes drew. The lowest index
 *    wins instead, and variety comes from the seeded tile choice. The candidate
 *    is found through a heap rather than by rescanning the grid — the naive scan
 *    is quadratic in the cell count and measured at 105 ms for a 96x96 grid, which
 *    is most of a frame spent counting bits nobody asked about.
 *  - **Propagation uses an explicit stack**, not recursion: a 128x128 grid can
 *    cascade deeper than a kernel stack tolerates.
 *  - **Contradictions are repaired locally before the grid is restarted.** The
 *    knowledge base is explicit that throwing the whole generation away is the
 *    naive response; the layered recovery it describes clears a radius around
 *    the failure and re-solves just that neighbourhood against the surviving
 *    borders. Only when the repair budget runs out does the solve restart with
 *    a derived seed. Backtracking would be faster still, but its search order
 *    is far harder to keep identical across targets.
 *
 * Tiles are limited to 64 so a cell's possibility set is a single u64 — no
 * allocation per cell, and propagation is bit arithmetic.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_WAVEFUNCTIONCOLLAPSE_HPP
#    define LPL_PROCGEN_WAVEFUNCTIONCOLLAPSE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/// Maximum tiles in a set: one bit each in a 64-bit possibility mask.
inline constexpr core::u32 kMaxTiles = 64u;

/// Resulting tile index per cell.
using TileGrid = Grid<core::u8>;

/// Marks a cell no tile could satisfy.
inline constexpr core::u8 kNoTile = 0xFFu;

/**
 * @struct TileSet
 * @brief Tiles, their relative frequencies, and what may sit beside what.
 *
 * Adjacency is stored as a mask per (tile, direction): bit j of
 * `allowed[tile * 4 + direction]` means "tile j may sit in that direction from
 * `tile`". Directions follow @ref kNeighbor4X / @ref kNeighbor4Z order.
 */
struct TileSet {
    core::u32 tileCount{0u};             ///< Number of tiles, <= kMaxTiles.
    lpl::pmr::vector<core::u64> allowed; ///< tileCount * 4 adjacency masks.
    lpl::pmr::vector<core::u32> weight;  ///< Relative frequency per tile (0 disables it).

    /**
     * @brief Allocates a set of @p count tiles with no adjacency permitted yet.
     * @param count Tile count; clamped to kMaxTiles.
     */
    void reset(core::u32 count);

    /**
     * @brief Permits @p neighbour to sit @p direction-ward of @p tile.
     *
     * Adjacency must be symmetric to be meaningful, so this also permits
     * @p tile on the opposite side of @p neighbour. Declaring one direction and
     * forgetting the mirror is the classic WFC authoring bug — it produces
     * contradictions that look like solver failures.
     *
     * @param tile      Reference tile.
     * @param direction Index into kNeighbor4X/Z.
     * @param neighbour Tile allowed there.
     */
    void allow(core::u32 tile, core::u32 direction, core::u32 neighbour);

    /// @brief Permits @p a and @p b to touch in every direction.
    void allowAnywhere(core::u32 a, core::u32 b);

    /// @brief Sets a tile's relative frequency.
    void setWeight(core::u32 tile, core::u32 value);

    /// @return true when the set has tiles and every one of them has an adjacency.
    [[nodiscard]] bool valid() const;
};

/**
 * @struct WfcParams
 * @brief Grid size, seed, and how hard to try.
 */
struct WfcParams {
    core::u32 width{32u};             ///< Cells along X.
    core::u32 depth{32u};             ///< Cells along Z.
    core::u32 seed{1337u};            ///< Determinism anchor.
    core::u32 maxAttempts{8u};        ///< Full restarts allowed after a contradiction.
    core::u32 localRepairRadius{3u};  ///< Cells cleared around a contradiction before restarting (0 disables).
    core::u32 localRepairBudget{32u}; ///< Local repairs allowed per attempt.
};

/**
 * @struct WfcResult
 * @brief What the solve produced, and how hard it was.
 */
struct WfcResult {
    bool solved{false};           ///< Did an attempt complete without contradiction?
    core::u32 attempts{0u};       ///< Full attempts used (1 when it worked first try).
    core::u32 contradictions{0u}; ///< Contradictions hit across all attempts.
    core::u32 localRepairs{0u};   ///< Contradictions absorbed without a full restart.
    TileGrid tiles;               ///< The arrangement; cells are kNoTile when unsolved.
};

/**
 * @brief Solves a tile arrangement satisfying every adjacency rule.
 *
 * @p preset, when given, pins cells: any cell holding a tile index rather than
 * @ref kNoTile is treated as an immovable constraint the solve must work
 * around. That is what lets a chunk read its already-generated neighbours as
 * rigid borders and stay seamless, and what lets a heightfield dictate where
 * water and rock must go before the solver fills in the rest.
 *
 * @param tiles  Tile set with adjacency and weights.
 * @param params Grid size, seed, repair and attempt budgets.
 * @param preset Optional pinned cells; must match the params' dimensions.
 * @return The result; check @c solved before using @c tiles.
 */
[[nodiscard]] WfcResult solveWfc(const TileSet &tiles, const WfcParams &params, const TileGrid *preset = nullptr);

/**
 * @brief Verifies that every neighbouring pair in @p grid is legal.
 *
 * The independent check on the solver: it re-derives nothing, it only reads the
 * rules back. A solver bug that produced a plausible-looking but illegal grid
 * would pass a determinism test and fail this one.
 *
 * @param grid  Arrangement to verify.
 * @param tiles The rules it should satisfy.
 * @return Number of violated adjacencies (0 when the grid is legal).
 */
[[nodiscard]] core::u32 countAdjacencyViolations(const TileGrid &grid, const TileSet &tiles);

/**
 * @brief A small ready-made tile set: water, sand, grass, forest, rock.
 *
 * Neighbours are allowed only between adjacent steps of that sequence, which is
 * what produces beaches between sea and land instead of hard edges.
 *
 * @return The tile set (5 tiles).
 */
[[nodiscard]] TileSet makeTerrainTileSet();

/// Tile indices of @ref makeTerrainTileSet.
enum class TerrainTile : core::u8 {
    Water = 0,
    Sand,
    Grass,
    Forest,
    Rock
};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WAVEFUNCTIONCOLLAPSE_HPP
