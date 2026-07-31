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
#    include <lpl/procgen/Grid.hpp>
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
    core::u32 size{32u};        ///< Cells per chunk edge.
    core::u32 worldSeed{1337u}; ///< The one seed the whole world comes from.
    NoiseParams noise{};        ///< Terrain sampling; its own seed is ignored.
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
 * @brief Generates one chunk's terrain WITH thermal erosion, seamlessly.
 *
 * Erosion is a relaxation over a grid, and a relaxation has no meaning on a
 * piece of a world that continues past its edge — which is why the endless world
 * shipped as raw noise. There is an exact answer for the LOCAL kind, and it is
 * arithmetic rather than a trick:
 *
 * thermal erosion moves material one cell per iteration, so after N iterations a
 * cell's value depends only on cells within N of it. Generate the chunk with an
 * APRON of N cells on every side, relax the enlarged grid, and keep the interior:
 * every kept cell was computed from exactly the set it would have seen in any
 * other chunk that contained it. The seam is still exact, and
 * @ref countSeamMismatches over an eroded patch is what proves it rather than
 * this paragraph.
 *
 * @warning HYDRAULIC erosion is not bounded this way. Water runs downhill for as
 *          far as the slope lasts, so a droplet's influence is the length of a
 *          river, not the iteration count — no finite apron makes it exact. It
 *          stays a bounded-world pass, and pretending otherwise would put a
 *          different world on either side of every border.
 *
 * @param params     World parameters.
 * @param coord      Chunk to generate.
 * @param iterations Thermal iterations; also the apron width.
 * @param talus      Slope above which material slides.
 * @return The eroded chunk, of size `params.size` squared.
 */
[[nodiscard]] Heightfield generateErodedChunkTerrain(const ChunkParams &params, ChunkCoord coord, core::u32 iterations,
                                                     core::f32 talus = 0.6f);

/**
 * @brief Height at one world cell, without generating its chunk.
 * @param params  World parameters.
 * @param worldX  World column.
 * @param worldZ  World row.
 * @return The elevation there.
 */
[[nodiscard]] math::Fixed32 sampleWorldHeight(const ChunkParams &params, core::i32 worldX, core::i32 worldZ);

/**
 * @struct EndlessRiverParams
 * @brief How a river is decided in a world that has no edges.
 *
 * Drainage on a bounded map is a GLOBAL question: fill the depressions, then
 * accumulate flow from every cell down to the sea. Neither half exists here —
 * there is no "every cell", and a depression may drain through terrain nobody
 * has generated. So the question is re-posed as a local one: a cell carries a
 * river when enough of the terrain WITHIN A BOUNDED RADIUS drains through it.
 *
 * That trade is what makes the answer chunk-independent. The verdict for a coarse
 * cell depends only on the cells within @ref basinRadius of it, and those are the
 * same cells whichever chunk asks — so two neighbouring chunks agree along their
 * seam by construction, exactly as their heights do.
 *
 * What it costs, stated plainly: rivers stop growing past the radius. A bounded
 * basin cannot produce a continental trunk, only the tributaries of one. A world
 * that needs the Amazon needs a coarse HiGen level routed once and cached, and
 * this is the level below it, not a replacement for it.
 */
struct EndlessRiverParams {
    core::u32 coarseCells{4u}; ///< Fine cells per coarse cell; the river's width scale.
    core::u32 basinRadius{7u}; ///< Coarse cells searched upstream. Cost is quadratic in it.
    /**
     * @brief Upstream cells that must drain through before it is a river.
     *
     * Six, and the number is measured rather than chosen. On a world that is
     * largely sea the points where flow CONVERGES are the drowned basins: the
     * global maximum over a chunk runs 14 to 36, while the highest count on dry
     * land is 2 to 13. A threshold calibrated on the global figure therefore
     * marks nothing but ocean — every candidate it accepts is then rejected for
     * already being the sea, and the map comes out with no rivers at all while
     * every check about rivers passes vacuously.
     *
     * What runs on land is a TRIBUTARY, and a tributary carries a small basin by
     * definition. This threshold is calibrated on those.
     */
    core::u32 riverThreshold{6u};
    core::f32 seaLevel{-1.0f}; ///< Below this the water is already the sea.

    // ── The trunk: a second, coarser level ──────────────────────────────────
    //
    // A bounded basin gives tributaries and nothing else — that is arithmetic,
    // not a limitation to be tuned away. A river that crosses a continent needs a
    // basin the size of a continent, and sampling one per cell is unaffordable.
    //
    // So the same question is asked twice, at two scales, and the coarse answer
    // constrains the fine one: exactly the cascade rule HiGen exists to enforce.
    // The macro level is sampled sparsely (one height per @ref trunkCells coarse
    // cells) and cached, so its reach costs a few hundred samples per macro cell
    // rather than per world cell.

    /**
     * @brief Octaves the macro level keeps.
     *
     * The macro level must sample a SMOOTHED field, not the detailed one at
     * sparse points — and the difference is the whole reason trunks work at all.
     * Measured: sampling the full five-octave terrain every 32 cells gave a macro
     * surface where every cell was a local pit, so every downhill walk stopped
     * immediately and not one macro cell in eighty-one reached an upstream count
     * of six. That is aliasing, not geography: fBm keeps most of its energy in
     * the small octaves, and reading it sparsely returns noise.
     *
     * Two octaves is the continental trend — the shape that actually has long
     * slopes and basins for water to run down.
     */
    core::u32 trunkOctaves{2u};

    /**
     * @brief Macro frequency, as a multiple of the terrain's.
     *
     * Dropping octaves was not enough, and the measurement says why: octaves
     * change the DETAIL, not the base scale. At the terrain's own frequency one
     * period spans about eleven world cells while macro cells sit thirty-two
     * apart — so the macro surface was still an aliased read of a field finer
     * than its own sampling, and every cell stayed a pit. One macro cell in
     * eighty-one reached an upstream count of two.
     *
     * At a twelfth of the terrain's frequency a period spans several macro cells,
     * which is the condition for a downhill walk to have somewhere to go. This is
     * the continental trend the trunk is routed on — and the fine terrain is then
     * CARVED to match it, because a river that ignores the ground it crosses is a
     * blue line painted over a hill.
     */
    core::f32 trunkFrequencyScale{0.08f};

    core::u32 trunkCells{8u};   ///< Coarse cells per macro cell.
    core::u32 trunkRadius{10u}; ///< Macro cells searched upstream. This is the reach.
    /**
     * @brief Upstream macro cells before a trunk runs.
     *
     * Six, measured over 625 macro cells: 163 carry a trunk at a threshold of
     * two, 18 at six, 6 at fourteen, 1 at eighteen. Six keeps the rivers that
     * drain a real basin and drops the pairs of cells that merely lean the same
     * way.
     *
     * @warning An earlier version of this comment claimed the demo world had no
     *          continental drainage at all — measured, honestly, and over a
     *          window of eighty-one macro cells that happened to be mostly sea.
     *          Widening the sample to 625 showed plenty. Sampling a corner and
     *          generalising to the world is the same mistake as sampling a chunk
     *          and generalising to the map, and it produced a confident wrong
     *          conclusion here.
     */
    core::u32 trunkThreshold{6u};

    core::u32 trunkWidth{1u}; ///< Coarse cells either side of the trunk's line.
    bool trunks{true};        ///< Whether the coarse level runs at all.
};

/**
 * @brief Whether a macro cell carries a trunk river.
 *
 * The same bounded-basin rule as @ref isRiverCoarseCell, one level up: the reach
 * is @ref EndlessRiverParams::trunkRadius macro cells, which is
 * `trunkRadius * trunkCells * coarseCells` world cells — a basin a tributary
 * rule could never afford to look at.
 *
 * @param params World parameters.
 * @param rivers How a river is decided.
 * @param macroX Macro column, absolute.
 * @param macroZ Macro row, absolute.
 * @return true when a trunk runs through it.
 */
[[nodiscard]] bool isTrunkMacroCell(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 macroX,
                                    core::i32 macroZ);

/**
 * @brief Which macro cell a trunk flows into, as a direction index into kNeighbor8.
 * @param params World parameters.
 * @param rivers How a river is decided.
 * @param macroX Macro column, absolute.
 * @param macroZ Macro row, absolute.
 * @return Index into kNeighbor8X/Z, or 0xFFFFFFFF when the water stops here.
 */
[[nodiscard]] core::u32 trunkFlowDirection(const ChunkParams &params, const EndlessRiverParams &rivers,
                                           core::i32 macroX, core::i32 macroZ);

/**
 * @brief Whether one coarse cell carries a river, judged from its OWN window.
 *
 * The reference implementation: obviously correct and obviously slow, since it
 * samples a whole basin for a single verdict. @ref markChunkRivers computes the
 * same answers in one pass over a chunk, and the test asserts the two agree —
 * which is the only way a batched version stays a rendering of a position-only
 * function rather than a function of who asked.
 *
 * @param params World parameters.
 * @param rivers How a river is decided.
 * @param coarseX Coarse column, absolute.
 * @param coarseZ Coarse row, absolute.
 * @return true when water runs there.
 */
[[nodiscard]] bool isRiverCoarseCell(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 coarseX,
                                     core::i32 coarseZ);

/**
 * @brief Marks the river cells of one chunk.
 *
 * Works on a coarse grid covering the chunk plus @ref EndlessRiverParams::basinRadius
 * on every side, sampled once and then walked — the walks are cheap, the height
 * samples are not, and doing it the other way round costs a full basin of noise
 * evaluations per cell.
 *
 * @param params      World parameters.
 * @param rivers      How a river is decided.
 * @param coord       Chunk to mark.
 * @return A 0/1 mask of the chunk's cells, 1 where water runs.
 */
[[nodiscard]] Grid<core::u8> markChunkRivers(const ChunkParams &params, const EndlessRiverParams &rivers,
                                             ChunkCoord coord);

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

/**
 * @brief The chunk parameters both sides of the endless gate use.
 *
 * One constexpr, like @ref parityWorldRecipe and
 * @ref ecology::parityLivingRecipe. The P9 gate shipped with its parameters
 * COPIED into the kernel smoke and the host oracle — the only one of the three
 * whose halves were kept in step by review rather than by the compiler, which is
 * a promise to be careful forever and those are worth nothing.
 *
 * @return The canonical parameters.
 */
[[nodiscard]] constexpr ChunkParams parityChunkParams() noexcept
{
    ChunkParams params{};
    params.size = 24u;
    params.worldSeed = 20260728u;
    params.noise.frequency = 0.09f;
    params.noise.amplitude = 14.0f;
    params.noise.octaves = 5u;
    return params;
}

/// @brief The river parameters both sides of the endless gate use.
[[nodiscard]] constexpr EndlessRiverParams parityRiverParams() noexcept { return EndlessRiverParams{}; }

/// @brief Chunks either side of the origin the endless gate folds.
inline constexpr core::u32 kParityPatchRadius = 1u;

/**
 * @struct EndlessFoldResult
 * @brief What folding a patch of the endless world produced.
 */
struct EndlessFoldResult {
    core::u32 heightSignature{0u}; ///< FNV-1a over every cell of every chunk folded.
    core::u32 riverSignature{0u};  ///< FNV-1a over the river masks.
    core::u32 chunks{0u};          ///< Chunks visited.
    core::u32 riverCells{0u};      ///< Cells carrying water.
    core::u32 seamMismatches{0u};  ///< Height disagreements across the patch's seams.
};

/**
 * @brief Folds a square patch of the endless world, for the cross-target gate.
 *
 * The bounded world has been under the determinism contract since P7 and the
 * running simulation since P8; the endless one was verified on the host and
 * merely assumed on the target — the exact assumption this project refuses
 * everywhere else. This is what puts it under contract: same seed, same chunks,
 * same bits, on Linux and in ring 0.
 *
 * The seam count travels with the signatures on purpose. A fold proves two
 * machines agree; it says nothing about whether they agree on something correct,
 * and a chunked world that seams identically on both targets would pass a
 * signature check every time.
 *
 * @param params World parameters.
 * @param rivers How a river is decided.
 * @param radius Chunks either side of the origin; the patch is (2r+1) squared.
 * @return The signatures and the counts behind them.
 */
[[nodiscard]] EndlessFoldResult foldEndlessPatch(const ChunkParams &params, const EndlessRiverParams &rivers,
                                                 core::u32 radius);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CHUNKING_HPP
