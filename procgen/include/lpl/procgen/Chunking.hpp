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
#    include <lpl/math/Geo.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>
#    include <lpl/procgen/WaveFunctionCollapse.hpp>

namespace lpl::procgen {

/**
 * @struct ChunkCoord
 * @brief Which chunk, in chunk units (not cells).
 */
struct ChunkCoord {
    core::i32 x{0}; /**< Chunk column; may be negative. */
    core::i32 z{0}; /**< Chunk row; may be negative. */

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
    core::u32 size{32u};        /**< Cells per chunk edge. */
    core::u32 worldSeed{1337u}; /**< The one seed the whole world comes from. */
    NoiseParams noise{};        /**< Terrain sampling; its own seed is ignored. */

    /**
     * Real ground, when there is any, displacing @ref noise as the lowest frequency.
     *
     * @warning A MOSAIC and not a lone field, even for a world with one tile. Two ways to attach
     * ground would be two code paths that could disagree about what the ground is; a mosaic of one
     * behaves exactly as a lone field did, and gate P21 folding unchanged is what says so.
     *
     * @warning **Null is the whole world today, and must stay costless.** A world with no survey
     * behind it takes exactly the path it always took -- @ref sampleWorldHeight returns the noise
     * and nothing else -- so every signature already under a parity gate is unmoved by this field
     * existing. That is asserted by the gates rather than argued here.
     *
     * @warning A raw pointer and not a value: in ring 0 this points into a section of a baked image
     * that was never copied, and the field itself is non-owning for the same reason. Its lifetime
     * is the image's, which outlives every chunk.
     */
    const math::ReliefMosaic *relief{nullptr};

    /**
     * The high frequencies real relief does not have.
     *
     * @warning **This is what makes measured ground octave ZERO rather than a replacement.** Thirty-metre
     * samples carry a mountain range and cannot carry a boulder, so @ref noise is displaced by the
     * real ground and this layer is added back on top. Declared as its own noise layer rather than
     * as a scale factor on @ref noise, because scaling a five-octave field down keeps its low
     * octaves too -- gentle, wrong, and indistinguishable from real terrain that happens to be
     * smooth.
     *
     * @warning **Silent by default, and the default had to be written out.** `NoiseParams` carries an
     * amplitude of sixteen, so a world that said nothing about detail was getting sixteen units of
     * roughness nobody asked for -- measured ground, quietly overwritten by invention. A caller who
     * declares real relief and leaves this alone has asked for real relief.
     *
     * @warning It lives here rather than on the field because a field is a SURVEY, and how rough a
     * world is on top of its survey is a decision about the world.
     */
    NoiseParams reliefDetail{.amplitude = 0.0f};

    /**
     * What happens when a body walks off the edge of the world.
     *
     * @warning **Applied BEFORE anything else reads the coordinate, which is what makes the seam
     * disappear rather than merely move.** Wrapping the cell first means the column just past the
     * last one IS the first one -- same cell, same survey lookup, same noise lattice -- so the
     * ground matches by construction instead of by two surfaces happening to agree. Wrapping only
     * the relief and leaving the noise unwrapped would put a cliff of invented terrain along the
     * antimeridian of an otherwise closed world.
     *
     * @warning Unset by default: a region-sized world has real edges, and closing it would join two
     * places that are nowhere near each other.
     *
     * @warning A pole is a singularity in any cylindrical projection, so the noise pattern is
     * discontinuous across it however this is done. The GROUND is not, because the survey is
     * continuous there; the invented layer simply cannot be, and that is a property of the
     * projection rather than of this field.
     */
    math::GlobeWrap globe{};
};

/**
 * @brief The seed a chunk's discrete passes should use.
 *
 * Derived from the world seed and the chunk coordinates, so it is reproducible
 * from the coordinates alone — no chunk has to have been generated before.
 *
 * Hash the coordinates rather than combine them arithmetically: chunk (1, 0) and
 * (0, 1) must not share a seed, and a simple sum or xor would give them one.
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
 * The apron is iterations PLUS ONE, and the extra cell is not caution — it is the
 * measurement. Thermal erosion moves material one cell per pass, so N passes
 * reach N cells; but the apron's own outer ring is itself wrong (it has no
 * neighbours beyond), and that error marches inward one cell per pass too.
 * An apron of exactly N left 9 cells of 5184 disagreeing with the unchunked
 * computation — the corners, where the two fronts meet. N+1 leaves none.
 *
 * @warning HYDRAULIC erosion is not bounded this way. Water runs downhill for as
 *          far as the slope lasts, so a droplet's influence is the length of a
 *          river, not the iteration count — no finite apron makes it exact. It
 *          stays a bounded-world pass, and pretending otherwise would put a
 *          different world on either side of every border.
 *
 * @param params     World parameters.
 * @param coord      Chunk to generate.
 * @param iterations Thermal iterations; the apron is one cell wider.
 * @param talus      Slope above which material slides.
 * @return The eroded chunk, of size `params.size` squared.
 */
[[nodiscard]] Heightfield generateErodedChunkTerrain(const ChunkParams &params, ChunkCoord coord, core::u32 iterations,
                                                     core::f32 talus = 0.6f);

/**
 * @struct ReliefBlend
 * @brief What it takes to lay measured ground over invented ground.
 *
 * @warning **Extracted so there is exactly ONE blend.** The streamed world reads it through
 * @ref sampleWorldHeight and the bounded world through `WorldBuilder::displaceWithRelief`, and the
 * two must agree at every cell or a map and the ground it describes are two different places. A
 * second copy is how a chunked world drifts from the unchunked one -- the failure this file's own
 * header already warns about for the noise layer.
 */
struct ReliefBlend {
    const math::ReliefMosaic *mosaic{nullptr}; /**< The survey, or null for a world without one. */
    NoiseParams detail{.amplitude = 0.0f};     /**< The high frequencies a survey cannot carry. */
    core::u32 seed{0u};                        /**< The world seed the detail layer derives from. */
};

/**
 * @brief Lays measured ground over invented ground at one cell.
 *
 * With no survey behind the world it returns @p invented: exactly the path it always took, so
 * nothing already folded moves because this branch exists.
 *
 * The weight is asked for AFTER the sample, because a cell can be inside the field and still be a
 * gap in the survey -- and a gap must fall back to invented ground at full strength rather than to
 * a fade towards nothing.
 *
 * Real ground is the lowest frequency; the detail layer is what thirty-metre samples cannot carry.
 * Added only where the real ground counts, so the border band does not gain roughness the invented
 * side does not have.
 *
 * Linear across the band. A body walking out of the survey must not meet a cliff made of whatever
 * the two surfaces happened to differ by.
 *
 * @param blend    The survey and its detail layer.
 * @param worldX   World column, already wrapped if the world is closed.
 * @param worldZ   World row.
 * @param invented What the generator would have produced here.
 * @return The ground, blended across the survey's border band.
 */
[[nodiscard]] math::Fixed32 applyRelief(const ReliefBlend &blend, core::i32 worldX, core::i32 worldZ,
                                        math::Fixed32 invented);

/**
 * @brief Height at one world cell, without generating its chunk.
 *
 * The coordinate is wrapped first, before anything reads it: on a closed world the cell past the
 * last column IS the first column, so the survey lookup and the noise lattice both land on the same
 * cell and the seam does not exist rather than being hidden. Costless when the world has real edges.
 *
 * World coordinates, not chunk-local ones. This single choice is what makes the whole scheme
 * seamless: the shared edge of two chunks is the same world position, so it evaluates to the same
 * height in both.
 *
 * The relief goes through @ref applyRelief, one blend shared with the bounded builder: see
 * @ref ReliefBlend for why it is not written twice.
 *
 * @param params  World parameters.
 * @param worldX  World column.
 * @param worldZ  World row.
 * @return The elevation there.
 */
[[nodiscard]] math::Fixed32 sampleWorldHeight(const ChunkParams &params, core::i32 worldX, core::i32 worldZ);

/**
 * @brief Ground moisture at one world cell, in [0, 1].
 *
 * The climate layer @ref generateChunkTerrain classifies biomes with, asked at a
 * single cell. Promoted out of that loop rather than copied beside it because a
 * SECOND moisture field would be a world whose caves are wet where its forests are
 * dry — and the two would drift the first time either was retuned.
 *
 * @warning Derived from `params.noise.frequency` and NOT from its amplitude, base height or
 * shaping. Those are instructions for a heightfield; applied to weather they say
 * nothing, and inheriting them is what once pinned this axis to zero over an entire
 * world.
 *
 * @param params World parameters.
 * @param worldX World column.
 * @param worldZ World row.
 * @return Moisture, clamped to [0, 1].
 */
[[nodiscard]] math::Fixed32 sampleWorldMoisture(const ChunkParams &params, core::i32 worldX, core::i32 worldZ);

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
    core::u32 coarseCells{4u}; /**< Fine cells per coarse cell; the river's width scale. */
    core::u32 basinRadius{7u}; /**< Coarse cells searched upstream. Cost is quadratic in it. */
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
    core::f32 seaLevel{-1.0f}; /**< Below this the water is already the sea. */

    /**
     * @name The trunk: a second, coarser level
     * @brief A bounded basin gives tributaries and nothing else — that is arithmetic,
     *        not a limitation to be tuned away.
     *
     * A river that crosses a continent needs a basin the size of a continent, and
     * sampling one per cell is unaffordable.
     *
     * So the same question is asked twice, at two scales, and the coarse answer
     * constrains the fine one: exactly the cascade rule HiGen exists to enforce.
     * The macro level is sampled sparsely (one height per @ref trunkCells coarse
     * cells) and cached, so its reach costs a few hundred samples per macro cell
     * rather than per world cell.
     * @{
     */

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

    core::u32 trunkCells{8u};   /**< Coarse cells per macro cell. */
    core::u32 trunkRadius{10u}; /**< Macro cells searched upstream. This is the reach. */
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

    core::u32 trunkWidth{1u}; /**< Coarse cells either side of the trunk's line. */
    bool trunks{true};        /**< Whether the coarse level runs at all. */
    /** @} */
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
 * @brief The upstream count that makes a target SHARE of cells carry a river.
 *
 * @warning `EndlessRiverParams::riverThreshold` is an ABSOLUTE count, and an absolute threshold
 * against a distribution that moves is this repository's most-repeated mistake — four
 * occurrences before this one. It moves here for a specific reason: `endlessPlanFromRecipe`
 * lowers the noise frequency to make landforms wide enough to walk across, and a smoother
 * terrain routes more cells through each coarse cell. The same 6 that marks 1.5% of a map
 * read from above marked **13.2%** of the world a walker gets — measured, and visible as
 * sheets of water lying across hillsides.
 *
 * So the threshold is calibrated instead of chosen: this returns the smallest count whose
 * river share is at or under @p targetShare.
 *
 * Raising the threshold can only mark fewer cells, so the share is monotonically
 * non-increasing in it and a bisection is exact rather than approximate. The upper bound is
 * the largest basin a bounded radius can hold, `(2 * basinRadius + 1)` squared: beyond it every
 * basin candidate is rejected and what is left is the cells a trunk covers, which no threshold
 * removes — so the search stops there, and returns that bound when even it marks more than
 * @p targetShare.
 *
 * **Chunk independence is preserved, and it is the whole difficulty.** A quantile over "the
 * world" does not exist — there is no total. So the calibration window is a fixed block of
 * chunks at the origin: a function of the parameters alone, identical for every chunk that
 * asks, computed ONCE when a plan is built and never per chunk. A window that followed the
 * walker would give two chunks two thresholds and a river would change its mind at a border.
 *
 * @param params      World parameters.
 * @param rivers      River parameters; its own @c riverThreshold is ignored.
 * @param targetShare Share of cells that should carry a river, in (0, 1).
 * @return The calibrated count, at least one.
 */
[[nodiscard]] core::u32 calibrateRiverThreshold(const ChunkParams &params, const EndlessRiverParams &rivers,
                                                core::f32 targetShare);

/**
 * @brief The share of cells a given threshold marks, over the calibration window.
 *
 * Exposed because the calibrator's answer is only as good as what it measured, and a caller
 * that wants to report what it got should not have to re-derive the window.
 *
 * @param params    World parameters.
 * @param rivers    River parameters, threshold included.
 * @return The share, in [0, 1].
 */
[[nodiscard]] core::f32 measureRiverShare(const ChunkParams &params, const EndlessRiverParams &rivers);

/**
 * @brief Marks the river cells of one chunk.
 *
 * Works on a coarse grid covering the chunk plus @ref EndlessRiverParams::basinRadius
 * on every side, sampled once and then walked — the walks are cheap, the height
 * samples are not, and doing it the other way round costs a full basin of noise
 * evaluations per cell.
 *
 * The window has to hold every cell that could drain into a cell of THIS chunk, or a
 * coarse cell near the border would be judged on a truncated basin and the two chunks
 * sharing it would disagree.
 *
 * Steepest descent, D8. Ties break on the lowest neighbour index, so a flat shelf routes
 * the same way on every machine and in every chunk that asks; a walk ends at a pit,
 * where the water stops.
 *
 * A walk credits a cell ONLY if it started within the basin radius OF THAT CELL. Without
 * the test the count is "walks starting anywhere in the window", and the window belongs
 * to the chunk — so the same world cell scores differently depending on which chunk
 * asked, and a river changes its mind at every border. The bound has to be centred on
 * the cell being credited, not on the caller.
 *
 * The trunk comes from the coarse level down, through the same helper the reference,
 * @ref isRiverCoarseCell, uses. A cell that is already the sea is not a river: without
 * that test every coastal cell qualifies, since the whole basin drains through it.
 *
 * @param params      World parameters.
 * @param rivers      How a river is decided.
 * @param coord       Chunk to mark.
 * @return A 0/1 mask of the chunk's cells, 1 where water runs.
 */
[[nodiscard]] Grid<core::u8> markChunkRivers(const ChunkParams &params, const EndlessRiverParams &rivers,
                                             ChunkCoord coord);

/**
 * @brief Which way water runs out of one coarse cell, as an index into kNeighbor8.
 *
 * @ref trunkFlowDirection one level down, and deliberately the same shape: steepest
 * descent over the eight neighbouring coarse centres. Nine height samples, all of them
 * a function of the position alone — so two chunks sharing a river agree on which way
 * it flows, which is the property that lets a renderer point a current at a seam
 * without the water changing its mind halfway across.
 *
 * The alternative — the steepest descent of the CARVED chunk heights — would have been
 * cheaper and wrong: a cell on the chunk's edge has neighbours in the next chunk, so it
 * would be judged on a truncated stencil and the two chunks would disagree.
 *
 * A neighbour must drop strictly more than the best so far, so ties break on the lowest
 * neighbour index: a flat reach has to route the same way on every machine and in every
 * chunk that asks.
 *
 * @param params  World parameters.
 * @param rivers  How a river is decided.
 * @param coarseX Coarse column, absolute.
 * @param coarseZ Coarse row, absolute.
 * @return Index into kNeighbor8X/Z, or 0xFFFFFFFF where the water stops.
 */
[[nodiscard]] core::u32 coarseFlowDirection(const ChunkParams &params, const EndlessRiverParams &rivers,
                                            core::i32 coarseX, core::i32 coarseZ);

/**
 * @brief Which way the water runs at each cell of a chunk that carries river.
 *
 * Computed once at generation and kept, because a renderer needs it per cell per frame
 * and a coarse verdict costs nine noise samples: derived at draw time it would be some
 * hundreds of fBm evaluations per frame for water that has not moved.
 *
 * One verdict per coarse cell, memoised: nine noise samples each, and every fine cell of
 * a coarse cell shares its answer. Asking per fine cell would cost the square of the
 * coarse ratio for the same result.
 *
 * @param params World parameters.
 * @param rivers How a river is decided.
 * @param coord  Chunk to describe.
 * @param mask   That chunk's river mask, from @ref markChunkRivers.
 * @return A @ref FlowDirection: kNeighbor8 indices, @ref kNoFlow where no water runs.
 */
[[nodiscard]] FlowDirection markChunkRiverFlow(const ChunkParams &params, const EndlessRiverParams &rivers,
                                               ChunkCoord coord, const Grid<core::u8> &mask);

/**
 * @brief Checks that two neighbouring chunks agree along their shared edge.
 *
 * The two edge cells are one world cell apart, never the same one, so there is nothing to
 * compare them to each other. What has to hold is that EACH chunk's edge cell equals what the
 * world function says at that chunk's own world position — checked on both sides, because a
 * chunk that agreed with the world function only on the side it was queried from would still
 * leave a step at the seam.
 *
 * Only 4-adjacent chunks share an edge; anything else has no seam to check and counts zero.
 *
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
 *                      (index into kNeighbor4X/Z: 0 east, 1 west, 2 south, 3 north).
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
[[nodiscard]] constexpr ChunkParams parityChunkParams() noexcept;

} // namespace lpl::procgen

#    include "Chunking.inl"

#endif // LPL_PROCGEN_CHUNKING_HPP
