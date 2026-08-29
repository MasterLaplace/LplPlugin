/**
 * @file Geo.hpp
 * @brief Putting a place on the ground: degrees to world cells, metres to world units.
 *
 * @warning **The projection is AUTHORITATIVE, and it is not neutral.** Where a place lands decides
 * where a body walks and how long it takes to arrive, so this arithmetic sits under the same
 * contract as everything else that decides state: integer, Fixed32, no libm, bit-identical between
 * the Linux oracle and ring 0. It is declared in the recipe rather than hard-coded for the reason
 * every other pass is: a world built with one projection and read back with another is two worlds.
 *
 * @warning **It lives in `math` because that is the only layer both sides can see, and the layering
 * is not a detail here.** A map projection is a coordinate transform between two number systems,
 * which is exactly what this module is for, and its only dependencies are Fixed32 and CORDIC --
 * both here. The alternatives were each a cycle or a violation: `procgen` cannot be seen by
 * LplKnowledge, which links core, math, memory and history and nothing else; `history` already
 * depends on `procgen`, so putting it there and having `procgen` read it closes a loop. A first
 * attempt did put it in `procgen`, and LplKnowledge stopped compiling the moment the bridge tried
 * to include it -- the same inter-repository break already paid for once, when `history/Parity.cpp`
 * reached for `ecs`.
 *
 * @warning **A cell index is an `i32`, never a Fixed32, and that is a deliberate escape from a whole
 * class of bug.** Q16.16 saturates at 32767, and one degree of longitude at thirty-metre cells is
 * already 3591 cells -- so nine degrees of world would overflow a Fixed32 coordinate. This module
 * answers in integer cells, which `sampleWorldHeight` and `ChunkCoord` already speak, and keeps
 * fixed point for the fraction inside a cell, where it is bounded by construction. This is the same
 * decision `ecs::WorldPosition` takes for bodies, taken here for geography.
 *
 * @warning **The two sea levels are reconciled HERE, once, and asserted.** Elevation data is referenced
 * to the geoid: zero metres IS mean sea level, by construction of the product. A world says where
 * its sea is in its own units, through `WorldRecipe::biomes::seaLevel`. So the projection maps
 * elevation zero ONTO that number and nowhere else -- @ref ReliefProjection::worldHeightOf is the
 * single place the two meet. This repository has paid the "two answers to where the ground is"
 * pattern twice already; the third time it is a function with a test rather than a convention.
 *
 * ### The shape of the projection, and what it costs
 *
 * Equidistant cylindrical with a standard parallel: a cell is a fixed number of metres, north-south
 * everywhere and east-west at one stated latitude. A degree of longitude shrinks with the cosine of
 * latitude, so a region far from its reference parallel comes out stretched. Chosen over Mercator
 * because Mercator preserves angle and destroys area, and a walking simulation cares which of two
 * roads is longer; chosen over per-row cosine because a grid whose column spacing varies by row is
 * not a grid, and every pass downstream assumes it is one.
 *
 * @warning The distortion is REAL and bounded rather than absent: at ten degrees from the reference
 * parallel around 40N the east-west scale is off by about 12%. That is fine for a region and wrong
 * for a hemisphere, which is why @ref GeoProjection carries the parallel it was built for instead
 * of pretending to be global.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_MATH_GEO_HPP
#    define LPL_MATH_GEO_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::math {

/**
 * Metres in one degree of latitude.
 *
 * @warning A spherical mean, not an ellipsoidal truth: the real figure runs from 110 574 m at the
 * equator to 111 694 m at the pole, a spread of about 1%. Stated rather than hidden because a
 * kilometre of drift across a continent is the kind of error that looks like a slightly different
 * coastline instead of like a bug. Refining it means an ellipsoidal series, which means either a
 * transcendental or a table, and neither is worth 1% here.
 */
inline constexpr core::i64 kMetresPerDegreeLatitude = 111320;

/**
 * @struct GeoProjection
 * @brief How a corpus's degrees become the world's cells.
 *
 * @warning Every field is part of the world's identity. Two runs that disagree on any of them put the
 * same place in two different valleys, which is not a rendering difference -- it is a different
 * journey with different arrivals.
 */
struct GeoProjection {
    /**
     * NORTH-west corner of the world, in degrees, as a raw Q16.16 word.
     *
     * @warning **North-west and not south-west, which is the opposite of how tiles are named, and the
     * choice removes a flip rather than hiding one.** Every elevation product on earth stores its
     * north row first, so a world whose origin is its north-west corner indexes relief directly:
     * cell row zero is data row zero. The alternative -- a south-west origin, matching tile names --
     * needs a north-south mirror somewhere between the tile and the sampler, and a mirror that
     * exists in one of two code paths is a world whose mountains sit opposite its own coastline
     * while still looking exactly like terrain. Converting a tile's southern edge into this is one
     * addition, done once at construction.
     *
     * @warning Raw words rather than floats because this is exactly what `knowledge::GazetteerEntryV1`
     * carries, and re-encoding a coordinate on the way in is where a last-bit disagreement is born.
     */
    core::i32 originLatitudeRaw{0};
    core::i32 originLongitudeRaw{0}; ///< West edge, degrees, raw Q16.16.

    /**
     * Latitude the east-west scale is exact at, in whole degrees.
     *
     * Whole degrees on purpose: it is a choice about a region, not a measurement, and a fractional
     * standard parallel invites the belief that the projection is more accurate than it is.
     */
    core::i32 referenceLatitude{0};

    core::u32 metresPerCell{30u}; ///< Ground a cell covers. Thirty is about one arc-second.

    /**
     * World units per metre of elevation.
     *
     * @warning This is where a mountain range becomes either scenery or a wall. It is separate from
     * @ref metresPerCell because vertical exaggeration is a legitimate artistic choice -- a real
     * landscape at true scale reads as flat from eye height -- while horizontal scale is not.
     */
    Fixed32 unitsPerMetre{Fixed32::one()};

    /**
     * World height that elevation zero maps to.
     *
     * @warning THE reconciliation. It must be the recipe's `biomes.seaLevel` and nothing else: the
     * data's zero is mean sea level by construction, so anything else here draws the coastline in
     * a place the classifier does not agree is a coastline.
     */
    Fixed32 seaLevelUnits{};
};

/**
 * @struct GlobeWrap
 * @brief What happens when a body walks off the edge of the world.
 *
 * @warning **A sphere is NOT a torus, and this is the whole reason the struct exists.** Wrapping both
 * axes the way a tiled texture wraps -- east to west AND north to south -- puts the Arctic directly
 * against the Antarctic, so a body walking north out of Siberia arrives in the Southern Ocean. On a
 * sphere, crossing the north pole sends you back SOUTH along the OPPOSITE meridian. East-west is the
 * easy half and is a plain modulo; north-south is a reflection plus half a turn.
 *
 * @warning Unset is the whole world today and must stay costless: a region-sized world has real edges
 * and wrapping it would join two places that are nowhere near each other.
 *
 * @warning **Exact in the projection's own coordinates, which is what "walk off the east edge and come
 * back on the west" means.** The geographic distortion of an equidistant cylindrical projection far
 * from its standard parallel is a separate, already-stated property -- a whole-earth world wants a
 * parallel near the equator, and the columns it wraps at are that parallel's.
 */
struct GlobeWrap {
    core::i32 columns{0}; ///< Cells around the globe. Zero disables wrapping entirely.
    core::i32 rows{0};    ///< Cells from pole to pole.

    /// @return Whether this describes a closed world.
    [[nodiscard]] constexpr bool valid() const noexcept { return columns > 0 && rows > 0; }
};

/**
 * @struct ReliefProjection
 * @brief A projection with its longitude scale already resolved.
 *
 * Built once by @ref makeReliefProjection because the cosine of the standard parallel is a CORDIC
 * evaluation, and doing it per sample would be both slower and -- worse -- a second opportunity to
 * get a different answer.
 */
struct ReliefProjection {
    GeoProjection projection{};

    /**
     * Metres in one degree of longitude at the standard parallel, as a Q16.16 word.
     *
     * Held at fixed-point precision rather than rounded to metres: rounding here is a systematic
     * east-west stretch of up to half a metre per degree, which accumulates across a region.
     */
    core::i64 metresPerDegreeLongitudeQ16{0};

    /**
     * @brief Cell column a longitude falls in.
     *
     * @param longitudeRaw Degrees east, raw Q16.16.
     * @return The cell column; negative west of the origin.
     */
    [[nodiscard]] core::i32 cellX(core::i32 longitudeRaw) const noexcept;

    /**
     * @brief Cell row a latitude falls in.
     *
     * @warning Rows increase SOUTHWARD, away from the north-west origin, so a region's rows run in the
     * same direction as the rows of the data it is made of. See @ref GeoProjection::originLatitudeRaw
     * for why that is worth an unfamiliar corner.
     *
     * @param latitudeRaw Degrees north, raw Q16.16.
     * @return The cell row; negative north of the origin, which is outside the region.
     */
    [[nodiscard]] core::i32 cellZ(core::i32 latitudeRaw) const noexcept;

    /**
     * @brief Longitude at a cell column's WEST edge.
     *
     * @warning The inverse lives beside the forward map on purpose: a resampler that derived cell
     * centres with its own arithmetic would be a second projection, and the samples would then be
     * laid down under one and read under another.
     *
     * @warning The EDGE and not the centre, so that @ref cellX round-trips: the west edge of column
     * n is the smallest longitude that falls in column n. A half-cell offset here shifts an entire
     * survey by fifteen metres, which reads as a slightly different coastline.
     *
     * @param column Cell column.
     * @return Degrees east, raw Q16.16.
     */
    [[nodiscard]] core::i32 longitudeRawOf(core::i32 column) const noexcept;

    /**
     * @brief Latitude at a cell row's NORTH edge.
     *
     * @param row Cell row, increasing southward.
     * @return Degrees north, raw Q16.16.
     */
    [[nodiscard]] core::i32 latitudeRawOf(core::i32 row) const noexcept;

    /**
     * @brief World height an elevation in metres maps to.
     *
     * @warning Saturates at the Fixed32 range rather than wrapping. A wrap would put the top of a
     * mountain below the sea, which reads as terrain and not as an error; saturation reads as a
     * plateau, which is visibly wrong. Use @ref verticalRangeFits before baking to find out
     * whether it will happen at all.
     *
     * @param metres Elevation above mean sea level; negative is bathymetry.
     * @return The world height.
     */
    [[nodiscard]] Fixed32 worldHeightOf(core::i32 metres) const noexcept;

    /**
     * @brief Whether a range of elevations survives the vertical scale without saturating.
     *
     * @param lowestMetres  Deepest sample.
     * @param highestMetres Highest sample.
     * @return false when @ref worldHeightOf would clamp either end.
     */
    [[nodiscard]] bool verticalRangeFits(core::i32 lowestMetres, core::i32 highestMetres) const noexcept;

    /**
     * @brief The closed world this projection describes, if it were carried all the way round.
     *
     * @warning **Derived here rather than written down by a caller**, because it is a function of the
     * projection and nothing else: how many cells go round is 360 degrees of longitude at this
     * projection's own scale, divided by its cell size. A hand-computed figure beside it would be a
     * second answer to how big the world is, and the two would part the first time a cell size
     * changed.
     *
     * @warning It closes at the STANDARD PARALLEL, not at the equator. That is the width this
     * projection's map actually has -- an equidistant cylindrical sheet is as wide as its parallel
     * is long -- so the wrap is exact in the coordinates everything else uses.
     *
     * @return The wrap; a projection with a degenerate scale yields an invalid one.
     */
    [[nodiscard]] GlobeWrap globe() const noexcept;
};

/**
 * @brief Resolves a projection's longitude scale.
 *
 * @warning The cosine comes from CORDIC, which is shifts and additions -- so this is exact on both
 * targets and callable from ring 0, where `cos` does not exist and must not.
 *
 * @param projection What to resolve.
 * @return The resolved form.
 */
[[nodiscard]] ReliefProjection makeReliefProjection(const GeoProjection &projection);


/**
 * @brief Brings a cell that walked off the edge back onto the world.
 *
 * @warning Idempotent and total: wrapping an already-wrapped cell changes nothing, and a cell any
 * distance outside comes back. A version that folded once would leave a body that ran two laps
 * somewhere off the map, and "somewhere off the map" reads as ground rather than as an error
 * because the sampler answers there.
 *
 * @param wrap  The closed world, or an invalid one to do nothing.
 * @param x     Column, adjusted in place.
 * @param z     Row, adjusted in place.
 */
void wrapCell(const GlobeWrap &wrap, core::i32 &x, core::i32 &z) noexcept;

/**
 * @brief The shorter way from one cell to another on a closed world.
 *
 * @warning **Without this, a closed world is worse than an open one.** Two places either side of the
 * antimeridian are neighbours, and a plain subtraction makes them almost a full circumference apart
 * -- so anything choosing "the nearest place" picks the wrong one, and anything walking a heading
 * sets off the long way round the planet. That is not a rendering artefact: it is a body spending a
 * voyage going the wrong direction, which looks exactly like a body on a very long journey.
 *
 * @warning **East-west only. A pole crossing is deliberately NOT offered as a shortcut**, and the
 * reason is that it is not expressible as one: going over a pole reverses the direction of travel
 * AND translates the longitude by half the globe, so a caller stepping along the returned vector
 * would not arrive. Two points at high latitude on opposite meridians really are closer over the
 * pole, and finding that route is a ROUTER's job -- something that can emit a path with a turn in
 * it -- not a delta's.
 *
 * @param wrap  The closed world; an invalid one gives a plain subtraction.
 * @param fromX Start column.
 * @param fromZ Start row.
 * @param toX   End column.
 * @param toZ   End row.
 * @param outX  Receives the column delta, shortest way round.
 * @param outZ  Receives the row delta.
 */
void shortestDelta(const GlobeWrap &wrap, core::i32 fromX, core::i32 fromZ, core::i32 toX, core::i32 toZ,
                   core::i32 &outX, core::i32 &outZ) noexcept;

/**
 * @brief Folds a separation onto the shorter way round a closed span.
 *
 * @warning **The rule lives here ONCE and is used in two unit systems.** Cells are integers and world
 * positions are Fixed32, so the shorter way round has to be computed in both -- and writing it twice
 * is how two answers to "how far apart are these" are born, one of which would be used to pick a
 * destination and the other to walk to it. @ref shortestDelta is the cell-space caller.
 *
 * @warning Anything past half the span is shorter the other way. Callers with an open world pass a
 * span of zero and get their separation back untouched.
 *
 * @tparam T An integer or Fixed32.
 * @param span      Width of the closed world, or zero for an open one.
 * @param separation The raw difference.
 * @return The separation, folded.
 */
template <typename T> [[nodiscard]] constexpr T foldOntoShorterWay(T span, T separation) noexcept
{
    if (!(span > T{}))
        return separation;
    // @warning **Doubled rather than halved, and that is not a style choice.** Writing `span / T{2}`
    // constructs a Fixed32 from the RAW word 2 -- two sixty-five-thousandths, not two -- so the
    // half came out enormous, the comparison inverted, and an ordinary inland separation of 120 was
    // rewritten as -880. That is the `Fixed32{N}` trap this repository has already audited and
    // fixed in five places, written again here the moment the arithmetic went generic. Adding a
    // value to itself needs no literal at all, so the trap cannot recur.
    //
    // @warning **Looped, because a separation can be WIDER than the world.** A single subtraction only
    // folds one lap, so a gap of 300 across a world 120 wide came back as 180 -- still most of the
    // way round, and still wrong in the direction that makes a body walk the long way. `shortestDelta`
    // never sees this because it wraps both ends first; a caller holding two raw positions does.
    // Bounded so a malformed span cannot spin here, the same guard `wrapCell` carries.
    T folded = separation;
    for (core::u32 guard = 0u; guard < 64u; ++guard)
    {
        const T doubled = folded + folded;
        if (doubled > span)
            folded = folded - span;
        else if (T{} - doubled > span)
            folded = folded + span;
        else
            break;
    }
    return folded;
}

/**
 * @struct ReliefField
 * @brief Real ground, resampled onto world cells, ready to be read one cell at a time.
 *
 * @warning **One sample per CELL, resampled at bake time and never at run time.** An arc-second is
 * about 30.87 metres at the equator and shorter east-west everywhere else, so it does not line up
 * with a thirty-metre cell -- something has to resample. Doing it here would put a resampler in
 * ring 0 and, worse, would be a SECOND resampler beside the baker's: two answers to what the ground
 * is at a cell, which is the pattern this repository has already paid for three times. The baker
 * owns it, using this same @ref ReliefProjection, so the two cannot disagree.
 *
 * @warning **Non-owning.** In ring 0 these samples are a window onto a section of a baked image that
 * was never copied. A field that owned its samples would need a heap on a path that has none.
 */
struct ReliefField {
    const core::i16 *samples{nullptr}; ///< Row-major, NORTH row first; @ref kReliefNoSample for a gap.
    core::u32 width{0u};               ///< Columns, in world cells.
    core::u32 height{0u};              ///< Rows, in world cells.
    core::i32 originCellX{0};          ///< World cell of column zero.
    core::i32 originCellZ{0};          ///< World cell of row zero.

    ReliefProjection projection{}; ///< How its metres become world units. Same one the baker used.

    /**
     * Cells over which real ground gives way to invented ground at the field's edge.
     *
     * @warning Not decoration and not a taste setting: without it the boundary of the data is a
     * vertical cliff of whatever the two surfaces happen to differ by, and a body walking east out
     * of Greece falls off the edge of the survey. Zero is legal and means a hard border, which is
     * only correct for a world that refuses to be walked past its data.
     */
    core::u32 blendCells{64u};

    /**
     * Which of this tile's four edges face nothing, and therefore fade.
     *
     * @warning **The fade belongs to the edge of the SURVEY, never to the edge of a tile, and this
     * mask is the whole difference between a tiled world and a quilt.** A tile that faded on every
     * side would put a band of half-invented ground around itself, so a world made of tiles would
     * come out cross-hatched with visible seams at every boundary -- and each seam would look like
     * terrain, because a fade between two similar surfaces is a gentle valley rather than an error.
     * Where a neighbour is resident the weight stays at one right up to the boundary, so the two
     * tiles meet exactly.
     *
     * @warning All four exposed is the default because a lone field IS a world whose every side faces
     * nothing -- which keeps a single-tile world behaving exactly as it did before tiling existed.
     */
    core::u32 exposedEdges{0xFu};

    /**
     * Reduction level: zero is full resolution, each step covers four times the ground.
     *
     * @warning **Carried so a mosaic can prefer the FINEST tile covering a cell, deterministically.**
     * Coarse and fine tiles legitimately overlap -- that is what a distance-based budget produces --
     * and without a level the tile that answers is whichever was streamed in first. Which ground a
     * world has would then depend on the order tiles happened to arrive, which is not a property of
     * the world at all, and two targets would disagree the moment their loaders raced.
     */
    core::u32 level{0u};

    /// @return Whether this field carries anything at all.
    [[nodiscard]] bool valid() const noexcept { return samples != nullptr && width > 0u && height > 0u; }

    /**
     * @brief Height of the real ground at a world cell, in world units.
     *
     * @param worldX World column.
     * @param worldZ World row.
     * @param out    Receives the height.
     * @return false outside the field, or at a cell nobody measured.
     */
    [[nodiscard]] bool heightAt(core::i32 worldX, core::i32 worldZ, Fixed32 &out) const noexcept;

    /**
     * @brief How much the real ground counts at a world cell.
     *
     * One at the interior, zero outside, and a linear ramp across @ref blendCells in between.
     *
     * @param worldX World column.
     * @param worldZ World row.
     * @return A weight in [0, 1].
     */
    [[nodiscard]] Fixed32 weightAt(core::i32 worldX, core::i32 worldZ) const noexcept;
};

/// @ref ReliefField::exposedEdges bits. West is -x, east is +x, north is -z, south is +z.
inline constexpr core::u32 kReliefEdgeWest = 1u << 0;
inline constexpr core::u32 kReliefEdgeEast = 1u << 1;
inline constexpr core::u32 kReliefEdgeNorth = 1u << 2;
inline constexpr core::u32 kReliefEdgeSouth = 1u << 3;

/**
 * Resident tiles a mosaic may hold at once.
 *
 * @warning A fixed array and therefore a bound, because this is read in ring 0 where there is no heap
 * and a survey is streamed rather than owned. Sixty-four thirty-metre tiles of a thousand cells a
 * side is thirty kilometres of ground in every direction, which is further than anything can see.
 */
inline constexpr core::u32 kMaxResidentReliefTiles = 64u;

/**
 * @struct ReliefMosaic
 * @brief The tiles of a survey that are resident right now.
 *
 * @warning **Non-owning, bounded, and allocation-free**, because the whole point of tiling is that the
 * earth does not fit: at thirty metres a global survey is 1.78 TB, while the window a body can
 * actually see is under a megabyte. Who decides which tiles are resident is a policy that lives with
 * whoever can do I/O; what a mosaic does is answer for the ones that are.
 *
 * @warning **It is also the only relief a world reads, and a single survey is a mosaic of one.** Two
 * ways to attach ground -- a lone field for small worlds and a mosaic for big ones -- would be two
 * code paths that could disagree about what the ground is, which is the duplication this repository
 * keeps paying for. A mosaic of one tile with all four edges exposed behaves exactly as a lone field
 * did, and gate P21 folding unchanged is what says so.
 */
struct ReliefMosaic {
    const ReliefField *tiles[kMaxResidentReliefTiles]{}; ///< Resident tiles, non-owning.
    core::u32 count{0u};                                 ///< How many.

    /// @return Whether anything is resident.
    [[nodiscard]] bool valid() const noexcept { return count > 0u; }

    /**
     * @brief Adds a tile.
     *
     * @param tile The tile; must outlive the mosaic.
     * @return false when the mosaic is full, which is a budget being hit rather than an error.
     */
    [[nodiscard]] bool add(const ReliefField *tile) noexcept
    {
        if (tile == nullptr || !tile->valid() || count >= kMaxResidentReliefTiles)
            return false;
        tiles[count++] = tile;
        return true;
    }

    /**
     * @brief The resident tile covering a world cell.
     *
     * @warning A linear scan, deliberately: sixty-four pointer comparisons is cheaper than the index
     * that would avoid them, and an index would be a second description of where the tiles are --
     * one that could disagree with the tiles themselves after a stream-in.
     *
     * @warning **The FINEST covering tile wins, and ties break on position in the mosaic.** Levels
     * overlap by design, so without a rule the answer depends on which tile was streamed in first --
     * making the ground a function of loader timing rather than of the world.
     *
     * @param worldX World column.
     * @param worldZ World row.
     * @return The tile, or nullptr where nothing is resident.
     */
    [[nodiscard]] const ReliefField *find(core::i32 worldX, core::i32 worldZ) const noexcept;

    /**
     * @brief The finest resident tile STRICTLY coarser than a level.
     *
     * The tile a fine one hands over to at its edge. Exposed because @ref heightAt blends between
     * the two, and a caller that wanted to know which pair it was blending could not otherwise ask.
     *
     * @param level  Levels at or below this are ignored.
     * @param worldX World column.
     * @param worldZ World row.
     * @return The tile, or nullptr when nothing coarser covers the cell.
     */
    [[nodiscard]] const ReliefField *findCoarserThan(core::u32 level, core::i32 worldX,
                                                     core::i32 worldZ) const noexcept;

    /**
     * @brief Height of the real ground at a world cell.
     *
     * @param worldX World column.
     * @param worldZ World row.
     * @param out    Receives the height.
     * @return false where nothing is resident, or at a cell nobody measured.
     */
    [[nodiscard]] bool heightAt(core::i32 worldX, core::i32 worldZ, Fixed32 &out) const noexcept;

    /**
     * @brief How much the real ground counts at a world cell.
     *
     * @param worldX World column.
     * @param worldZ World row.
     * @return A weight in [0, 1]; one deep inside the survey, zero outside it.
     */
    [[nodiscard]] Fixed32 weightAt(core::i32 worldX, core::i32 worldZ) const noexcept;
};

/**
 * @struct ReliefTileRequest
 * @brief One tile a residency plan asks for.
 */
struct ReliefTileRequest {
    core::u32 level{0u}; ///< Zero is full resolution; each step covers four times the ground.
    core::i32 tileX{0};  ///< Tile column, in THIS level's own lattice.
    core::i32 tileZ{0};  ///< Tile row, in this level's lattice.
};

/**
 * @struct ReliefResidencyParams
 * @brief How much ground to keep resident, and at what detail.
 */
struct ReliefResidencyParams {
    /**
     * Cells per tile side, the same at every level.
     *
     * @warning **A level boundary is a STEP in the ground, and the size of it is measured rather than
     * hoped for.** A coarse sample is the mean of the fine ones under it, so where two levels meet
     * the surface jumps. On a slope with ridges, against the fine samples it replaces:
     *
     * | level | cell   | worst step | mean step |
     * |-------|--------|-----------|-----------|
     * | L1    | 60 m   | 17 m      | 4.6 m     |
     * | L2    | 120 m  | 30 m      | 10.8 m    |
     * | L3    | 240 m  | 45 m      | 17.2 m    |
     *
     * That is what sets the radii: a boundary has to sit far enough away that its step is under a
     * pixel. At 1080p and a 60-degree field of view a pixel is about 1e-3 radians, so a 30 m step
     * needs 30 km of distance. A 1024-cell tile at 30 m is 31 km, so the defaults below put the
     * first boundary around 46 km and the second around 92 km -- comfortably beyond. Shrink
     * @ref tileCells and that margin goes with it.
     */
    core::u32 tileCells{1024u};
    core::u32 fineRadiusTiles{1u};   ///< Tiles of level 0 kept either side of the eye.
    core::u32 levels{4u};            ///< How many levels to stack, finest first.
    core::u32 radiusPerLevel{1u};    ///< Tiles either side of the eye at each coarser level.
};

/**
 * @brief Chooses which tiles of a survey should be resident.
 *
 * @warning **A pyramid, not a window, and the arithmetic is why the earth fits.** A global survey at
 * thirty metres is 1.78 TB; the level whose cells are 256 times coarser is 27 MB, small enough to
 * hold entirely. So detail is spent where it is looked at and coarse ground covers everywhere else:
 * what a body sees under its feet is a megabyte, and what it sees on the horizon is already resident.
 *
 * @warning **Nearest first, so a budget that truncates drops the FARTHEST tile rather than an arbitrary
 * one.** A plan whose overflow behaviour depended on iteration order would give two machines with
 * different budgets two different worlds, and the one with less memory would lose ground under its
 * own feet rather than at the horizon.
 *
 * @warning **Levels overlap on purpose**: coarse tiles cover the same ground the fine ones do, so a
 * fine tile arriving late or being evicted leaves coarse ground behind instead of a hole. That is
 * also why @ref ReliefMosaic::find must prefer the finest rather than the first.
 *
 * @warning It emits requests and reads nothing: choosing is arithmetic and belongs anywhere, LOADING
 * needs a filesystem and belongs to whoever has one. Same seam as everywhere else here.
 *
 * @param params   The budget.
 * @param eyeCellX Where the eye is, in level-0 cells.
 * @param eyeCellZ Where the eye is.
 * @param out      Receives the requests, nearest first.
 * @param capacity How many @p out can hold.
 * @return How many were written; equal to @p capacity when the budget truncated the plan.
 */
[[nodiscard]] core::u32 planReliefResidency(const ReliefResidencyParams &params, core::i32 eyeCellX,
                                            core::i32 eyeCellZ, ReliefTileRequest *out,
                                            core::u32 capacity) noexcept;

/**
 * @brief Ground one cell of a level covers, in level-0 cells.
 *
 * @param level Reduction level.
 * @return The factor; 1 at level zero, then 2, 4, 8.
 */
[[nodiscard]] constexpr core::u32 reliefLevelScale(core::u32 level) noexcept
{
    return level >= 16u ? 65536u : (1u << level);
}

/**
 * Sample value meaning "nobody measured this cell".
 *
 * @warning The same word the source tiles use, carried through rather than translated: a gap in a
 * survey is a fact about the survey, and filling it at bake time would make invented ground
 * indistinguishable from measured ground for everything downstream.
 */
inline constexpr core::i16 kReliefNoSample = -32768;

} // namespace lpl::math

#endif // LPL_MATH_GEO_HPP
