/**
 * @file Geo.cpp
 * @brief Degrees to cells, metres to world units, in integers only.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/math/Geo.hpp>

#include <lpl/math/Cordic.hpp>

namespace lpl::math {

namespace {

/// One Q16.16 unit.
constexpr core::i64 kOne = 65536;

/**
 * Radians in one degree, as a Q32 word.
 *
 * @warning Held at Q32 and not Q16.16, and the extra bits are not decoration. Rounding pi/180 to
 * Q16.16 gives 1144 against a true 1143.9976, which is 2e-6 relative -- harmless as an angle and
 * not harmless downstream, because it reaches the longitude scale through a cosine and comes out
 * as about seventeen metres of east-west error per degree. The scale is derived once, so paying
 * for precision here costs nothing per sample.
 */
constexpr core::i64 kRadiansPerDegreeQ32 = 74961321;

/**
 * @brief Division that floors instead of truncating.
 *
 * @warning C++ truncates toward zero, which makes the two cells either side of the origin cover
 * twice the ground of every other cell -- a seam exactly at zero, in the one place a world is
 * most likely to be tested and least likely to be looked at.
 *
 * @param numerator   Dividend.
 * @param denominator Divisor; must not be zero.
 * @return The floor of the quotient.
 */
[[nodiscard]] core::i64 floorDivide(core::i64 numerator, core::i64 denominator) noexcept
{
    const core::i64 quotient = numerator / denominator;
    const core::i64 remainder = numerator % denominator;
    return (remainder != 0 && ((remainder < 0) != (denominator < 0))) ? quotient - 1 : quotient;
}

/**
 * @brief Division that rounds up.
 *
 * @warning **The inverse of a flooring map is a CEILING, and getting this wrong is invisible.** The
 * forward map floors, so the cell holding a coordinate is the largest whose edge is at or below it;
 * the edge of cell n is therefore the SMALLEST coordinate that still floors to n, which is a
 * ceiling. Using floor here returned a coordinate one raw unit inside the previous cell, so every
 * cell round-tripped to one less than itself -- measured: 115 of 115 sample cells, all off by
 * exactly one, in a direction that would have shifted an entire survey thirty metres west and
 * north.
 *
 * @param numerator   Dividend.
 * @param denominator Divisor; must not be zero.
 * @return The ceiling of the quotient.
 */
[[nodiscard]] core::i64 ceilDivide(core::i64 numerator, core::i64 denominator) noexcept
{
    const core::i64 quotient = numerator / denominator;
    const core::i64 remainder = numerator % denominator;
    return (remainder != 0 && ((remainder < 0) == (denominator < 0))) ? quotient + 1 : quotient;
}

} // namespace

ReliefProjection makeReliefProjection(const GeoProjection &projection)
{
    ReliefProjection out{};
    out.projection = projection;
    if (out.projection.metresPerCell == 0u)
        out.projection.metresPerCell = 1u;

    // Degrees to radians at Q32, then down to the Q16.16 the CORDIC speaks.
    const core::i64 radiansQ32 = static_cast<core::i64>(projection.referenceLatitude) * kRadiansPerDegreeQ32;
    const auto radians = Fixed32::fromRaw(static_cast<core::i32>(radiansQ32 >> 16));

    // Shifts and additions only: this is callable from ring 0, where cos does not exist and must not.
    core::i64 cosineRaw = static_cast<core::i64>(Cordic::cos(radians).raw());
    // A standard parallel at the pole would collapse the east-west scale to zero and divide the
    // world by nothing. Clamped rather than refused: a projection is a struct a document can hold,
    // and a document that names 90 has asked for something degenerate, not something malformed.
    if (cosineRaw < 1)
        cosineRaw = 1;

    out.metresPerDegreeLongitudeQ16 = kMetresPerDegreeLatitude * cosineRaw;
    return out;
}

core::i32 ReliefProjection::cellX(core::i32 longitudeRaw) const noexcept
{
    const core::i64 degreesRaw =
        static_cast<core::i64>(longitudeRaw) - static_cast<core::i64>(projection.originLongitudeRaw);
    // degreesRaw is Q16.16 and metresPerDegreeLongitudeQ16 is Q16.16, so the product carries two
    // scalings; the denominator undoes both and converts metres to cells in the same divide.
    const core::i64 numerator = degreesRaw * metresPerDegreeLongitudeQ16;
    const core::i64 denominator = kOne * kOne * static_cast<core::i64>(projection.metresPerCell);
    return static_cast<core::i32>(floorDivide(numerator, denominator));
}

core::i32 ReliefProjection::cellZ(core::i32 latitudeRaw) const noexcept
{
    // Southward from the north-west origin, so rows run the way the data's rows run.
    const core::i64 degreesRaw =
        static_cast<core::i64>(projection.originLatitudeRaw) - static_cast<core::i64>(latitudeRaw);
    const core::i64 numerator = degreesRaw * kMetresPerDegreeLatitude;
    const core::i64 denominator = kOne * static_cast<core::i64>(projection.metresPerCell);
    return static_cast<core::i32>(floorDivide(numerator, denominator));
}

core::i32 ReliefProjection::longitudeRawOf(core::i32 column) const noexcept
{
    if (metresPerDegreeLongitudeQ16 <= 0)
        return projection.originLongitudeRaw;
    const core::i64 metres = static_cast<core::i64>(column) * static_cast<core::i64>(projection.metresPerCell);
    // Undoes cellX exactly: that divided by (65536 * 65536 * metresPerCell) after multiplying by
    // the Q16.16 metres-per-degree, so this multiplies by 65536 * 65536 and divides by the same
    // scale. CEILED, so the value returned is the west EDGE of the column -- the smallest
    // longitude that still floors to it -- rather than a point one raw unit inside the previous
    // column. See @ref ceilDivide for what flooring here measured.
    const core::i64 degreesRaw = ceilDivide(metres * kOne * kOne, metresPerDegreeLongitudeQ16);
    return static_cast<core::i32>(static_cast<core::i64>(projection.originLongitudeRaw) + degreesRaw);
}

core::i32 ReliefProjection::latitudeRawOf(core::i32 row) const noexcept
{
    const core::i64 metres = static_cast<core::i64>(row) * static_cast<core::i64>(projection.metresPerCell);
    const core::i64 degreesRaw = ceilDivide(metres * kOne, kMetresPerDegreeLatitude);
    // Rows run southward, so a larger row is a SMALLER latitude.
    return static_cast<core::i32>(static_cast<core::i64>(projection.originLatitudeRaw) - degreesRaw);
}

Fixed32 ReliefProjection::worldHeightOf(core::i32 metres) const noexcept
{
    // THE reconciliation, and it is one line on purpose: elevation zero is mean sea level by
    // construction of the data, so it lands on the world's sea level and nowhere else.
    const core::i64 scaled =
        (static_cast<core::i64>(metres) * static_cast<core::i64>(projection.unitsPerMetre.raw()));
    const core::i64 raw = static_cast<core::i64>(projection.seaLevelUnits.raw()) + scaled;

    constexpr core::i64 kMaxRaw = 2147483647;
    constexpr core::i64 kMinRaw = -2147483647 - 1;
    if (raw > kMaxRaw)
        return Fixed32::fromRaw(static_cast<core::i32>(kMaxRaw));
    if (raw < kMinRaw)
        return Fixed32::fromRaw(static_cast<core::i32>(kMinRaw));
    return Fixed32::fromRaw(static_cast<core::i32>(raw));
}

GlobeWrap ReliefProjection::globe() const noexcept
{
    GlobeWrap out{};
    if (projection.metresPerCell == 0u || metresPerDegreeLongitudeQ16 <= 0)
        return out;
    const core::i64 cell = static_cast<core::i64>(projection.metresPerCell);
    // 360 degrees of longitude at this projection's own scale, and 180 of latitude. The Q16.16 in
    // the longitude scale is undone by the same divide that converts metres to cells.
    out.columns = static_cast<core::i32>((360 * metresPerDegreeLongitudeQ16) / (kOne * cell));
    out.rows = static_cast<core::i32>((180 * kMetresPerDegreeLatitude) / cell);
    return out;
}

bool ReliefProjection::verticalRangeFits(core::i32 lowestMetres, core::i32 highestMetres) const noexcept
{
    constexpr core::i64 kMaxRaw = 2147483647;
    constexpr core::i64 kMinRaw = -2147483647 - 1;
    const core::i64 unit = static_cast<core::i64>(projection.unitsPerMetre.raw());
    const core::i64 sea = static_cast<core::i64>(projection.seaLevelUnits.raw());

    const core::i64 low = sea + static_cast<core::i64>(lowestMetres) * unit;
    const core::i64 high = sea + static_cast<core::i64>(highestMetres) * unit;
    return low >= kMinRaw && low <= kMaxRaw && high >= kMinRaw && high <= kMaxRaw;
}

void wrapCell(const GlobeWrap &wrap, core::i32 &x, core::i32 &z) noexcept
{
    if (!wrap.valid())
        return;

    const core::i64 columns = wrap.columns;
    const core::i64 rows = wrap.rows;
    core::i64 cx = x;
    core::i64 cz = z;

    // Crossing a pole reflects the row and turns the longitude by half the globe. Looped rather
    // than done once because a body may in principle run past both poles in a single step at a
    // silly speed, and a single fold would leave it off the map -- where the sampler still
    // answers, so it would read as ground rather than as an error. Bounded so a malformed wrap
    // cannot spin here forever.
    for (core::u32 guard = 0u; guard < 8u; ++guard)
    {
        if (cz < 0)
        {
            // The cell one row north of the top row IS the top row, half a world away.
            cz = -cz - 1;
            cx += columns / 2;
            continue;
        }
        if (cz >= rows)
        {
            cz = 2 * rows - cz - 1;
            cx += columns / 2;
            continue;
        }
        break;
    }

    // East-west is the easy half. Floored modulo, not the remainder operator: C++ leaves a
    // negative dividend negative, so a body one cell west of the prime meridian would come back
    // at -1 rather than at the far east.
    cx %= columns;
    if (cx < 0)
        cx += columns;

    // A wrap so malformed that the reflection never converged leaves the row out of range; clamp
    // rather than emit a cell nothing can index.
    if (cz < 0)
        cz = 0;
    if (cz >= rows)
        cz = rows - 1;

    x = static_cast<core::i32>(cx);
    z = static_cast<core::i32>(cz);
}

void shortestDelta(const GlobeWrap &wrap, core::i32 fromX, core::i32 fromZ, core::i32 toX, core::i32 toZ,
                   core::i32 &outX, core::i32 &outZ) noexcept
{
    if (wrap.valid())
    {
        // Both ends brought onto the world first: a delta between an unwrapped cell and a wrapped
        // one is a lap of the planet plus the real distance, and the lap would survive the fold
        // below because it is a whole number of circumferences.
        wrapCell(wrap, fromX, fromZ);
        wrapCell(wrap, toX, toZ);
    }

    core::i64 dx = static_cast<core::i64>(toX) - static_cast<core::i64>(fromX);
    const core::i64 dz = static_cast<core::i64>(toZ) - static_cast<core::i64>(fromZ);

    // One rule, shared with the Fixed32 callers: see foldOntoShorterWay for why it is not written
    // out twice.
    if (wrap.valid())
        dx = foldOntoShorterWay<core::i64>(wrap.columns, dx);

    outX = static_cast<core::i32>(dx);
    outZ = static_cast<core::i32>(dz);
}

bool ReliefField::heightAt(core::i32 worldX, core::i32 worldZ, Fixed32 &out) const noexcept
{
    out = Fixed32::zero();
    if (!valid())
        return false;

    const core::i64 col = static_cast<core::i64>(worldX) - static_cast<core::i64>(originCellX);
    const core::i64 row = static_cast<core::i64>(worldZ) - static_cast<core::i64>(originCellZ);
    if (col < 0 || row < 0 || col >= static_cast<core::i64>(width) || row >= static_cast<core::i64>(height))
        return false;

    const core::i16 metres = samples[static_cast<core::usize>(row) * width + static_cast<core::usize>(col)];
    // A gap is refused rather than answered with a plausible number: the caller then falls back to
    // invented ground, which is honest, instead of walking onto a measurement nobody made.
    if (metres == kReliefNoSample)
        return false;

    out = projection.worldHeightOf(metres);
    return true;
}

Fixed32 ReliefField::weightAt(core::i32 worldX, core::i32 worldZ) const noexcept
{
    if (!valid())
        return Fixed32::zero();

    const core::i64 col = static_cast<core::i64>(worldX) - static_cast<core::i64>(originCellX);
    const core::i64 row = static_cast<core::i64>(worldZ) - static_cast<core::i64>(originCellZ);
    if (col < 0 || row < 0 || col >= static_cast<core::i64>(width) || row >= static_cast<core::i64>(height))
        return Fixed32::zero();

    if (blendCells == 0u)
        return Fixed32::one();

    // Distance to the nearest EXPOSED edge, in cells. Two things are going on and both matter:
    // the ramp is driven by the CLOSEST edge, because a corner is near two at once and taking
    // either alone would leave it at full strength while both its sides had faded; and an edge
    // that faces a resident neighbour is skipped entirely, because the fade belongs to the border
    // of the survey and not to the border of a tile. Without the second rule a tiled world is
    // cross-hatched with gentle valleys at every tile boundary.
    core::i64 nearest = -1;
    const auto consider = [&nearest](core::u32 mask, core::u32 edge, core::i64 distance) {
        if ((mask & edge) == 0u)
            return;
        if (nearest < 0 || distance < nearest)
            nearest = distance;
    };
    consider(exposedEdges, kReliefEdgeWest, col);
    consider(exposedEdges, kReliefEdgeEast, static_cast<core::i64>(width) - 1 - col);
    consider(exposedEdges, kReliefEdgeNorth, row);
    consider(exposedEdges, kReliefEdgeSouth, static_cast<core::i64>(height) - 1 - row);

    // No exposed edge at all: this tile is surrounded, so it is entirely real ground.
    if (nearest < 0)
        return Fixed32::one();

    if (nearest >= static_cast<core::i64>(blendCells))
        return Fixed32::one();

    // Integer ramp: a Fixed32 divide would round differently on the two sides of the field for no
    // gain, and the whole point of the band is that both sides agree on where the ground is.
    const core::i64 raw = (nearest * 65536) / static_cast<core::i64>(blendCells);
    return Fixed32::fromRaw(static_cast<core::i32>(raw));
}

const ReliefField *ReliefMosaic::find(core::i32 worldX, core::i32 worldZ) const noexcept
{
    // The FINEST covering tile, and ties broken on position rather than left to chance. Levels
    // overlap by design -- a distance-based budget keeps coarse ground resident under fine ground --
    // so returning the first match would make the ground depend on the order tiles were streamed in.
    const ReliefField *best = nullptr;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const ReliefField *tile = tiles[i];
        if (tile == nullptr || !tile->valid())
            continue;
        const core::i64 col = static_cast<core::i64>(worldX) - static_cast<core::i64>(tile->originCellX);
        const core::i64 row = static_cast<core::i64>(worldZ) - static_cast<core::i64>(tile->originCellZ);
        if (col < 0 || row < 0 || col >= static_cast<core::i64>(tile->width) ||
            row >= static_cast<core::i64>(tile->height))
            continue;
        if (best == nullptr || tile->level < best->level)
            best = tile;
    }
    return best;
}

const ReliefField *ReliefMosaic::findCoarserThan(core::u32 level, core::i32 worldX,
                                                 core::i32 worldZ) const noexcept
{
    // The finest tile STRICTLY coarser than `level`: the one a fine tile hands over to at its edge.
    const ReliefField *best = nullptr;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const ReliefField *tile = tiles[i];
        if (tile == nullptr || !tile->valid() || tile->level <= level)
            continue;
        const core::i64 col = static_cast<core::i64>(worldX) - static_cast<core::i64>(tile->originCellX);
        const core::i64 row = static_cast<core::i64>(worldZ) - static_cast<core::i64>(tile->originCellZ);
        if (col < 0 || row < 0 || col >= static_cast<core::i64>(tile->width) ||
            row >= static_cast<core::i64>(tile->height))
            continue;
        if (best == nullptr || tile->level < best->level)
            best = tile;
    }
    return best;
}

bool ReliefMosaic::heightAt(core::i32 worldX, core::i32 worldZ, Fixed32 &out) const noexcept
{
    out = Fixed32::zero();
    const ReliefField *fine = find(worldX, worldZ);
    if (fine == nullptr || !fine->heightAt(worldX, worldZ, out))
        return false;

    // @warning **A level boundary is a STEP in the ground, and this is what removes it.** A coarse
    // sample is the mean of the fine ones it replaces, so where a fine tile ends and a coarse one
    // takes over the surface jumps -- measured at 17 m one level up, 30 m two, 45 m three, which at
    // any distance a body can see is a visible ledge running along a square. Inside the fine tile's
    // own border band the two are mixed instead, so the handover is a slope rather than a wall.
    //
    // @warning It uses the fine tile's OWN band -- the one that already exists to fade into invented
    // ground -- rather than a second setting. A separate LOD band would be a second answer to "how
    // wide is the transition", and the two would drift.
    const Fixed32 weight = fine->weightAt(worldX, worldZ);
    if (weight.raw() >= Fixed32::one().raw())
        return true;

    const ReliefField *coarse = findCoarserThan(fine->level, worldX, worldZ);
    Fixed32 coarseHeight{};
    if (coarse == nullptr || !coarse->heightAt(worldX, worldZ, coarseHeight))
        return true; // Nothing coarser underneath: the caller blends this against invented ground.

    out = out * weight + coarseHeight * (Fixed32::one() - weight);
    return true;
}

Fixed32 ReliefMosaic::weightAt(core::i32 worldX, core::i32 worldZ) const noexcept
{
    // @warning The STRONGEST weight across every covering tile, not the finest tile's own. This
    // number answers "how much of this cell is measured at all", and a cell deep inside a coarse
    // tile is entirely measured even when it sits in a fine tile's fading border. Taking the finest
    // tile's weight instead would fade real ground into invented ground in the middle of a survey,
    // wherever a fine tile happened to end -- which is exactly the ledge the blend above removes,
    // reintroduced one function later.
    Fixed32 best = Fixed32::zero();
    for (core::u32 i = 0u; i < count; ++i)
    {
        const ReliefField *tile = tiles[i];
        if (tile == nullptr || !tile->valid())
            continue;
        const core::i64 col = static_cast<core::i64>(worldX) - static_cast<core::i64>(tile->originCellX);
        const core::i64 row = static_cast<core::i64>(worldZ) - static_cast<core::i64>(tile->originCellZ);
        if (col < 0 || row < 0 || col >= static_cast<core::i64>(tile->width) ||
            row >= static_cast<core::i64>(tile->height))
            continue;
        const Fixed32 here = tile->weightAt(worldX, worldZ);
        if (here.raw() > best.raw())
            best = here;
    }
    return best;
}

core::u32 planReliefResidency(const ReliefResidencyParams &params, core::i32 eyeCellX, core::i32 eyeCellZ,
                              ReliefTileRequest *out, core::u32 capacity) noexcept
{
    if (out == nullptr || capacity == 0u || params.tileCells == 0u || params.levels == 0u)
        return 0u;

    core::u32 written = 0u;
    for (core::u32 level = 0u; level < params.levels; ++level)
    {
        const core::u32 scale = reliefLevelScale(level);
        // A tile at this level covers `tileCells * scale` level-0 cells, so the eye's tile index is
        // in THIS lattice. Floored, or the two tiles either side of the origin would both be tile
        // zero and the plan would ask for one tile where it needs two.
        const core::i64 span = static_cast<core::i64>(params.tileCells) * scale;
        const core::i64 eyeTileX = floorDivide(eyeCellX, span);
        const core::i64 eyeTileZ = floorDivide(eyeCellZ, span);
        const core::i32 radius =
            static_cast<core::i32>(level == 0u ? params.fineRadiusTiles : params.radiusPerLevel);

        // Nearest first, ring by ring, so a budget that truncates loses the FARTHEST tile. A plan
        // whose overflow depended on iteration order would give a machine with less memory a hole
        // under its own feet rather than at the horizon.
        for (core::i32 ring = 0; ring <= radius; ++ring)
        {
            for (core::i32 dz = -ring; dz <= ring; ++dz)
            {
                for (core::i32 dx = -ring; dx <= ring; ++dx)
                {
                    // Only the ring's own edge: the interior was emitted by a smaller ring, and
                    // asking for a tile twice would spend budget on ground already covered.
                    const core::i32 ax = dx < 0 ? -dx : dx;
                    const core::i32 az = dz < 0 ? -dz : dz;
                    if ((ax > az ? ax : az) != ring)
                        continue;
                    if (written >= capacity)
                        return written;
                    out[written].level = level;
                    out[written].tileX = static_cast<core::i32>(eyeTileX) + dx;
                    out[written].tileZ = static_cast<core::i32>(eyeTileZ) + dz;
                    ++written;
                }
            }
        }
    }
    return written;
}

} // namespace lpl::math
