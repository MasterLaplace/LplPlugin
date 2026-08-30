/**
 * @file test_geo_projection.cpp
 * @brief Where a degree lands, and the one place two sea levels are allowed to meet.
 *
 * @warning Every failure here is one where a wrong answer still looks like a world: a mirrored
 * hemisphere is still terrain, a truncating divide is still a grid, a saturated mountain is still
 * a plateau. None of them raise anything at runtime, which is why they are asserted.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/math/Geo.hpp>
#include <lpl/procgen/Chunking.hpp>

#include <cstdio>

namespace {

int gChecks = 0;
int gFailures = 0;

/**
 * @brief Records one assertion.
 *
 * @param what Description.
 * @param ok   Whether it held.
 */
void check(const char *what, bool ok)
{
    ++gChecks;
    if (ok)
        return;
    ++gFailures;
    std::printf("  (fail) %s\n", what);
}

/**
 * @brief Degrees as the raw Q16.16 word a gazetteer carries.
 *
 * @param degrees Decimal degrees.
 * @return The raw word.
 */
[[nodiscard]] lpl::core::i32 deg(double degrees) { return static_cast<lpl::core::i32>(degrees * 65536.0); }

} // namespace

int main()
{
    using namespace lpl;

    // The southern Peloponnese, which is the ground the relief reader was measured against.
    math::GeoProjection spec{};
    spec.originLatitudeRaw = deg(38.0); // NORTH edge.
    spec.originLongitudeRaw = deg(21.0);
    spec.referenceLatitude = 37;
    spec.metresPerCell = 30u;
    spec.unitsPerMetre = math::Fixed32::fromFloat(0.05f);
    spec.seaLevelUnits = math::Fixed32::fromFloat(-1.0f);

    const math::ReliefProjection projection = math::makeReliefProjection(spec);

    std::printf("-- the origin is the north-west corner, and nothing is mirrored\n");
    {
        check("the origin is cell zero",
              projection.cellX(spec.originLongitudeRaw) == 0 && projection.cellZ(spec.originLatitudeRaw) == 0);

        // @warning The failure this exists for: a north-south flip produces a world that still has
        // coastlines and still has mountains, just not where its own places are. Going SOUTH from
        // the north-west origin must increase the row, the way the data's rows increase.
        check("south of the origin is a larger row", projection.cellZ(deg(37.0)) > 0);
        check("north of the origin is negative", projection.cellZ(deg(39.0)) < 0);
        check("east of the origin is a larger column", projection.cellX(deg(22.0)) > 0);
        check("west of the origin is negative", projection.cellX(deg(20.0)) < 0);
    }

    std::printf("-- a degree is the ground it really covers\n");
    {
        // A degree of latitude is 111 320 m, so at thirty-metre cells it is 3710 rows. Computed
        // here rather than read off the implementation: an expected value taken from the code
        // under test only asserts that the code agrees with itself.
        const core::i32 expectedRows = static_cast<core::i32>(111320 / 30);
        const core::i32 measuredRows = projection.cellZ(deg(37.0));
        check("a degree of latitude is its real size", measuredRows == expectedRows);

        // @warning A degree of LONGITUDE is shorter, by the cosine of the standard parallel. Getting
        // this wrong -- treating a degree as square -- stretches the world east-west by a quarter
        // at this latitude, which reads as a slightly different country rather than as an error.
        const core::i32 measuredCols = projection.cellX(deg(22.0));
        check("a degree of longitude is shorter than one of latitude", measuredCols < measuredRows);

        // cos(37 degrees) is 0.7986, so 111320 * 0.7986 / 30 = 2963 columns. Two counts either way
        // for the CORDIC's own resolution.
        const core::i32 expectedCols = static_cast<core::i32>(111320.0 * 0.798636 / 30.0);
        const core::i32 error = measuredCols > expectedCols ? measuredCols - expectedCols : expectedCols - measuredCols;
        check("and it is short by the cosine of the standard parallel", error <= 2);
    }

    std::printf("-- the divide floors, so the cell at the origin is not twice the size\n");
    {
        // @warning Truncation toward zero makes the two cells either side of zero cover twice the
        // ground of every other cell. A seam exactly at the origin, in the one place a world is
        // most likely to be tested and least likely to be looked at.
        //
        // One cell west of the origin must be cell -1 for the WHOLE cell, not just its far half.
        const core::i32 justWest = projection.cellX(spec.originLongitudeRaw - deg(0.0001));
        check("a hair west of the origin is already cell -1", justWest == -1);
        const core::i32 justNorth = projection.cellZ(spec.originLatitudeRaw + deg(0.0001));
        check("a hair north of the origin is already row -1", justNorth == -1);

        // And every cell is the same width: walking a fixed step must advance the column by a
        // constant, on both sides of zero.
        bool uniform = true;
        core::i32 previous = projection.cellX(deg(20.0));
        for (int i = 1; i <= 40; ++i)
        {
            const core::i32 here = projection.cellX(deg(20.0) + i * deg(0.05));
            if (here - previous < 148 || here - previous > 149)
                uniform = false;
            previous = here;
        }
        check("cells are the same width across the origin", uniform);
    }

    std::printf("-- the two sea levels meet HERE, and nowhere else\n");
    {
        // @warning THE reconciliation. Elevation data is referenced to the geoid, so zero metres IS
        // mean sea level by construction of the product; the world states where its sea is in its
        // own units. If these two ever disagree, the coastline is drawn in a place the biome
        // classifier does not agree is a coastline -- and this repository has already paid that
        // exact pattern twice, once as "two answers to the height of the ground" and once as
        // three answers to where the sea was in a shipped cartridge.
        check("elevation zero is exactly the world's sea level",
              projection.worldHeightOf(0).raw() == spec.seaLevelUnits.raw());

        check("land is above it", projection.worldHeightOf(100) > spec.seaLevelUnits);
        check("bathymetry is below it", projection.worldHeightOf(-100) < spec.seaLevelUnits);

        // @warning The expectation is the scale the recipe REALLY holds, not the decimal that was
        // typed. 0.05 is not representable in Q16.16 -- it stores as 3276 against an exact 3276.8 --
        // so asserting "100 m is exactly 5.0 units" asserts against a number the world does not
        // contain. The recipe's Fixed32 is the truth here; there is no more precise value to be had,
        // and rounding differently per sample would be worse than a consistent scale.
        check("and the vertical scale is the one the recipe really holds",
              projection.worldHeightOf(100).raw() == spec.seaLevelUnits.raw() + 100 * spec.unitsPerMetre.raw());

        // @warning The bound is DERIVED from the format, not chosen so today's numbers pass. A stored
        // scale is at most one raw unit below the decimal it came from, so scaling by it is wrong by
        // at most one part in `unitsPerMetre.raw()` -- and it is wrong by exactly that much no
        // matter how high the ground gets, because a multiply by a rounded constant is a rounded
        // scale and not an accumulating sum. Measured at the highest ground on earth: 2.441e-4
        // against the scale's own 2.4414e-4. An implementation that stepped instead of multiplying
        // would fail this while looking correct at every single sample.
        const core::i64 summit = projection.worldHeightOf(8849).raw() - spec.seaLevelUnits.raw();
        const core::i64 ideal = static_cast<core::i64>(8849.0 * 0.05 * 65536.0);
        const core::i64 drift = summit > ideal ? summit - ideal : ideal - summit;
        check("its error is the scale's own rounding and never more",
              drift * static_cast<core::i64>(spec.unitsPerMetre.raw()) <= ideal);
    }

    std::printf("-- a mountain that does not fit is refused, not wrapped\n");
    {
        // Everest, and the Challenger Deep, are the real bounds of the product.
        check("the true range of the earth fits at this scale", projection.verticalRangeFits(-11000, 8849));

        // @warning At a large vertical exaggeration the raw Q16.16 word overflows, and a WRAP would put
        // the top of a mountain below the sea -- which reads as terrain, not as an error. The
        // saturating path reads as a plateau, which is visibly wrong, and this predicate says so
        // before anything is baked.
        math::GeoProjection tall = spec;
        tall.unitsPerMetre = math::Fixed32::fromFloat(4.0f);
        const math::ReliefProjection steep = math::makeReliefProjection(tall);
        check("a scale that would overflow is reported", !steep.verticalRangeFits(-11000, 8849));
        check("and saturates upward rather than wrapping", steep.worldHeightOf(8849) > steep.worldHeightOf(0));
        check("and downward too", steep.worldHeightOf(-11000) < steep.worldHeightOf(0));
    }

    std::printf("-- a standard parallel at the pole degrades rather than divides by nothing\n");
    {
        math::GeoProjection polar = spec;
        polar.referenceLatitude = 90;
        const math::ReliefProjection atPole = math::makeReliefProjection(polar);
        check("the longitude scale stays positive", atPole.metresPerDegreeLongitudeQ16 > 0);
        check("and a lookup still answers", atPole.cellX(deg(22.0)) >= 0);

        math::GeoProjection zeroCell = spec;
        zeroCell.metresPerCell = 0u;
        const math::ReliefProjection clamped = math::makeReliefProjection(zeroCell);
        check("a zero-metre cell is clamped rather than dividing by zero", clamped.projection.metresPerCell >= 1u);
    }

    std::printf("-- the equator is the case where the two axes agree\n");
    {
        math::GeoProjection equator{};
        equator.originLatitudeRaw = deg(1.0);
        equator.originLongitudeRaw = deg(0.0);
        equator.referenceLatitude = 0;
        equator.metresPerCell = 30u;
        const math::ReliefProjection flat = math::makeReliefProjection(equator);

        // cos(0) is one, so a degree east and a degree south must be the same number of cells.
        const core::i32 cols = flat.cellX(deg(1.0));
        const core::i32 rows = flat.cellZ(deg(0.0));
        check("at the equator a degree is square", cols == rows);
        check("and it is the real size of a degree", rows == static_cast<core::i32>(111320 / 30));
    }

    std::printf("-- a sphere is not a torus, and walking off the edge proves which\n");
    {
        math::GlobeWrap globe{};
        globe.columns = 1000;
        globe.rows = 400;

        auto wrapped = [&](core::i32 x, core::i32 z) {
            math::wrapCell(globe, x, z);
            return math::GlobeWrap{x, z};
        };

        check("a cell already on the world is untouched",
              wrapped(500, 200).columns == 500 && wrapped(500, 200).rows == 200);

        // East-west is the easy half, and must be a FLOORED modulo: C++ leaves a negative dividend
        // negative, so a body one cell west of the prime meridian would come back at -1.
        check("one cell west of zero is the far east", wrapped(-1, 200).columns == 999);
        check("and the column past the last is the first", wrapped(1000, 200).columns == 0);
        check("two laps still land somewhere real", wrapped(2500, 200).columns == 500);

        // @warning THE assertion this whole struct exists for. A torus would send a body walking north
        // out of the arctic straight into the antarctic. A sphere sends it back SOUTH along the
        // OPPOSITE meridian -- so the row reflects and the column turns by half the globe.
        const math::GlobeWrap overPole = wrapped(100, -1);
        check("crossing the north pole comes back south", overPole.rows == 0);
        check("on the opposite meridian", overPole.columns == 600);
        check("and NOT at the other pole", overPole.rows != globe.rows - 1);

        const math::GlobeWrap overSouth = wrapped(100, 400);
        check("crossing the south pole comes back north", overSouth.rows == 399);
        check("also on the opposite meridian", overSouth.columns == 600);

        // Further past the pole is further back down the far side, not further off the map.
        check("two rows past the pole is one row in", wrapped(100, -2).rows == 1);

        // Idempotent: wrapping an already-wrapped cell is a no-op, or a body that crossed twice
        // would drift a half-turn every time anything asked where it was.
        core::i32 x = -37;
        core::i32 z = -5;
        math::wrapCell(globe, x, z);
        const core::i32 onceX = x;
        const core::i32 onceZ = z;
        math::wrapCell(globe, x, z);
        check("wrapping is idempotent", x == onceX && z == onceZ);

        // And an unset wrap is the world as it was, which is what every existing world relies on.
        core::i32 openX = -37;
        core::i32 openZ = -5;
        math::wrapCell(math::GlobeWrap{}, openX, openZ);
        check("an open world is left alone", openX == -37 && openZ == -5);
    }

    std::printf("-- Magellan: the short way round, which a subtraction does not give\n");
    {
        math::GlobeWrap globe{};
        globe.columns = 1000;
        globe.rows = 400;
        core::i32 dx = 0;
        core::i32 dz = 0;

        // @warning THE case. Two places either side of the antimeridian are NEIGHBOURS. A plain
        // subtraction makes them 990 apart, so anything picking "the nearest place" picks the wrong
        // one and anything walking a heading sets off the long way round the planet -- which looks
        // exactly like a body on a very long journey rather than like a defect.
        math::shortestDelta(globe, 995, 100, 5, 100, dx, dz);
        check("across the antimeridian is ten cells east", dx == 10 && dz == 0);
        math::shortestDelta(globe, 5, 100, 995, 100, dx, dz);
        check("and ten cells west the other way", dx == -10);

        // Inside the sheet nothing changes.
        math::shortestDelta(globe, 100, 10, 300, 40, dx, dz);
        check("a short hop is just the difference", dx == 200 && dz == 30);

        // Just under and just over half a circumference: the answer must flip direction, because
        // that is the definition of the shorter way.
        math::shortestDelta(globe, 0, 0, 499, 0, dx, dz);
        check("just under half way is eastward", dx == 499);
        math::shortestDelta(globe, 0, 0, 501, 0, dx, dz);
        check("just over half way is westward", dx == -499);

        // A body that has already run laps must not be reported a lap away from where it stands.
        math::shortestDelta(globe, 2005, 100, 5, 100, dx, dz);
        check("laps do not accumulate into the answer", dx == 0);

        // @warning North-south is NOT shortcut through a pole, and the header says why: a pole
        // crossing reverses the direction of travel and translates the longitude by half the
        // globe, so it cannot be expressed as a vector a caller could step along.
        math::shortestDelta(globe, 100, 5, 100, 395, dx, dz);
        check("the poles are not a shortcut", dz == 390);

        // And an open world is a plain subtraction, which every existing caller relies on.
        math::shortestDelta(math::GlobeWrap{}, 995, 100, 5, 100, dx, dz);
        check("an open world just subtracts", dx == -990);
    }

    std::printf("-- and the ground is continuous across the seam, not merely close\n");
    {
        procgen::ChunkParams params{};
        params.size = 32u;
        params.worldSeed = 77u;
        params.noise.amplitude = 25.0f;
        params.noise.frequency = 0.03f;
        params.noise.octaves = 4u;
        params.globe.columns = 512;
        params.globe.rows = 256;

        // @warning The seam has to be IDENTICAL, not similar. Wrapping the coordinate before anything
        // reads it means the column past the last one IS the first one -- same cell, same noise
        // lattice -- so the ground matches by construction. A design that wrapped only the survey
        // and left the noise alone would leave a cliff of invented terrain along the antimeridian.
        bool seamless = true;
        for (core::i32 z = 0; z < 256; z += 16)
            if (procgen::sampleWorldHeight(params, 512, z).raw() != procgen::sampleWorldHeight(params, 0, z).raw())
                seamless = false;
        check("the antimeridian has no seam at all", seamless);

        bool westSeamless = true;
        for (core::i32 z = 0; z < 256; z += 16)
            if (procgen::sampleWorldHeight(params, -1, z).raw() != procgen::sampleWorldHeight(params, 511, z).raw())
                westSeamless = false;
        check("and neither does its western side", westSeamless);

        // Crossing the pole must reach the ground half a world away, not the ground next door.
        check("over the pole is the far meridian", procgen::sampleWorldHeight(params, 10, -1).raw() ==
                                                       procgen::sampleWorldHeight(params, 10 + 256, 0).raw());

        // And an open world is untouched by any of this.
        procgen::ChunkParams open = params;
        open.globe = math::GlobeWrap{};
        check("an open world still has edges",
              procgen::sampleWorldHeight(open, 512, 32).raw() != procgen::sampleWorldHeight(open, 0, 32).raw());
    }

    std::printf("-- tiles meet exactly, and only the outside of the survey fades\n");
    {
        // Four tiles in a 2x2 block, each 32 cells square, every sample encoding its WORLD position
        // so a tile picked from the wrong place shows up as the wrong ground rather than as noise.
        constexpr core::u32 kSide = 32u;
        static core::i16 quad[4][kSide * kSide];
        static math::ReliefField fields[4];
        math::ReliefMosaic mosaic{};

        for (core::u32 t = 0u; t < 4u; ++t)
        {
            const core::i32 baseX = static_cast<core::i32>((t % 2u) * kSide);
            const core::i32 baseZ = static_cast<core::i32>((t / 2u) * kSide);
            for (core::u32 r = 0u; r < kSide; ++r)
                for (core::u32 c = 0u; c < kSide; ++c)
                    quad[t][r * kSide + c] = static_cast<core::i16>((baseZ + static_cast<core::i32>(r)) * 100 + baseX +
                                                                    static_cast<core::i32>(c));
            fields[t].samples = quad[t];
            fields[t].width = kSide;
            fields[t].height = kSide;
            fields[t].originCellX = baseX;
            fields[t].originCellZ = baseZ;
            fields[t].projection = projection;
            fields[t].blendCells = 8u;
            // Only the OUTSIDE of the 2x2 block is exposed. West column faces nothing on its west,
            // and so on; the inner edges face a resident neighbour.
            core::u32 edges = 0u;
            if ((t % 2u) == 0u)
                edges |= math::kReliefEdgeWest;
            else
                edges |= math::kReliefEdgeEast;
            if ((t / 2u) == 0u)
                edges |= math::kReliefEdgeNorth;
            else
                edges |= math::kReliefEdgeSouth;
            fields[t].exposedEdges = edges;
            check("the tile joins the mosaic", mosaic.add(&fields[t]));
        }

        // Each tile answers for its own ground.
        math::Fixed32 h{};
        check("the north-west tile answers",
              mosaic.heightAt(1, 1, h) && h.raw() == projection.worldHeightOf(101).raw());
        check("and the south-east one answers its own",
              mosaic.heightAt(40, 40, h) && h.raw() == projection.worldHeightOf(4040).raw());
        check("outside the block nothing is resident", !mosaic.heightAt(-1, 0, h));

        // @warning **THE claim of this whole lot.** An inner edge faces a resident neighbour, so the
        // weight must stay at ONE right across it. A tile that faded on every side would ring
        // itself with half-invented ground, and a tiled world would come out cross-hatched with a
        // gentle valley at every boundary -- each of which reads as terrain, not as an error.
        // @warning The sampled band stays clear of the block's OUTER edges, which fade legitimately.
        // A first version walked z from 4, four cells inside the exposed northern band, and read
        // that correct fade as a failure of the seam -- the test was wrong, not the code.
        bool innerSolid = true;
        core::i32 seamProbed = 0;
        for (core::i32 z = 12; z <= 51; ++z)
        {
            for (core::i32 x = 24; x <= 39; ++x)
            {
                ++seamProbed;
                if (mosaic.weightAt(x, z).raw() != math::Fixed32::one().raw())
                    innerSolid = false;
            }
        }
        check("the vertical seam between tiles does not fade", innerSolid);
        check("and it was actually walked", seamProbed > 400);

        // The horizontal seam too, kept clear of the exposed east and west sides.
        bool horizontalSolid = true;
        for (core::i32 x = 12; x <= 51; ++x)
            for (core::i32 z = 24; z <= 39; ++z)
                if (mosaic.weightAt(x, z).raw() != math::Fixed32::one().raw())
                    horizontalSolid = false;
        check("the horizontal seam does not fade either", horizontalSolid);

        // And the ground is continuous across it: the last column of one tile and the first of the
        // next are adjacent world cells, so their samples must differ by exactly one.
        math::Fixed32 west{};
        math::Fixed32 east{};
        check("both sides of the seam answer", mosaic.heightAt(31, 10, west) && mosaic.heightAt(32, 10, east));
        check("and they are one cell apart in ground",
              east.raw() - west.raw() == projection.worldHeightOf(1).raw() - projection.worldHeightOf(0).raw());

        // The OUTSIDE still fades, or the survey would end in a cliff.
        check("the western outside fades", mosaic.weightAt(0, 30).raw() < math::Fixed32::one().raw());
        check("the eastern outside fades", mosaic.weightAt(63, 30).raw() < math::Fixed32::one().raw());
        check("the northern outside fades", mosaic.weightAt(30, 0).raw() < math::Fixed32::one().raw());
        check("the southern outside fades", mosaic.weightAt(30, 63).raw() < math::Fixed32::one().raw());

        // A tile with NO exposed edge is entirely real ground, which is what an interior tile of a
        // large survey is.
        math::ReliefField interior = fields[0];
        interior.exposedEdges = 0u;
        check("a surrounded tile never fades", interior.weightAt(0, 0).raw() == math::Fixed32::one().raw());

        // The mosaic is bounded, and hitting the bound is a budget rather than an error.
        math::ReliefMosaic full{};
        core::u32 added = 0u;
        while (full.add(&fields[0]))
            ++added;
        check("the mosaic is bounded", added == math::kMaxResidentReliefTiles);
    }

    std::printf("-- the pyramid: detail where it is looked at, coarse ground everywhere else\n");
    {
        math::ReliefResidencyParams res{};
        res.tileCells = 1024u;
        res.fineRadiusTiles = 1u;
        res.levels = 4u;
        res.radiusPerLevel = 1u;

        math::ReliefTileRequest plan[64];
        const core::u32 n = math::planReliefResidency(res, 5000, 7000, plan, 64u);
        check("a plan is produced", n > 0u);
        check("and it stacks every level", plan[n - 1u].level == res.levels - 1u);

        // Nine tiles a level at radius one, four levels.
        check("nine tiles a level", n == 36u);

        // @warning Nearest FIRST, so a budget that truncates loses the farthest tile rather than an
        // arbitrary one. A machine with less memory must lose ground at the horizon, never under
        // its own feet.
        check("the eye's own tile comes first", plan[0].level == 0u && plan[0].tileX == 4 && plan[0].tileZ == 6);
        math::ReliefTileRequest tight[5];
        const core::u32 few = math::planReliefResidency(res, 5000, 7000, tight, 5u);
        check("a tight budget fills exactly", few == 5u);
        check("and keeps the nearest tiles", tight[0].tileX == plan[0].tileX && tight[0].tileZ == plan[0].tileZ);
        bool nearestKept = true;
        for (core::u32 i = 0u; i < few; ++i)
            if (tight[i].level != plan[i].level || tight[i].tileX != plan[i].tileX)
                nearestKept = false;
        check("truncation drops the far ones, never the near ones", nearestKept);

        // A coarse tile covers four times the ground per step, so its lattice is coarser.
        check("level one covers twice the span per axis", math::reliefLevelScale(1u) == 2u);
        check("level three, eight times", math::reliefLevelScale(3u) == 8u);
        bool coarserLattice = true;
        for (core::u32 i = 0u; i < n; ++i)
            if (plan[i].level == 3u && plan[i].tileX != 0 && plan[i].tileX != 1 && plan[i].tileX != -1)
                coarserLattice = false;
        check("so a coarse tile index is a smaller number", coarserLattice);

        // Determinism: the same eye gives the same plan, in the same order.
        math::ReliefTileRequest again[64];
        const core::u32 m = math::planReliefResidency(res, 5000, 7000, again, 64u);
        bool identical = m == n;
        for (core::u32 i = 0u; i < n && identical; ++i)
            identical =
                again[i].level == plan[i].level && again[i].tileX == plan[i].tileX && again[i].tileZ == plan[i].tileZ;
        check("the plan is deterministic", identical);

        // @warning Floored, or the two tiles either side of the origin would both be tile zero and the
        // plan would ask for one tile where the eye needs two.
        math::ReliefTileRequest west[8];
        const core::u32 w = math::planReliefResidency(res, -1, -1, west, 8u);
        check("a cell west of the origin is in tile -1", w > 0u && west[0].tileX == -1 && west[0].tileZ == -1);

        // No tile is asked for twice: budget spent on ground already covered is budget lost.
        bool unique = true;
        for (core::u32 i = 0u; i < n; ++i)
            for (core::u32 j = i + 1u; j < n; ++j)
                if (plan[i].level == plan[j].level && plan[i].tileX == plan[j].tileX && plan[i].tileZ == plan[j].tileZ)
                    unique = false;
        check("and no tile is requested twice", unique);
    }

    std::printf("-- overlapping levels: the finest tile wins, whatever the arrival order\n");
    {
        // The same ground covered twice: a fine tile and a coarse one.
        constexpr core::u32 kSide = 16u;
        static core::i16 fine[kSide * kSide];
        static core::i16 coarse[kSide * kSide];
        for (core::u32 i = 0u; i < kSide * kSide; ++i)
        {
            fine[i] = 500;
            coarse[i] = 100;
        }

        math::ReliefField fineTile{};
        fineTile.samples = fine;
        fineTile.width = kSide;
        fineTile.height = kSide;
        fineTile.projection = projection;
        fineTile.blendCells = 0u;
        fineTile.level = 0u;

        math::ReliefField coarseTile = fineTile;
        coarseTile.samples = coarse;
        coarseTile.level = 2u;

        math::Fixed32 h{};
        math::ReliefMosaic coarseFirst{};
        check("coarse then fine", coarseFirst.add(&coarseTile) && coarseFirst.add(&fineTile));
        check("the fine tile answers", coarseFirst.heightAt(4, 4, h) && h.raw() == projection.worldHeightOf(500).raw());

        // @warning THE reason a level is stored. Without it the tile that answers is whichever was
        // streamed in first, so which ground a world has would depend on loader timing -- and two
        // targets would disagree the moment their loaders raced. Same answer, opposite order.
        math::ReliefMosaic fineFirst{};
        check("fine then coarse", fineFirst.add(&fineTile) && fineFirst.add(&coarseTile));
        check("still the fine tile", fineFirst.heightAt(4, 4, h) && h.raw() == projection.worldHeightOf(500).raw());

        // And where only the coarse tile reaches, coarse ground is what there is -- which is why
        // levels overlap: an evicted fine tile leaves ground behind, not a hole.
        math::ReliefField farCoarse = coarseTile;
        farCoarse.originCellX = 100;
        math::ReliefMosaic gapped{};
        check("a lone coarse tile is resident", gapped.add(&farCoarse));
        check("and it answers where nothing finer is",
              gapped.heightAt(104, 4, h) && h.raw() == projection.worldHeightOf(100).raw());
    }

    std::printf("-- a level boundary is a slope, not a ledge\n");
    {
        // A fine tile sitting on top of a coarse one covering the same ground, with the coarse
        // surface deliberately OFFSET: that offset is exactly what a level boundary steps by in a
        // real survey, because a coarse sample is the mean of the fine ones it replaces.
        constexpr core::u32 kFine = 48u;
        static core::i16 fineSamples[kFine * kFine];
        for (core::u32 i = 0u; i < kFine * kFine; ++i)
            fineSamples[i] = 800;

        math::ReliefField fine{};
        fine.samples = fineSamples;
        fine.width = kFine;
        fine.height = kFine;
        fine.projection = projection;
        fine.blendCells = 8u;
        fine.exposedEdges = 0xFu;
        fine.level = 0u;

        // The coarse tile is WIDER, so it still covers the ground where the fine one has faded out,
        // and sits 300 m below -- which is the step a level boundary shows in a real survey, because
        // a coarse sample is the mean of the fine ones it replaces.
        static core::i16 wideSamples[(kFine * 3u) * (kFine * 3u)];
        for (core::u32 i = 0u; i < (kFine * 3u) * (kFine * 3u); ++i)
            wideSamples[i] = 500;
        math::ReliefField coarse{};
        coarse.samples = wideSamples;
        coarse.width = kFine * 3u;
        coarse.height = kFine * 3u;
        coarse.originCellX = -static_cast<core::i32>(kFine);
        coarse.originCellZ = -static_cast<core::i32>(kFine);
        coarse.projection = projection;
        coarse.blendCells = 8u;
        coarse.exposedEdges = 0xFu;
        coarse.level = 2u;

        math::ReliefMosaic mosaic{};
        check("both levels are resident", mosaic.add(&fine) && mosaic.add(&coarse));

        // Deep inside the fine tile the fine ground wins outright.
        math::Fixed32 middle{};
        check("the interior is the fine survey",
              mosaic.heightAt(24, 24, middle) && middle.raw() == projection.worldHeightOf(800).raw());
        // Well outside it, the coarse survey answers.
        math::Fixed32 outside{};
        check("beyond it the coarse survey answers",
              mosaic.heightAt(-20, 24, outside) && outside.raw() == projection.worldHeightOf(500).raw());

        // @warning **THE measurement.** Walking across the boundary, the largest jump between two
        // adjacent cells must be a fraction of the 300 m the two surfaces differ by. Without the
        // blend the transition is that whole difference in a single cell -- a ledge running along
        // the edge of every fine tile, which reads as terrain rather than as a defect.
        core::i32 worstStep = 0;
        math::Fixed32 previous{};
        bool havePrevious = false;
        for (core::i32 x = -12; x <= 24; ++x)
        {
            math::Fixed32 here{};
            if (!mosaic.heightAt(x, 24, here))
            {
                havePrevious = false;
                continue;
            }
            if (havePrevious)
            {
                const core::i32 step =
                    here.raw() > previous.raw() ? here.raw() - previous.raw() : previous.raw() - here.raw();
                if (step > worstStep)
                    worstStep = step;
            }
            previous = here;
            havePrevious = true;
        }
        const core::i32 fullStep = projection.worldHeightOf(800).raw() - projection.worldHeightOf(500).raw();
        check("the handover is spread, not a single ledge", worstStep * 4 < fullStep);
        std::printf("     worst adjacent step %d raw against a %d raw difference between levels\n", worstStep,
                    fullStep);

        // @warning And the survey must still count as MEASURED across the boundary. Taking the finest
        // tile's own weight would fade real ground into invented ground in the middle of a survey,
        // wherever a fine tile happened to end -- the same ledge, one function later.
        check("a cell in the fine tile's band is still fully measured",
              mosaic.weightAt(2, 24).raw() == math::Fixed32::one().raw());
    }

    std::printf("-- the inverse map round-trips, which is what lets a baker resample\n");
    {
        // @warning The resampler walks cells and asks where each one is. If the inverse disagreed with
        // the forward map by even one cell, every sample would land one cell off -- a survey shifted
        // thirty metres, which is a coastline in slightly the wrong place and nothing that looks
        // like an error.
        bool roundTrips = true;
        for (core::i32 cell = -400; cell <= 400; cell += 7)
        {
            if (projection.cellX(projection.longitudeRawOf(cell)) != cell)
                roundTrips = false;
            if (projection.cellZ(projection.latitudeRawOf(cell)) != cell)
                roundTrips = false;
        }
        check("every cell maps back to itself", roundTrips);

        // The EDGE, not the centre: the value returned must be the smallest coordinate that still
        // falls in the cell, so one step back is the previous cell.
        check("it is the west edge of the column", projection.cellX(projection.longitudeRawOf(50) - 1) == 49);
        check("and the north edge of the row", projection.cellZ(projection.latitudeRawOf(50) - 1) == 50);

        // Rows run southward, so a larger row is a smaller latitude.
        check("a larger row is further south", projection.latitudeRawOf(100) < projection.latitudeRawOf(0));
        check("a larger column is further east", projection.longitudeRawOf(100) > projection.longitudeRawOf(0));
    }

    std::printf("-- a field reads its own cells, and refuses the ones nobody measured\n");
    {
        // A tiny field whose every sample says where it is, so a transposed index shows up as a
        // wrong PLACE rather than as noise.
        constexpr core::u32 kSide = 16u;
        static core::i16 samples[kSide * kSide];
        for (core::u32 row = 0u; row < kSide; ++row)
            for (core::u32 col = 0u; col < kSide; ++col)
                samples[row * kSide + col] = static_cast<core::i16>(row * 100 + col);
        samples[5u * kSide + 5u] = math::kReliefNoSample;

        math::ReliefField field{};
        field.samples = samples;
        field.width = kSide;
        field.height = kSide;
        field.originCellX = 1000;
        field.originCellZ = 2000;
        field.projection = projection;
        field.blendCells = 0u;

        math::Fixed32 height{};
        check("a cell inside the field answers", field.heightAt(1000, 2000, height));
        check("and it is the sea level plus its elevation", height.raw() == spec.seaLevelUnits.raw());

        check("rows run north to south",
              field.heightAt(1000, 2001, height) && height.raw() == projection.worldHeightOf(100).raw());
        check("columns run west to east",
              field.heightAt(1001, 2000, height) && height.raw() == projection.worldHeightOf(1).raw());

        // @warning A gap must be REFUSED, not answered. -32768 metres is a plausible-looking number
        // that would drag a whole block to the bottom of the sea, in a place a reader would have to
        // visit to doubt.
        check("a gap in the survey is refused", !field.heightAt(1005, 2005, height));

        check("west of the field is outside", !field.heightAt(999, 2000, height));
        check("east of the field is outside", !field.heightAt(1000 + kSide, 2000, height));
        check("and an empty field answers nothing", !math::ReliefField{}.heightAt(0, 0, height));
    }

    std::printf("-- the border fades, because the edge of a survey is not a cliff\n");
    {
        constexpr core::u32 kSide = 64u;
        static core::i16 flat[kSide * kSide] = {};

        math::ReliefField field{};
        field.samples = flat;
        field.width = kSide;
        field.height = kSide;
        field.projection = projection;
        field.blendCells = 8u;

        check("the middle is entirely real", field.weightAt(32, 32).raw() == math::Fixed32::one().raw());
        check("the very edge is nothing", field.weightAt(0, 32).raw() == 0);
        check("outside is nothing", field.weightAt(-1, 32).raw() == 0);

        // @warning A cell is near the CLOSEST edge, not near one nominated axis. The discriminating
        // case is a cell deep along one axis and close along the other: driving the ramp from the
        // column alone leaves a cell one row from the northern edge at full strength, so the survey
        // keeps a full-strength strip along two of its four sides.
        //
        // @warning The first version of this check compared (2,2) against (2,32), where BOTH distances
        // are two -- so it read the same number either way and passed against an implementation
        // that ignored three edges out of four. A probe caught it; re-reading it did not.
        check("a cell close in z fades even when it is deep in x",
              field.weightAt(30, 1).raw() < math::Fixed32::one().raw());
        check("a cell close in x fades even when it is deep in z",
              field.weightAt(1, 30).raw() < math::Fixed32::one().raw());
        check("and the two axes fade alike", field.weightAt(30, 1).raw() == field.weightAt(1, 30).raw());
        check("the southern edge fades like the northern",
              field.weightAt(32, static_cast<core::i32>(kSide) - 2).raw() == field.weightAt(32, 1).raw());
        check("the eastern edge fades like the western",
              field.weightAt(static_cast<core::i32>(kSide) - 2, 32).raw() == field.weightAt(1, 32).raw());

        // Monotone rather than a threshold: a fixed expected value would be a number chosen so
        // today's band width passes.
        bool rises = true;
        for (core::u32 d = 1u; d < 8u; ++d)
            if (field.weightAt(static_cast<core::i32>(d), 32).raw() <=
                field.weightAt(static_cast<core::i32>(d) - 1, 32).raw())
                rises = false;
        check("and it climbs steadily across the band", rises);
        check("reaching full strength at the band's width", field.weightAt(8, 32).raw() == math::Fixed32::one().raw());

        // A hard border is legal, and means what it says.
        field.blendCells = 0u;
        check("a zero band is a hard border", field.weightAt(0, 32).raw() == math::Fixed32::one().raw());
    }

    std::printf("-- real ground displaces the lowest frequency, and costs nothing when absent\n");
    {
        procgen::ChunkParams params{};
        params.size = 32u;
        params.worldSeed = 4242u;
        params.noise.amplitude = 20.0f;
        params.noise.frequency = 0.01f;
        params.noise.octaves = 4u;

        // @warning THE invariant that keeps every gate already folded from moving: a world with no
        // survey behind it must take exactly the path it always took. Asserted rather than argued,
        // because "I only added a branch" is how a signature moves.
        const math::Fixed32 withoutField = procgen::sampleWorldHeight(params, 12, 34);

        constexpr core::u32 kSide = 8u;
        static core::i16 samples[kSide * kSide];
        for (core::u32 i = 0u; i < kSide * kSide; ++i)
            samples[i] = 500;

        math::ReliefField field{};
        field.samples = samples;
        field.width = kSide;
        field.height = kSide;
        field.originCellX = 100;
        field.originCellZ = 100;
        field.projection = projection;
        field.blendCells = 0u;

        // A mosaic of one: the shape every world uses, small or global.
        math::ReliefMosaic lone{};
        check("the lone tile joins a mosaic of one", lone.add(&field));
        params.relief = &lone;
        check("a cell outside the field is untouched by it",
              procgen::sampleWorldHeight(params, 12, 34).raw() == withoutField.raw());

        // Inside, with no detail layer asked for, the ground is exactly what was measured.
        const math::Fixed32 measured = procgen::sampleWorldHeight(params, 104, 104);
        check("a cell inside the field is the real ground", measured.raw() == projection.worldHeightOf(500).raw());

        // @warning The detail layer is declared SEPARATELY rather than as a scale on the main noise,
        // and this is why: scaling a five-octave field down keeps its low octaves too, which reads
        // as real terrain that happens to be smooth. Here the roughness must move the sample
        // without moving what it is roughness ON.
        params.reliefDetail.amplitude = 2.0f;
        params.reliefDetail.frequency = 0.4f;
        params.reliefDetail.octaves = 2u;
        const math::Fixed32 rough = procgen::sampleWorldHeight(params, 104, 104);
        check("the detail layer roughens it", rough.raw() != measured.raw());
        const core::i32 added = rough.raw() - measured.raw();
        const core::i32 magnitude = added < 0 ? -added : added;
        check("but only by the detail layer's own amplitude", magnitude <= math::Fixed32::fromFloat(2.0f).raw());

        // And a gap inside the field falls back to invented ground at FULL strength, not to a fade
        // towards nothing: the survey has no opinion there, so the generator's is the only one.
        samples[0] = math::kReliefNoSample;
        params.reliefDetail.amplitude = 0.0f;
        const math::Fixed32 atGap = procgen::sampleWorldHeight(params, 100, 100);
        params.relief = nullptr;
        check("a gap falls back to invented ground", atGap.raw() == procgen::sampleWorldHeight(params, 100, 100).raw());
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
