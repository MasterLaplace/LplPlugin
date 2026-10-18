#include <lpl/procgen/Chunking.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/Erosion.hpp>
#include <lpl/procgen/Landmark.hpp>

#include <cstdio>

namespace {

constexpr lpl::core::i32 kSpan = 6;

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    if (!condition)
    {
        ++gFailures;
        std::printf("  FAIL %s\n", what);
    }
}

/**
 * @brief The canonical parameters, from the one constexpr the engine test `chunking` also reads.
 */
[[nodiscard]] lpl::procgen::ChunkParams makeParams() { return lpl::procgen::parityChunkParams(); }

void testHeightSeamsAreExact()
{
    std::printf("neighbouring chunks share their edge exactly\n");

    const lpl::procgen::ChunkParams params = makeParams();
    lpl::core::u32 mismatches = 0u;
    for (lpl::core::i32 z = -2; z <= 2; ++z)
        for (lpl::core::i32 x = -2; x <= 2; ++x)
        {
            mismatches += lpl::procgen::countSeamMismatches(params, {x, z}, {x + 1, z});
            mismatches += lpl::procgen::countSeamMismatches(params, {x, z}, {x, z + 1});
        }
    std::printf("    %u mismatched cells over 50 seams\n", mismatches);
    check(mismatches == 0u, "no height differs across any seam");
}

/**
 * @brief And so do the rivers, which are DERIVED rather than sampled.
 *
 * The real invariant, and the one that took two wrong tests to state.
 *
 * Comparing the last column of one chunk with the first of the next compares
 * ADJACENT cells, not the same cell twice — and a river is perfectly entitled
 * to end between two columns, so that test can only fail by accident.
 *
 * What must hold is that the batched per-chunk pass is a rendering of a
 * function of POSITION ALONE. isRiverCoarseCell computes each verdict from a
 * window centred on the cell itself; markChunkRivers computes a chunk's worth
 * in one pass. If they agree everywhere, no cell's answer depends on which
 * chunk asked, and the seams follow from that rather than being spot-checked.
 */
void testRiversAgreeAcrossSeams()
{
    std::printf("and so do the rivers, which are DERIVED rather than sampled\n");

    const lpl::procgen::ChunkParams params = makeParams();
    lpl::procgen::EndlessRiverParams rivers;

    lpl::core::u32 disagreements = 0u;
    lpl::core::u32 riverCells = 0u;
    lpl::core::u32 coarseChecked = 0u;
    const lpl::core::u32 coarsePerChunk = params.size / rivers.coarseCells;

    for (lpl::core::i32 cz = -1; cz <= 1; ++cz)
        for (lpl::core::i32 cx = -1; cx <= 1; ++cx)
        {
            const lpl::procgen::Grid<lpl::core::u8> mask = lpl::procgen::markChunkRivers(params, rivers, {cx, cz});
            for (lpl::core::u32 z = 0u; z < coarsePerChunk; ++z)
                for (lpl::core::u32 x = 0u; x < coarsePerChunk; ++x)
                {
                    const lpl::core::i32 coarseX =
                        cx * static_cast<lpl::core::i32>(coarsePerChunk) + static_cast<lpl::core::i32>(x);
                    const lpl::core::i32 coarseZ =
                        cz * static_cast<lpl::core::i32>(coarsePerChunk) + static_cast<lpl::core::i32>(z);
                    const bool reference = lpl::procgen::isRiverCoarseCell(params, rivers, coarseX, coarseZ);
                    const bool batched = mask.at(x * rivers.coarseCells, z * rivers.coarseCells) != 0u;
                    if (reference != batched)
                        ++disagreements;
                    ++coarseChecked;
                }
            for (lpl::core::u32 i = 0u; i < mask.cellCount(); ++i)
                riverCells += mask[i] != 0u ? 1u : 0u;
        }

    std::printf("    %u disagreements over %u coarse cells, %u river cells in nine chunks\n", disagreements,
                coarseChecked, riverCells);
    check(disagreements == 0u, "the batched pass answers exactly what the position-only reference does");
    check(riverCells != 0u, "and there are rivers at all (otherwise the check above is free)");
}

/**
 * @brief Water runs downhill, and not on the peaks.
 *
 * Nine chunks, not one. A single chunk may hold no water at all, and the
 * first version of this test then printed "nothing to compare" and passed —
 * a check that cannot fail, which is the one kind this project keeps
 * catching itself writing.
 *
 * Not a tautology: the rule counts upstream cells and never looks at altitude
 * except to exclude the sea. Water ending up low is the terrain agreeing with
 * the routing, which is the whole point.
 */
void testRiversFollowTheTerrain()
{
    std::printf("water runs downhill, and not on the peaks\n");

    const lpl::procgen::ChunkParams params = makeParams();
    lpl::procgen::EndlessRiverParams rivers;

    lpl::math::Fixed32 riverSum = lpl::math::Fixed32::zero();
    lpl::math::Fixed32 landSum = lpl::math::Fixed32::zero();
    lpl::core::u32 river = 0u;
    lpl::core::u32 land = 0u;
    for (lpl::core::i32 cz = -1; cz <= 1; ++cz)
        for (lpl::core::i32 cx = -1; cx <= 1; ++cx)
        {
            const lpl::procgen::Grid<lpl::core::u8> mask = lpl::procgen::markChunkRivers(params, rivers, {cx, cz});
            const lpl::procgen::Heightfield height = lpl::procgen::generateChunkTerrain(params, {cx, cz});
            for (lpl::core::u32 i = 0u; i < mask.cellCount(); ++i)
            {
                if (mask[i] != 0u)
                {
                    riverSum = riverSum + height[i];
                    ++river;
                }
                else
                {
                    landSum = landSum + height[i];
                    ++land;
                }
            }
        }

    check(river != 0u && land != 0u, "the sample holds both water and land");
    if (river == 0u || land == 0u)
        return;

    const lpl::core::f32 riverMean =
        (riverSum / lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(river))).toFloat();
    const lpl::core::f32 landMean =
        (landSum / lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(land))).toFloat();
    std::printf("    %u river cells average %.2f, %u others average %.2f\n", river, riverMean, land, landMean);
    check(riverMean < landMean, "rivers sit lower than the land around them");
}

/**
 * @brief The same chunk is the same river, every time.
 *
 * It is tested on a chunk KNOWN to carry water: comparing two empty masks proves
 * only that nothing equals nothing.
 *
 * A wider basin can only add water: it is a superset of the same walks.
 */
void testDeterminism()
{
    std::printf("the same chunk is the same river, every time\n");

    const lpl::procgen::ChunkParams params = makeParams();
    lpl::procgen::EndlessRiverParams rivers;

    lpl::procgen::ChunkCoord wet{0, 0};
    lpl::core::u32 wetCells = 0u;
    for (lpl::core::i32 cz = -1; cz <= 1 && wetCells == 0u; ++cz)
        for (lpl::core::i32 cx = -1; cx <= 1 && wetCells == 0u; ++cx)
        {
            const lpl::procgen::Grid<lpl::core::u8> probe = lpl::procgen::markChunkRivers(params, rivers, {cx, cz});
            lpl::core::u32 count = 0u;
            for (lpl::core::u32 i = 0u; i < probe.cellCount(); ++i)
                count += probe[i] != 0u ? 1u : 0u;
            if (count != 0u)
            {
                wet = {cx, cz};
                wetCells = count;
            }
        }
    check(wetCells != 0u, "a chunk with water was found to test against");

    const lpl::procgen::Grid<lpl::core::u8> a = lpl::procgen::markChunkRivers(params, rivers, wet);
    const lpl::procgen::Grid<lpl::core::u8> b = lpl::procgen::markChunkRivers(params, rivers, wet);
    bool identical = a.cellCount() == b.cellCount();
    for (lpl::core::u32 i = 0u; identical && i < a.cellCount(); ++i)
        identical = a[i] == b[i];
    check(identical, "two runs give the same mask");

    lpl::procgen::EndlessRiverParams wider = rivers;
    wider.basinRadius = rivers.basinRadius + 3u;
    const lpl::procgen::Grid<lpl::core::u8> big = lpl::procgen::markChunkRivers(params, wider, wet);
    lpl::core::u32 narrow = 0u;
    lpl::core::u32 broad = 0u;
    for (lpl::core::u32 i = 0u; i < a.cellCount(); ++i)
    {
        narrow += a[i] != 0u ? 1u : 0u;
        broad += big[i] != 0u ? 1u : 0u;
    }
    std::printf("    radius %u marks %u cells, radius %u marks %u\n", rivers.basinRadius, narrow, wider.basinRadius,
                broad);
    check(broad >= narrow, "a wider basin never marks less water");
}

/**
 * @brief Erosion survives being cut into chunks, if the apron is wide enough.
 *
 * The real invariant, and the third time this session that stating it took
 * two attempts: the first version compared the SAME call twice and measured
 * determinism, which was already tested and could not fail here.
 *
 * What must hold is that the chunked assembly equals the MONOLITHIC
 * computation. So: erode one 3x3-chunk field in a single pass, with the same
 * apron, and compare it cell by cell against the nine chunks assembled. An
 * apron too small shows up immediately, as a band of disagreement along every
 * internal border.
 *
 * The reference gets a GENEROUS apron — four times the iteration count —
 * because a reference with the same margin as the thing it judges is not a
 * reference. Measured: at an apron of exactly N the chunks and this monolith
 * disagreed on 10 cells of 5184, and the fault was here, in the monolith's
 * own contaminated border. Against a wide reference, N is wrong by 10 cells
 * and N+1 is exact — which is the number the implementation uses.
 *
 * And erosion actually did something: two identical fields would pass the
 * comparison with the monolith without a grain of material having moved.
 */
void testErodedSeamsAreExact()
{
    std::printf("erosion survives being cut into chunks, if the apron is wide enough\n");

    const lpl::procgen::ChunkParams params = makeParams();
    const lpl::core::u32 iterations = 6u;
    const lpl::core::u32 chunkSize = params.size;

    const lpl::core::u32 referenceApron = 4u * iterations;
    const lpl::core::u32 wideSide = chunkSize * 3u + 2u * referenceApron;
    const lpl::core::i32 wideOrigin =
        -static_cast<lpl::core::i32>(chunkSize) - static_cast<lpl::core::i32>(referenceApron);

    lpl::procgen::Heightfield monolith{wideSide, wideSide, lpl::math::Fixed32::zero()};
    for (lpl::core::u32 z = 0u; z < wideSide; ++z)
        for (lpl::core::u32 x = 0u; x < wideSide; ++x)
            monolith.at(x, z) = lpl::procgen::sampleWorldHeight(params, wideOrigin + static_cast<lpl::core::i32>(x),
                                                                wideOrigin + static_cast<lpl::core::i32>(z));

    lpl::procgen::ThermalErosionParams thermal;
    thermal.iterations = iterations;
    thermal.talus = 0.6f;
    (void) lpl::procgen::thermalErode(monolith, thermal);

    lpl::core::u32 mismatches = 0u;
    lpl::core::u32 compared = 0u;
    for (lpl::core::i32 cz = -1; cz <= 1; ++cz)
        for (lpl::core::i32 cx = -1; cx <= 1; ++cx)
        {
            const lpl::procgen::Heightfield chunk =
                lpl::procgen::generateErodedChunkTerrain(params, {cx, cz}, iterations);
            for (lpl::core::u32 z = 0u; z < chunkSize; ++z)
                for (lpl::core::u32 x = 0u; x < chunkSize; ++x)
                {
                    const lpl::core::i32 worldX =
                        cx * static_cast<lpl::core::i32>(chunkSize) + static_cast<lpl::core::i32>(x);
                    const lpl::core::i32 worldZ =
                        cz * static_cast<lpl::core::i32>(chunkSize) + static_cast<lpl::core::i32>(z);
                    const lpl::core::u32 wx = static_cast<lpl::core::u32>(worldX - wideOrigin);
                    const lpl::core::u32 wz = static_cast<lpl::core::u32>(worldZ - wideOrigin);
                    if (chunk.at(x, z).raw() != monolith.at(wx, wz).raw())
                        ++mismatches;
                    ++compared;
                }
        }

    std::printf("    %u cells differ from the monolithic erosion, over %u compared\n", mismatches, compared);
    check(mismatches == 0u, "the chunked assembly IS the unchunked computation");

    const lpl::procgen::Heightfield raw = lpl::procgen::generateChunkTerrain(params, {0, 0});
    const lpl::procgen::Heightfield eroded = lpl::procgen::generateErodedChunkTerrain(params, {0, 0}, iterations);
    lpl::core::u32 changed = 0u;
    for (lpl::core::u32 i = 0u; i < raw.cellCount(); ++i)
        changed += raw[i].raw() != eroded[i].raw() ? 1u : 0u;
    std::printf("    erosion moved %u of %u cells\n", changed, raw.cellCount());
    check(changed != 0u, "erosion changed the terrain (otherwise the check above is free)");
}

/**
 * @brief A trunk river is longer than the chunk it starts in.
 *
 * Search wide. An earlier measurement scanned eighty-one macro cells around
 * the origin, found nothing, and concluded the world had no continental
 * drainage; six hundred and twenty-five cells later there was plenty. A test
 * that samples one corner and generalises is the same error with a green tick
 * on it.
 *
 * A macro cell maps to a chunk through the macro span in coarse cells times
 * the coarse cells per fine cell, over the chunk size.
 *
 * Continuity: the chunk holding a trunk and its four neighbours should not all
 * be dry. A trunk that lives inside a single chunk is a puddle with ambition.
 */
void testTrunksCrossChunks()
{
    std::printf("a trunk river is longer than the chunk it starts in\n");

    const lpl::procgen::ChunkParams params = makeParams();
    lpl::procgen::EndlessRiverParams rivers;

    lpl::core::u32 trunkMacro = 0u;
    lpl::procgen::ChunkCoord firstTrunk{0, 0};
    for (lpl::core::i32 mz = -12; mz <= 12; ++mz)
        for (lpl::core::i32 mx = -12; mx <= 12; ++mx)
            if (lpl::procgen::isTrunkMacroCell(params, rivers, mx, mz))
            {
                if (trunkMacro == 0u)
                {
                    const lpl::core::i32 fineX =
                        mx * static_cast<lpl::core::i32>(rivers.trunkCells * rivers.coarseCells);
                    const lpl::core::i32 fineZ =
                        mz * static_cast<lpl::core::i32>(rivers.trunkCells * rivers.coarseCells);
                    firstTrunk = {fineX / static_cast<lpl::core::i32>(params.size),
                                  fineZ / static_cast<lpl::core::i32>(params.size)};
                }
                ++trunkMacro;
            }

    std::printf("    %u macro cells carry a trunk over 625 searched\n", trunkMacro);
    check(trunkMacro != 0u, "the world has continental drainage somewhere");
    if (trunkMacro == 0u)
        return;

    lpl::core::u32 wetNeighbours = 0u;
    const lpl::procgen::ChunkCoord around[5] = {
        firstTrunk,
        {firstTrunk.x + 1, firstTrunk.z    },
        {firstTrunk.x - 1, firstTrunk.z    },
        {firstTrunk.x,     firstTrunk.z + 1},
        {firstTrunk.x,     firstTrunk.z - 1}
    };
    for (const lpl::procgen::ChunkCoord &coord : around)
    {
        const lpl::procgen::Grid<lpl::core::u8> mask = lpl::procgen::markChunkRivers(params, rivers, coord);
        lpl::core::u32 cells = 0u;
        for (lpl::core::u32 i = 0u; i < mask.cellCount(); ++i)
            cells += mask[i] != 0u ? 1u : 0u;
        if (cells != 0u)
            ++wetNeighbours;
    }
    std::printf("    %u of 5 chunks around the first trunk carry water\n", wetNeighbours);
    check(wetNeighbours >= 2u, "the trunk continues past the chunk it was found in");
}

/**
 * @brief Density, over a patch big enough to mean something.
 *
 * A landmark cell that placed a site in EVERY chunk would be a lattice, and one that
 * placed one every few chunks is a world. The upper bound is what catches a runaway
 * rule; the lower is what catches a rule that never fires.
 */
void checkLandmarkDensity(const lpl::procgen::EndlessPlan &plan)
{
    lpl::core::u32 caves = 0u;
    lpl::core::u32 villages = 0u;
    lpl::core::u32 buildings = 0u;
    for (lpl::core::i32 cz = -kSpan; cz <= kSpan; ++cz)
        for (lpl::core::i32 cx = -kSpan; cx <= kSpan; ++cx)
        {
            const lpl::procgen::ChunkCoord coord{cx, cz};
            lpl::core::u32 ownedHere = 0u;
            lpl::procgen::forEachLandmarkNear(plan.chunk, plan.rule.caveMouths, lpl::procgen::LandmarkKind::CaveMouth,
                                              plan.rule.seaLevel, coord, [&](const lpl::procgen::LandmarkSite &site) {
                                                  if (lpl::procgen::chunkOwnsLandmark(plan.chunk, site, coord))
                                                      ++ownedHere;
                                              });
            caves += ownedHere;

            lpl::procgen::forEachLandmarkNear(
                plan.chunk, plan.rule.villages, lpl::procgen::LandmarkKind::Settlement, plan.rule.seaLevel, coord,
                [&](const lpl::procgen::LandmarkSite &site) {
                    if (!lpl::procgen::chunkOwnsLandmark(plan.chunk, site, coord))
                        return;
                    ++villages;
                    const lpl::procgen::VillagePlan village = lpl::procgen::planVillage(plan.chunk, site);
                    lpl::procgen::forEachVillageBuilding(village,
                                                         [&](const lpl::procgen::LandmarkBuilding &) { ++buildings; });
                });
        }
    const lpl::core::u32 chunksScanned = static_cast<lpl::core::u32>((2 * kSpan + 1) * (2 * kSpan + 1));
    std::printf("    over %u chunks: %u cave mouths, %u villages, %u buildings\n", chunksScanned, caves, villages,
                buildings);
    check(caves > 0u, "the world has cave mouths in it");
    check(villages > 0u, "and villages");
    check(buildings > 0u, "and the villages have buildings in them");
    check(caves < chunksScanned * 4u, "cave mouths are not one per cell of the lattice");
}

/**
 * @brief A site in reach of two chunks is seen by both.
 *
 * The property the carve depends on. If only the owner saw it, its shelf would stop
 * dead at the chunk border and the ground either side would disagree by the whole
 * drop — the most visible seam a world can have. A site is in reach of the eastern
 * chunk when its footprint crosses into it.
 */
void checkASiteInReachOfTwoChunksIsSeenByBoth(const lpl::procgen::EndlessPlan &plan)
{
    lpl::core::u32 shared = 0u;
    lpl::core::u32 disagreements = 0u;
    for (lpl::core::i32 cz = -kSpan; cz <= kSpan; ++cz)
        for (lpl::core::i32 cx = -kSpan; cx <= kSpan; ++cx)
        {
            const lpl::procgen::ChunkCoord here{cx, cz};
            const lpl::procgen::ChunkCoord east{cx + 1, cz};
            lpl::procgen::forEachLandmarkNear(
                plan.chunk, plan.rule.caveMouths, lpl::procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, here,
                [&](const lpl::procgen::LandmarkSite &mine) {
                    const lpl::core::i32 border = (cx + 1) * 24;
                    if (mine.cellX + static_cast<lpl::core::i32>(mine.radius) < border)
                        return;
                    ++shared;
                    bool seen = false;
                    lpl::procgen::forEachLandmarkNear(
                        plan.chunk, plan.rule.caveMouths, lpl::procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel,
                        east, [&](const lpl::procgen::LandmarkSite &theirs) {
                            if (theirs.cellX == mine.cellX && theirs.cellZ == mine.cellZ &&
                                theirs.height == mine.height && theirs.seed == mine.seed &&
                                theirs.facing == mine.facing)
                                seen = true;
                        });
                    if (!seen)
                        ++disagreements;
                });
        }
    std::printf("    %u footprints cross a border, %u seen by only one side\n", shared, disagreements);
    check(shared > 0u, "some footprints do cross a chunk border");
    check(disagreements == 0u, "and both chunks see them identically");
}

/**
 * @brief Exactly one owner.
 *
 * Ownership decides who DRAWS. Two owners is a village rendered twice — z-fighting on
 * every wall — and none is a village whose ground was flattened for nothing.
 */
void checkEverySiteHasExactlyOneOwner(const lpl::procgen::EndlessPlan &plan)
{
    lpl::core::u32 doubleOwned = 0u;
    lpl::core::u32 unowned = 0u;
    for (lpl::core::i32 lz = -3; lz <= 3; ++lz)
        for (lpl::core::i32 lx = -3; lx <= 3; ++lx)
        {
            lpl::procgen::LandmarkSite site;
            if (!lpl::procgen::landmarkAt(plan.chunk, plan.rule.villages, lpl::procgen::LandmarkKind::Settlement,
                                          plan.rule.seaLevel, lx, lz, site))
                continue;
            lpl::core::u32 owners = 0u;
            for (lpl::core::i32 cz = -kSpan * 2; cz <= kSpan * 2; ++cz)
                for (lpl::core::i32 cx = -kSpan * 2; cx <= kSpan * 2; ++cx)
                    if (lpl::procgen::chunkOwnsLandmark(plan.chunk, site, {cx, cz}))
                        ++owners;
            if (owners != 1u)
                ++unowned;
            doubleOwned += owners > 1u ? 1u : 0u;
        }
    std::printf("    %u sites with anything other than one owner, %u with more than one\n", unowned, doubleOwned);
    check(unowned == 0u, "every site has exactly one owning chunk");
}

/**
 * @brief Landmarks: sited once, agreed on by every chunk that touches them.
 *
 * A landmark is the first thing in this world that OCCUPIES AN AREA. Terrain needs no
 * siting rule because a height is a function of a position; a village is either here or
 * it is not, and every chunk overlapping it has to reach the same verdict — including
 * the one holding only its far corner.
 *
 * Three things are measured rather than asserted from the code: how many sites there
 * actually are (a rule that places none passes every other check vacuously), that a
 * site in reach of two chunks is seen by BOTH, and that exactly one chunk claims to own
 * each of them.
 */
void testLandmarksAreAgreedOn()
{
    std::printf("landmarks are sited once and agreed on\n");

    const lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    const lpl::procgen::EndlessPlan plan = lpl::procgen::endlessPlanFromRecipe(recipe, 24u);

    check(plan.rule.carveCaveMouths, "the walked world asks for cave mouths");
    check(plan.rule.raiseVillages, "and for villages");

    checkLandmarkDensity(plan);
    checkASiteInReachOfTwoChunksIsSeenByBoth(plan);
    checkEverySiteHasExactlyOneOwner(plan);
}

} // namespace

/**
 * @brief The endless world is the recipe's world, scaled — not a second one.
 *
 * The chunk parameters and the content rule used to be written out by hand beside the
 * recipe they were supposed to agree with. Two descriptions of one world, free to
 * drift, with nothing to say when they had — and one of them did: where the sea is
 * was a constant in a sample while the classifier used the recipe's, so the same world
 * had two shorelines depending on which half of the code you asked.
 *
 * The structure is the recipe's, so editing a document changes the world you walk
 * through. That was true only by coincidence while the two were written apart.
 *
 * The sea level is the one that matters. Where the sea is has to be ONE answer: a
 * second one classifies a cell as land, draws water over it and refuses to let anything
 * walk on it, all at once, with nothing looking wrong anywhere.
 *
 * The scaling is not the identity, and that is its whole point: a map read from above
 * and a place walked through at eye height are not the same terrain. It is checked
 * monotone rather than against a threshold: a fixed number here would be a number
 * chosen so that today's tuning passes, which is this repository's most-repeated
 * mistake.
 */
void testTheEndlessWorldComesFromTheRecipe()
{
    std::printf("the endless world is the recipe's world, scaled up to walk through\n");

    lpl::procgen::WorldRecipe recipe;
    recipe.seed = 20260804u;
    recipe.terrain.seed = 20260804u;
    recipe.terrain.amplitude = 14.0f;
    recipe.terrain.frequency = 0.09f;
    recipe.terrain.octaves = 5u;
    recipe.terrain.kind = lpl::procgen::NoiseKind::Ridged;
    recipe.biomes.seaLevel = -1.0f;
    recipe.biomes.beachHeight = 0.9f;

    const lpl::procgen::EndlessPlan plan = lpl::procgen::endlessPlanFromRecipe(recipe, 24u);

    check(plan.chunk.noise.seed == recipe.seed, "the plan carries the recipe's seed");
    check(plan.chunk.noise.kind == lpl::procgen::NoiseKind::Ridged, "and its noise kind");
    check(plan.chunk.noise.octaves == 5u, "and its octave structure");
    check(plan.chunk.size == 24u, "and the chunk size it was asked for");

    check(plan.rule.seaLevel == recipe.biomes.seaLevel, "the sea it floods is the sea the recipe classifies");
    check(plan.rule.beachBand == recipe.biomes.beachHeight, "and the shore it draws is the shore it classifies");

    check(plan.chunk.noise.amplitude > recipe.terrain.amplitude, "a walked world has taller relief");
    check(plan.chunk.noise.frequency < recipe.terrain.frequency, "and fewer, wider landforms");

    lpl::core::f32 previousSpan = -1.0f;
    for (lpl::core::f32 relief : {1.0f, 2.0f, 4.0f})
    {
        lpl::procgen::WalkScale scale;
        scale.reliefScale = relief;
        const lpl::procgen::EndlessPlan scaled = lpl::procgen::endlessPlanFromRecipe(recipe, 24u);
        const lpl::procgen::EndlessPlan bigger = lpl::procgen::endlessPlanFromRecipe(recipe, 24u, scale);
        (void) scaled;

        const lpl::procgen::Heightfield field = lpl::procgen::generateChunkTerrain(bigger.chunk, {0, 0});
        lpl::math::Fixed32 low{};
        lpl::math::Fixed32 high{};
        (void) lpl::procgen::heightRange(field, low, high);
        const lpl::core::f32 span = high.toFloat() - low.toFloat();
        check(span >= previousSpan, "raising the relief scale never flattens the world");
        previousSpan = span;
    }
    std::printf("    tallest span at relief x4: %.2f m\n", static_cast<double>(previousSpan));
}

/**
 * @brief What the walked world is actually MADE of.
 *
 * An instrument before a change, not a check. Looking at the client, the endless world
 * reads as overwhelmingly blue, and there are two candidate explanations that call for
 * opposite fixes: too many rivers, or a sea level sitting in the middle of the noise
 * range. Guessing between them is how a threshold gets tuned to make today's screenshot
 * look right.
 *
 * So this measures the composition of the terrain a walker actually gets — the recipe run
 * through `endlessPlanFromRecipe` with the default `WalkScale`, which is what
 * `TerrainWorld::setInfinite` builds — and prints it. It asserts only what must be true
 * whatever the tuning: the shares add up, and water is not the whole world.
 *
 * The rivers are the plan's OWN river parameters, which is the point of them being in
 * the plan: a test that built its own set would be measuring a configuration nothing
 * runs.
 *
 * Water not being the whole world is the only claim worth asserting without a
 * measurement to back a number: a world that is ENTIRELY water is not a world you can
 * walk through, and that is what the client looked like.
 */
void testWhatTheWalkedWorldIsMadeOf()
{
    std::printf("what the walked world is made of\n");

    lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    const lpl::procgen::EndlessPlan plan = lpl::procgen::endlessPlanFromRecipe(recipe, 32u);

    const lpl::procgen::EndlessRiverParams &rivers = plan.rivers;

    lpl::core::u32 cells = 0u;
    lpl::core::u32 sea = 0u;
    lpl::core::u32 shore = 0u;
    lpl::core::u32 river = 0u;
    lpl::core::f32 lowest = 1.0e9f;
    lpl::core::f32 highest = -1.0e9f;

    for (lpl::core::i32 cz = -2; cz <= 2; ++cz)
        for (lpl::core::i32 cx = -2; cx <= 2; ++cx)
        {
            const lpl::procgen::ChunkTerrain terrain = lpl::procgen::generateChunkTerrain(
                plan.chunk, rivers, lpl::procgen::ChunkCoord{cx, cz}, plan.rule, [](lpl::core::i32, lpl::core::i32) {});
            for (lpl::core::u32 z = 0u; z < plan.chunk.size; ++z)
                for (lpl::core::u32 x = 0u; x < plan.chunk.size; ++x)
                {
                    const lpl::core::f32 h = terrain.height.at(x, z).toFloat();
                    ++cells;
                    lowest = h < lowest ? h : lowest;
                    highest = h > highest ? h : highest;
                    if (h < plan.rule.seaLevel)
                        ++sea;
                    else if (h < plan.rule.seaLevel + plan.rule.beachBand)
                        ++shore;
                    if (terrain.rivers.at(x, z) != 0u)
                        ++river;
                }
        }

    const auto share = [cells](lpl::core::u32 n) { return cells == 0u ? 0.0 : (100.0 * n) / cells; };
    std::printf("    relief [%.2f .. %.2f], sea level %.2f\n", static_cast<double>(lowest),
                static_cast<double>(highest), static_cast<double>(plan.rule.seaLevel));
    std::printf("    %u cells: %.1f%% under the sea, %.1f%% shore, %.1f%% river\n", cells, share(sea), share(shore),
                share(river));

    check(cells > 0u, "the walked world generates cells");
    check(sea + shore <= cells, "the shares are a partition of the cells");
    check(sea < cells, "the walked world is not entirely sea");
}

/**
 * @brief A world with no edges still has to agree with itself.
 *
 * The whole scheme rests on one property: a value at a world position does not
 * depend on which chunk asked for it. Height gets that for free — it is a noise
 * sample at absolute coordinates. A DERIVED layer does not, and rivers are the
 * interesting case, because drainage on a bounded map is a global computation:
 * fill every depression, then accumulate flow from every cell to the sea. There
 * is no "every cell" here.
 *
 * So the verdict is re-posed as a bounded question — enough of the terrain within
 * a radius drains through this cell — and the tests here are what make that a
 * claim rather than a hope: the same cell judged from two different chunks must
 * come out the same, or the world visibly seams every time a river crosses a
 * border.
 */
int main()
{
    std::printf("== procgen chunking: a world with no edges ==\n");
    testHeightSeamsAreExact();
    testRiversAgreeAcrossSeams();
    testRiversFollowTheTerrain();
    testErodedSeamsAreExact();
    testTrunksCrossChunks();
    testTheEndlessWorldComesFromTheRecipe();
    testWhatTheWalkedWorldIsMadeOf();
    testLandmarksAreAgreedOn();
    testDeterminism();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
