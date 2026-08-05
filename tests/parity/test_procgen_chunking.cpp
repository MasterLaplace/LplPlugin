/**
 * @file test_procgen_chunking.cpp
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
 * a radius drains through this cell — and the tests below are what make that a
 * claim rather than a hope: the same cell judged from two different chunks must
 * come out the same, or the world visibly seams every time a river crosses a
 * border.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Chunking.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/Erosion.hpp>

#include <cstdio>

namespace {

using namespace lpl;

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

/// The canonical parameters, from the one constexpr the kernel smoke also reads.
[[nodiscard]] procgen::ChunkParams makeParams() { return procgen::parityChunkParams(); }

void testHeightSeamsAreExact()
{
    std::printf("neighbouring chunks share their edge exactly\n");

    const procgen::ChunkParams params = makeParams();
    core::u32 mismatches = 0u;
    for (core::i32 z = -2; z <= 2; ++z)
        for (core::i32 x = -2; x <= 2; ++x)
        {
            mismatches += procgen::countSeamMismatches(params, {x, z}, {x + 1, z});
            mismatches += procgen::countSeamMismatches(params, {x, z}, {x, z + 1});
        }
    std::printf("    %u mismatched cells over 50 seams\n", mismatches);
    check(mismatches == 0u, "no height differs across any seam");
}

void testRiversAgreeAcrossSeams()
{
    std::printf("and so do the rivers, which are DERIVED rather than sampled\n");

    const procgen::ChunkParams params = makeParams();
    procgen::EndlessRiverParams rivers;

    // The real invariant, and the one that took two wrong tests to state.
    //
    // Comparing the last column of one chunk with the first of the next compares
    // ADJACENT cells, not the same cell twice — and a river is perfectly entitled
    // to end between two columns, so that test can only fail by accident.
    //
    // What must hold is that the batched per-chunk pass is a rendering of a
    // function of POSITION ALONE. isRiverCoarseCell computes each verdict from a
    // window centred on the cell itself; markChunkRivers computes a chunk's worth
    // in one pass. If they agree everywhere, no cell's answer depends on which
    // chunk asked, and the seams follow from that rather than being spot-checked.
    core::u32 disagreements = 0u;
    core::u32 riverCells = 0u;
    core::u32 coarseChecked = 0u;
    const core::u32 coarsePerChunk = params.size / rivers.coarseCells;

    for (core::i32 cz = -1; cz <= 1; ++cz)
        for (core::i32 cx = -1; cx <= 1; ++cx)
        {
            const procgen::Grid<core::u8> mask = procgen::markChunkRivers(params, rivers, {cx, cz});
            for (core::u32 z = 0u; z < coarsePerChunk; ++z)
                for (core::u32 x = 0u; x < coarsePerChunk; ++x)
                {
                    const core::i32 coarseX = cx * static_cast<core::i32>(coarsePerChunk) + static_cast<core::i32>(x);
                    const core::i32 coarseZ = cz * static_cast<core::i32>(coarsePerChunk) + static_cast<core::i32>(z);
                    const bool reference = procgen::isRiverCoarseCell(params, rivers, coarseX, coarseZ);
                    const bool batched = mask.at(x * rivers.coarseCells, z * rivers.coarseCells) != 0u;
                    if (reference != batched)
                        ++disagreements;
                    ++coarseChecked;
                }
            for (core::u32 i = 0u; i < mask.cellCount(); ++i)
                riverCells += mask[i] != 0u ? 1u : 0u;
        }

    std::printf("    %u disagreements over %u coarse cells, %u river cells in nine chunks\n", disagreements,
                coarseChecked, riverCells);
    check(disagreements == 0u, "the batched pass answers exactly what the position-only reference does");
    check(riverCells != 0u, "and there are rivers at all (otherwise the check above is free)");
}

void testRiversFollowTheTerrain()
{
    std::printf("water runs downhill, and not on the peaks\n");

    const procgen::ChunkParams params = makeParams();
    procgen::EndlessRiverParams rivers;

    // Nine chunks, not one. A single chunk may hold no water at all, and the
    // first version of this test then printed "nothing to compare" and passed —
    // a check that cannot fail, which is the one kind this project keeps
    // catching itself writing.
    math::Fixed32 riverSum = math::Fixed32::zero();
    math::Fixed32 landSum = math::Fixed32::zero();
    core::u32 river = 0u;
    core::u32 land = 0u;
    for (core::i32 cz = -1; cz <= 1; ++cz)
        for (core::i32 cx = -1; cx <= 1; ++cx)
        {
            const procgen::Grid<core::u8> mask = procgen::markChunkRivers(params, rivers, {cx, cz});
            const procgen::Heightfield height = procgen::generateChunkTerrain(params, {cx, cz});
            for (core::u32 i = 0u; i < mask.cellCount(); ++i)
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

    const core::f32 riverMean = (riverSum / math::Fixed32::fromInt(static_cast<core::i32>(river))).toFloat();
    const core::f32 landMean = (landSum / math::Fixed32::fromInt(static_cast<core::i32>(land))).toFloat();
    std::printf("    %u river cells average %.2f, %u others average %.2f\n", river, riverMean, land, landMean);
    // Not a tautology: the rule counts upstream cells and never looks at altitude
    // except to exclude the sea. Water ending up low is the terrain agreeing with
    // the routing, which is the whole point.
    check(riverMean < landMean, "rivers sit lower than the land around them");
}

void testDeterminism()
{
    std::printf("the same chunk is the same river, every time\n");

    const procgen::ChunkParams params = makeParams();
    procgen::EndlessRiverParams rivers;

    // A chunk KNOWN to carry water: comparing two empty masks proves only that
    // nothing equals nothing.
    procgen::ChunkCoord wet{0, 0};
    core::u32 wetCells = 0u;
    for (core::i32 cz = -1; cz <= 1 && wetCells == 0u; ++cz)
        for (core::i32 cx = -1; cx <= 1 && wetCells == 0u; ++cx)
        {
            const procgen::Grid<core::u8> probe = procgen::markChunkRivers(params, rivers, {cx, cz});
            core::u32 count = 0u;
            for (core::u32 i = 0u; i < probe.cellCount(); ++i)
                count += probe[i] != 0u ? 1u : 0u;
            if (count != 0u)
            {
                wet = {cx, cz};
                wetCells = count;
            }
        }
    check(wetCells != 0u, "a chunk with water was found to test against");

    const procgen::Grid<core::u8> a = procgen::markChunkRivers(params, rivers, wet);
    const procgen::Grid<core::u8> b = procgen::markChunkRivers(params, rivers, wet);
    bool identical = a.cellCount() == b.cellCount();
    for (core::u32 i = 0u; identical && i < a.cellCount(); ++i)
        identical = a[i] == b[i];
    check(identical, "two runs give the same mask");

    // A wider basin can only add water: it is a superset of the same walks.
    procgen::EndlessRiverParams wider = rivers;
    wider.basinRadius = rivers.basinRadius + 3u;
    const procgen::Grid<core::u8> big = procgen::markChunkRivers(params, wider, wet);
    core::u32 narrow = 0u;
    core::u32 broad = 0u;
    for (core::u32 i = 0u; i < a.cellCount(); ++i)
    {
        narrow += a[i] != 0u ? 1u : 0u;
        broad += big[i] != 0u ? 1u : 0u;
    }
    std::printf("    radius %u marks %u cells, radius %u marks %u\n", rivers.basinRadius, narrow, wider.basinRadius,
                broad);
    check(broad >= narrow, "a wider basin never marks less water");
}

void testErodedSeamsAreExact()
{
    std::printf("erosion survives being cut into chunks, if the apron is wide enough\n");

    const procgen::ChunkParams params = makeParams();
    const core::u32 iterations = 6u;
    const core::u32 chunkSize = params.size;

    // The real invariant, and the third time this session that stating it took
    // two attempts: the first version compared the SAME call twice and measured
    // determinism, which was already tested and could not fail here.
    //
    // What must hold is that the chunked assembly equals the MONOLITHIC
    // computation. So: erode one 3x3-chunk field in a single pass, with the same
    // apron, and compare it cell by cell against the nine chunks assembled. An
    // apron too small shows up immediately, as a band of disagreement along every
    // internal border.
    // The reference gets a GENEROUS apron — four times the iteration count —
    // because a reference with the same margin as the thing it judges is not a
    // reference. Measured: at an apron of exactly N the chunks and this monolith
    // disagreed on 10 cells of 5184, and the fault was here, in the monolith's
    // own contaminated border. Against a wide reference, N is wrong by 10 cells
    // and N+1 is exact — which is the number the implementation uses.
    const core::u32 referenceApron = 4u * iterations;
    const core::u32 wideSide = chunkSize * 3u + 2u * referenceApron;
    const core::i32 wideOrigin = -static_cast<core::i32>(chunkSize) - static_cast<core::i32>(referenceApron);

    procgen::Heightfield monolith{wideSide, wideSide, math::Fixed32::zero()};
    for (core::u32 z = 0u; z < wideSide; ++z)
        for (core::u32 x = 0u; x < wideSide; ++x)
            monolith.at(x, z) = procgen::sampleWorldHeight(params, wideOrigin + static_cast<core::i32>(x),
                                                           wideOrigin + static_cast<core::i32>(z));

    procgen::ThermalErosionParams thermal;
    thermal.iterations = iterations;
    thermal.talus = 0.6f;
    (void) procgen::thermalErode(monolith, thermal);

    core::u32 mismatches = 0u;
    core::u32 compared = 0u;
    for (core::i32 cz = -1; cz <= 1; ++cz)
        for (core::i32 cx = -1; cx <= 1; ++cx)
        {
            const procgen::Heightfield chunk = procgen::generateErodedChunkTerrain(params, {cx, cz}, iterations);
            for (core::u32 z = 0u; z < chunkSize; ++z)
                for (core::u32 x = 0u; x < chunkSize; ++x)
                {
                    const core::i32 worldX = cx * static_cast<core::i32>(chunkSize) + static_cast<core::i32>(x);
                    const core::i32 worldZ = cz * static_cast<core::i32>(chunkSize) + static_cast<core::i32>(z);
                    const core::u32 wx = static_cast<core::u32>(worldX - wideOrigin);
                    const core::u32 wz = static_cast<core::u32>(worldZ - wideOrigin);
                    if (chunk.at(x, z).raw() != monolith.at(wx, wz).raw())
                        ++mismatches;
                    ++compared;
                }
        }

    std::printf("    %u cells differ from the monolithic erosion, over %u compared\n", mismatches, compared);
    check(mismatches == 0u, "the chunked assembly IS the unchunked computation");

    // And erosion actually did something: two identical fields would pass the
    // check above without a grain of material having moved.
    const procgen::Heightfield raw = procgen::generateChunkTerrain(params, {0, 0});
    const procgen::Heightfield eroded = procgen::generateErodedChunkTerrain(params, {0, 0}, iterations);
    core::u32 changed = 0u;
    for (core::u32 i = 0u; i < raw.cellCount(); ++i)
        changed += raw[i].raw() != eroded[i].raw() ? 1u : 0u;
    std::printf("    erosion moved %u of %u cells\n", changed, raw.cellCount());
    check(changed != 0u, "erosion changed the terrain (otherwise the check above is free)");
}

void testTrunksCrossChunks()
{
    std::printf("a trunk river is longer than the chunk it starts in\n");

    const procgen::ChunkParams params = makeParams();
    procgen::EndlessRiverParams rivers;

    // Search wide. An earlier measurement scanned eighty-one macro cells around
    // the origin, found nothing, and concluded the world had no continental
    // drainage; six hundred and twenty-five cells later there was plenty. A test
    // that samples one corner and generalises is the same error with a green tick
    // on it.
    core::u32 trunkMacro = 0u;
    procgen::ChunkCoord firstTrunk{0, 0};
    for (core::i32 mz = -12; mz <= 12; ++mz)
        for (core::i32 mx = -12; mx <= 12; ++mx)
            if (procgen::isTrunkMacroCell(params, rivers, mx, mz))
            {
                if (trunkMacro == 0u)
                {
                    // Macro cell to chunk: macro span in coarse cells times coarse
                    // cells per fine cell, over the chunk size.
                    const core::i32 fineX = mx * static_cast<core::i32>(rivers.trunkCells * rivers.coarseCells);
                    const core::i32 fineZ = mz * static_cast<core::i32>(rivers.trunkCells * rivers.coarseCells);
                    firstTrunk = {fineX / static_cast<core::i32>(params.size),
                                  fineZ / static_cast<core::i32>(params.size)};
                }
                ++trunkMacro;
            }

    std::printf("    %u macro cells carry a trunk over 625 searched\n", trunkMacro);
    check(trunkMacro != 0u, "the world has continental drainage somewhere");
    if (trunkMacro == 0u)
        return;

    // Continuity: the chunk holding a trunk and its four neighbours should not all
    // be dry. A trunk that lives inside a single chunk is a puddle with ambition.
    core::u32 wetNeighbours = 0u;
    const procgen::ChunkCoord around[5] = {
        firstTrunk,
        {firstTrunk.x + 1, firstTrunk.z    },
        {firstTrunk.x - 1, firstTrunk.z    },
        {firstTrunk.x,     firstTrunk.z + 1},
        {firstTrunk.x,     firstTrunk.z - 1}
    };
    for (const procgen::ChunkCoord &coord : around)
    {
        const procgen::Grid<core::u8> mask = procgen::markChunkRivers(params, rivers, coord);
        core::u32 cells = 0u;
        for (core::u32 i = 0u; i < mask.cellCount(); ++i)
            cells += mask[i] != 0u ? 1u : 0u;
        if (cells != 0u)
            ++wetNeighbours;
    }
    std::printf("    %u of 5 chunks around the first trunk carry water\n", wetNeighbours);
    check(wetNeighbours >= 2u, "the trunk continues past the chunk it was found in");
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
 */
void testTheEndlessWorldComesFromTheRecipe()
{
    std::printf("the endless world is the recipe's world, scaled up to walk through\n");

    procgen::WorldRecipe recipe;
    recipe.seed = 20260804u;
    recipe.terrain.seed = 20260804u;
    recipe.terrain.amplitude = 14.0f;
    recipe.terrain.frequency = 0.09f;
    recipe.terrain.octaves = 5u;
    recipe.terrain.kind = procgen::NoiseKind::Ridged;
    recipe.biomes.seaLevel = -1.0f;
    recipe.biomes.beachHeight = 0.9f;

    const procgen::EndlessPlan plan = procgen::endlessPlanFromRecipe(recipe, 24u);

    // The structure is the recipe's, so editing a document changes the world you walk
    // through. That was true only by coincidence while the two were written apart.
    check(plan.chunk.noise.seed == recipe.seed, "the plan carries the recipe's seed");
    check(plan.chunk.noise.kind == procgen::NoiseKind::Ridged, "and its noise kind");
    check(plan.chunk.noise.octaves == 5u, "and its octave structure");
    check(plan.chunk.size == 24u, "and the chunk size it was asked for");

    // The one that matters. Where the sea is has to be ONE answer: a second one
    // classifies a cell as land, draws water over it and refuses to let anything walk
    // on it, all at once, with nothing looking wrong anywhere.
    check(plan.rule.seaLevel == recipe.biomes.seaLevel, "the sea it floods is the sea the recipe classifies");
    check(plan.rule.beachBand == recipe.biomes.beachHeight, "and the shore it draws is the shore it classifies");

    // The scaling is not the identity, and that is its whole point: a map read from
    // above and a place walked through at eye height are not the same terrain.
    check(plan.chunk.noise.amplitude > recipe.terrain.amplitude, "a walked world has taller relief");
    check(plan.chunk.noise.frequency < recipe.terrain.frequency, "and fewer, wider landforms");

    // Monotone rather than a threshold: a fixed number here would be a number chosen
    // so that today's tuning passes, which is this repository's most-repeated mistake.
    core::f32 previousSpan = -1.0f;
    for (core::f32 relief : {1.0f, 2.0f, 4.0f})
    {
        procgen::WalkScale scale;
        scale.reliefScale = relief;
        const procgen::EndlessPlan scaled = procgen::endlessPlanFromRecipe(recipe, 24u);
        const procgen::EndlessPlan bigger = procgen::endlessPlanFromRecipe(recipe, 24u, scale);
        (void) scaled;

        const procgen::Heightfield field = procgen::generateChunkTerrain(bigger.chunk, {0, 0});
        math::Fixed32 low{};
        math::Fixed32 high{};
        (void) procgen::heightRange(field, low, high);
        const core::f32 span = high.toFloat() - low.toFloat();
        check(span >= previousSpan, "raising the relief scale never flattens the world");
        previousSpan = span;
    }
    std::printf("    tallest span at relief x4: %.2f m\n", static_cast<double>(previousSpan));
}

int main()
{
    std::printf("== procgen chunking: a world with no edges ==\n");
    testHeightSeamsAreExact();
    testRiversAgreeAcrossSeams();
    testRiversFollowTheTerrain();
    testErodedSeamsAreExact();
    testTrunksCrossChunks();
    testTheEndlessWorldComesFromTheRecipe();
    testDeterminism();

    // ── The signatures the kernel must reproduce ────────────────────────────
    {
        const procgen::ChunkParams params = makeParams();
        const procgen::EndlessFoldResult folded =
            procgen::foldEndlessPatch(params, procgen::parityRiverParams(), procgen::kParityPatchRadius);
        std::printf("\n-- signatures the kernel must reproduce --\n");
        std::printf("  height_sig = 0x%08X\n", folded.heightSignature);
        std::printf("  river_sig  = 0x%08X\n", folded.riverSignature);
        std::printf("  chunks     = %u\n", folded.chunks);
        std::printf("  river      = %u\n", folded.riverCells);
        std::printf("  seams      = %u\n", folded.seamMismatches);
    }

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
