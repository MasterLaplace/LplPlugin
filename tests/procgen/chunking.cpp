#include <lpl/procgen/Chunking.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(chunking);

/**
 * @brief Gate P9 endless: a square patch of the endless world, chunk by chunk, folds the same ground
 *        and rivers on both targets, and its chunks agree at every seam.
 */
LPL_TEST(endless_patch_folds_the_same)
{
    const lpl::procgen::EndlessFoldResult folded = lpl::procgen::foldEndlessPatch(
        lpl::procgen::parityChunkParams(), lpl::procgen::parityRiverParams(), lpl::procgen::kParityPatchRadius);

    test.check(folded.chunks > 0u, "the patch is folded chunk by chunk");
    test.check(folded.riverCells > 0u, "rivers run through it");
    test.check(folded.seamMismatches == 0u, "and neighbouring chunks agree on every cell of their seams");

    test.measureHexadecimal("height_signature", folded.heightSignature);
    test.measureHexadecimal("river_signature", folded.riverSignature);
    test.measure("chunks", folded.chunks);
    test.measure("river_cells", folded.riverCells);
}
