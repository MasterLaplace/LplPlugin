#include <lpl/procgen/Chunking.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(chunking);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Chunks either side of the origin gate P9 endless folds: the patch is three by three.
 */
constexpr lpl::core::u32 kPatchRadius = 1u;

/**
 * @struct EndlessFoldResult
 * @brief What gate P9 endless records of a patch of the endless world.
 *
 * @details The seam count travels with the signatures on purpose: a fold proves two machines
 *          agree, not that they agree on something correct, and a chunked world that seams
 *          identically on both targets would pass a signature check every time.
 */
struct EndlessFoldResult {
    lpl::core::u32 heightSignature{0u}; /**< FNV-1a over every cell of every chunk folded. */
    lpl::core::u32 riverSignature{0u};  /**< FNV-1a over the river masks. */
    lpl::core::u32 chunks{0u};          /**< Chunks visited. */
    lpl::core::u32 riverCells{0u};      /**< Cells carrying water. */
    lpl::core::u32 seamMismatches{0u};  /**< Height disagreements across the patch's seams. */
};

/**
 * @brief Folds one word into a running signature, byte by byte: raw Q16.16 words, never a decimal
 *        rendering, so the fold is an identity on the bits.
 */
void foldWord(lpl::core::u32 &hash, lpl::core::u32 word)
{
    for (lpl::core::u32 byte = 0u; byte < 4u; ++byte)
    {
        hash ^= (word >> (byte * 8u)) & 0xFFu;
        hash *= kFnv1aPrime;
    }
}

/**
 * @brief Folds one chunk of the patch, its ground and its rivers, and counts its water.
 */
void foldChunk(const lpl::procgen::ChunkParams &params, const lpl::procgen::EndlessRiverParams &rivers,
               lpl::procgen::ChunkCoord coord, EndlessFoldResult &result)
{
    const lpl::procgen::Heightfield height = lpl::procgen::generateChunkTerrain(params, coord);
    const lpl::procgen::Grid<lpl::core::u8> water = lpl::procgen::markChunkRivers(params, rivers, coord);

    for (lpl::core::u32 cell = 0u; cell < height.cellCount(); ++cell)
        foldWord(result.heightSignature, static_cast<lpl::core::u32>(height[cell].raw()));
    for (lpl::core::u32 cell = 0u; cell < water.cellCount(); ++cell)
    {
        foldWord(result.riverSignature, water[cell]);
        result.riverCells += water[cell] != 0u ? 1u : 0u;
    }
}

/**
 * @brief Folds a square patch of the endless world, chunk by chunk, and counts the cells its
 *        neighbouring chunks disagree on.
 *
 * @details The bounded world has been under the determinism contract since gate P7 and the running
 *          simulation since gate P8; this puts the endless one under it: same seed, same chunks,
 *          same bits, on Linux and in ring 0.
 *
 * @param params World parameters.
 * @param rivers How a river is decided.
 * @param radius Chunks either side of the origin; the patch is (2r+1) squared.
 */
[[nodiscard]] EndlessFoldResult foldEndlessPatch(const lpl::procgen::ChunkParams &params,
                                                 const lpl::procgen::EndlessRiverParams &rivers, lpl::core::u32 radius)
{
    EndlessFoldResult result{};
    const lpl::core::i32 reach = static_cast<lpl::core::i32>(radius);

    result.heightSignature = kFnv1aOffsetBasis;
    result.riverSignature = kFnv1aOffsetBasis;
    for (lpl::core::i32 chunkZ = -reach; chunkZ <= reach; ++chunkZ)
    {
        for (lpl::core::i32 chunkX = -reach; chunkX <= reach; ++chunkX)
        {
            const lpl::procgen::ChunkCoord coord{chunkX, chunkZ};

            foldChunk(params, rivers, coord, result);
            if (chunkX < reach)
                result.seamMismatches += lpl::procgen::countSeamMismatches(params, coord, {chunkX + 1, chunkZ});
            if (chunkZ < reach)
                result.seamMismatches += lpl::procgen::countSeamMismatches(params, coord, {chunkX, chunkZ + 1});
            ++result.chunks;
        }
    }
    return result;
}

} // namespace

/**
 * @brief Gate P9 endless: a square patch of the endless world, chunk by chunk, folds the same ground
 *        and rivers on both targets, and its chunks agree at every seam.
 */
LPL_TEST(endless_patch_folds_the_same)
{
    const EndlessFoldResult folded =
        foldEndlessPatch(lpl::procgen::parityChunkParams(), lpl::procgen::EndlessRiverParams{}, kPatchRadius);

    test.check(folded.chunks > 0u, "the patch is folded chunk by chunk");
    test.check(folded.riverCells > 0u, "rivers run through it");
    test.check(folded.seamMismatches == 0u, "and neighbouring chunks agree on every cell of their seams");

    test.measureHexadecimal("height_signature", folded.heightSignature);
    test.measureHexadecimal("river_signature", folded.riverSignature);
    test.measure("chunks", folded.chunks);
    test.measure("river_cells", folded.riverCells);
}
