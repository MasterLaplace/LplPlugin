/**
 * @file test_relief_parity.cpp
 * @brief Gate P21 oracle: a world standing on measured ground.
 *
 * @warning The numbers printed here are what the kernel must fold, bit for bit. What is asserted
 * here is everything a stable signature cannot say by itself -- that the survey was actually read,
 * that the border band actually blended, and that the world is different from the one that ignores
 * it. A gate whose signatures agree because neither side ever applied the relief agrees perfectly
 * and proves nothing.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/ReliefParity.hpp>

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

} // namespace

int main()
{
    using namespace lpl;

    const engine::ReliefFoldResult measured = engine::foldReliefParity();
    const engine::ReliefFoldResult invented = engine::foldInventedParity();

    std::printf("-- the survey is read, and the counts say so\n");
    {
        // @warning Without these the gate is satisfied by a field that never applies: every signature
        // is perfectly stable on both targets whether the relief was consulted or not.
        check("measured ground was stood on", measured.measuredCells > 0u);
        check("invented ground was reached beyond it", measured.inventedCells > 0u);
        check("and the border band was crossed", measured.blendedCells > 0u);

        // The coastline is INSIDE the survey, so both signs of elevation are exercised. A world
        // entirely above water never tests the reconciliation the projection exists to make.
        check("the coast is inside the world", measured.seaCells > 0u &&
                                                   measured.seaCells < measured.measuredCells +
                                                                           measured.inventedCells +
                                                                           measured.blendedCells);
    }

    std::printf("-- the control: a world that ignores the survey is a different world\n");
    {
        // @warning THE discriminating comparison. Each of these would still be stable across targets
        // if `sampleWorldHeight` dropped the relief branch entirely.
        check("the ground differs", measured.heightSignature != invented.heightSignature);
        check("the coastline differs", measured.coastSignature != invented.coastSignature);
        check("and the body goes somewhere else", measured.walkSignature != invented.walkSignature);

        // The survey itself is the same in both runs -- it is folded before anything reads it -- so
        // this signature must NOT move. If it does, the two runs are not the same world with one
        // difference, and nothing above compares what it claims to.
        check("but the survey itself is one survey", measured.sampleSignature == invented.sampleSignature);
    }

    std::printf("-- the body actually walked, and downhill\n");
    {
        check("it took steps", measured.walkSteps > 0u);
        // @warning A walk that stops at once folds perfectly and proves nothing about the ground it
        // did not cross.
        check("more than a handful", measured.walkSteps >= 8u);
        check("and it ended lower than it started", measured.descended > 0);
    }

    std::printf("\n== signatures (must match the kernel fold) ==\n");
    std::printf("  relief_sample=0x%08X relief_height=0x%08X relief_walk=0x%08X relief_coast=0x%08X\n",
                measured.sampleSignature, measured.heightSignature, measured.walkSignature,
                measured.coastSignature);
    std::printf("  relief_measured=%u relief_invented=%u relief_blended=%u relief_sea=%u\n",
                measured.measuredCells, measured.inventedCells, measured.blendedCells, measured.seaCells);
    std::printf("  relief_steps=%u relief_descended=%d\n", measured.walkSteps, measured.descended);
    std::printf("  relief_plainheight=0x%08X relief_plainwalk=0x%08X\n", invented.heightSignature,
                invented.walkSignature);

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
