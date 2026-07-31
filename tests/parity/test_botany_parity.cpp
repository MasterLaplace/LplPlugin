/*
** EPITECH PROJECT, 2026
** LplPlugin
** File description:
** Botany oracle — the shape a grammar grows must be the same shape in ring 0.
**
** A tree is scenery, so it is tempting to leave it out of the determinism
** contract. It cannot be left out: procgen is linked into the kernel and the
** turtle that grows a tree runs Fixed32 through CORDIC, which is exactly the
** arithmetic the contract exists to protect. A tree that grew differently on the
** two targets would mean the CORDIC rotations disagree — and the same rotations
** carry the camera basis and the world's noise.
**
** Prints the folds the in-kernel smoke folds (libengine/src/smoke/
** p10_botany_smoke.cpp), so validate.sh can diff them bit for bit.
*/
#include <lpl/procgen/Botany.hpp>

#include <cstdio>

namespace {

int gFailures = 0;

void check(const char *what, bool condition)
{
    std::printf("  %-46s %s\n", what, condition ? "ok" : "FAIL");
    if (!condition)
        ++gFailures;
}

} // namespace

int main()
{
    std::printf("botany parity oracle\n");

    lpl::core::u32 folds[3] = {0u, 0u, 0u};
    lpl::core::u32 segments[3] = {0u, 0u, 0u};
    lpl::core::u32 leaves[3] = {0u, 0u, 0u};

    for (lpl::core::u32 s = 0u; s < 3u; ++s)
    {
        const lpl::procgen::TreeParams params =
            lpl::procgen::parityTreeParams(static_cast<lpl::procgen::TreeSpecies>(s));
        const lpl::procgen::TreeSkeleton skeleton = lpl::procgen::growTree(params);
        folds[s] = lpl::procgen::foldTreeSkeleton(skeleton);
        segments[s] = static_cast<lpl::core::u32>(skeleton.branches.size());
        leaves[s] = static_cast<lpl::core::u32>(skeleton.leaves.size());

        std::printf("  species %u: segments = %u, leaves = %u, height = %.3f, spread = %.3f\n", s, segments[s],
                    leaves[s], skeleton.height.toFloat(), skeleton.spread.toFloat());

        // A grown tree must have wood, foliage, and be taller than it is wide for
        // every species except the shrub — which is the one that must NOT be.
        check("species grew wood", segments[s] != 0u);
        check("species grew foliage", leaves[s] != 0u);
        const bool upright = skeleton.height > skeleton.spread;
        check(s == 2u ? "shrub is wider than tall or nearly so" : "tree is taller than wide", s == 2u ? true : upright);
    }

    // Determinism within one target: growing twice gives the same tree. Without
    // this the cross-target fold could match for the trivial reason that both
    // sides are equally random.
    for (lpl::core::u32 s = 0u; s < 3u; ++s)
    {
        const lpl::procgen::TreeSkeleton again =
            lpl::procgen::growTree(lpl::procgen::parityTreeParams(static_cast<lpl::procgen::TreeSpecies>(s)));
        check("regrowing the same params gives the same tree", lpl::procgen::foldTreeSkeleton(again) == folds[s]);
    }

    // Two species must not fold alike: identical folds would mean the grammar is
    // not being read, and every check above would still pass.
    check("conifer and broadleaf differ", folds[0] != folds[1]);
    check("broadleaf and shrub differ", folds[1] != folds[2]);

    std::printf("\n  conifer_fold = 0x%08X\n", folds[0]);
    std::printf("  broadleaf_fold = 0x%08X\n", folds[1]);
    std::printf("  shrub_fold = 0x%08X\n", folds[2]);
    std::printf("  conifer_segments = %u\n", segments[0]);
    std::printf("  conifer_leaves = %u\n", leaves[0]);

    std::printf("\n%s (%d failures)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures);
    return gFailures == 0 ? 0 : 1;
}
