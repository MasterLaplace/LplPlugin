#include <lpl/procgen/Botany.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(botany);

namespace {

[[nodiscard]] lpl::procgen::TreeSkeleton grow(lpl::procgen::TreeSpecies species)
{
    return lpl::procgen::growTree(lpl::procgen::parityTreeParams(species));
}

} // namespace

/**
 * @brief Gate P10 botany: a conifer, a broadleaf and a shrub grow into the same skeletons on both
 *        targets, each with wood and leaves, and no two alike.
 */
LPL_TEST(each_species_grows_the_same_tree)
{
    const lpl::procgen::TreeSkeleton conifer = grow(lpl::procgen::TreeSpecies::Conifer);
    const lpl::procgen::TreeSkeleton broadleaf = grow(lpl::procgen::TreeSpecies::Broadleaf);
    const lpl::procgen::TreeSkeleton shrub = grow(lpl::procgen::TreeSpecies::Shrub);
    const lpl::core::u32 coniferFold = lpl::procgen::foldTreeSkeleton(conifer);
    const lpl::core::u32 broadleafFold = lpl::procgen::foldTreeSkeleton(broadleaf);
    const lpl::core::u32 shrubFold = lpl::procgen::foldTreeSkeleton(shrub);

    test.check(!conifer.branches.empty() && !broadleaf.branches.empty() && !shrub.branches.empty(),
               "every species grows wood");
    test.check(!conifer.leaves.empty() && !broadleaf.leaves.empty() && !shrub.leaves.empty(), "and leaves");
    test.check(conifer.height > conifer.spread && broadleaf.height > broadleaf.spread,
               "the two trees are taller than they are wide");
    test.check(coniferFold != broadleafFold && broadleafFold != shrubFold, "no two species fold alike");
    test.check(lpl::procgen::foldTreeSkeleton(grow(lpl::procgen::TreeSpecies::Conifer)) == coniferFold &&
                   lpl::procgen::foldTreeSkeleton(grow(lpl::procgen::TreeSpecies::Broadleaf)) == broadleafFold &&
                   lpl::procgen::foldTreeSkeleton(grow(lpl::procgen::TreeSpecies::Shrub)) == shrubFold,
               "growing each one again gives the same tree");

    test.measureHexadecimal("conifer_signature", coniferFold);
    test.measureHexadecimal("broadleaf_signature", broadleafFold);
    test.measureHexadecimal("shrub_signature", shrubFold);
    test.measure("conifer_branches", static_cast<lpl::core::u32>(conifer.branches.size()));
    test.measure("conifer_leaves", static_cast<lpl::core::u32>(conifer.leaves.size()));
    test.measure("broadleaf_branches", static_cast<lpl::core::u32>(broadleaf.branches.size()));
    test.measure("broadleaf_leaves", static_cast<lpl::core::u32>(broadleaf.leaves.size()));
    test.measure("shrub_branches", static_cast<lpl::core::u32>(shrub.branches.size()));
    test.measure("shrub_leaves", static_cast<lpl::core::u32>(shrub.leaves.size()));
}
