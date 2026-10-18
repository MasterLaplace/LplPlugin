#include <algorithm>
#include <cstdio>
#include <lpl/math/AABB.hpp>
#include <lpl/physics/Octree.hpp>
#include <vector>

static int failures = 0;

/**
 * @name The world
 * A world that is deliberately NOT centred on the origin and NOT a power-of-two
 * cube: those are the two cases the old absolute-grid key accidentally survived.
 * @{
 */
static constexpr int kSide = 8;
static constexpr float kSpacing = 24.0f;
static constexpr float kOriginX = 137.0f;
static constexpr float kOriginZ = -412.0f;
/** @} */

static void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++failures;
}

static lpl::math::AABB<lpl::math::Fixed32> boxAt(float x, float y, float z, float half)
{
    return lpl::math::AABB<lpl::math::Fixed32>{
        lpl::math::Vec3<lpl::math::Fixed32>{lpl::math::Fixed32::fromFloat(x - half),
                                            lpl::math::Fixed32::fromFloat(y - half),
                                            lpl::math::Fixed32::fromFloat(z - half)},
        lpl::math::Vec3<lpl::math::Fixed32>{lpl::math::Fixed32::fromFloat(x + half),
                                            lpl::math::Fixed32::fromFloat(y + half),
                                            lpl::math::Fixed32::fromFloat(z + half)}
    };
}

/**
 * @brief Checks that a cull over a region returns exactly the objects inside it, and rejects whole subtrees.
 *
 * @details The region is a 3x3 window of the grid, well inside it, so nothing is decided by a boundary case.
 *          The ground truth is computed without the tree.
 */
static void checkCullReturnsExactlyTheRegion(const lpl::physics::Octree &tree,
                                             const std::vector<lpl::math::Vec3<float>> &centres)
{
    const float regionMinX = kOriginX + 2.0f * kSpacing - kSpacing * 0.5f;
    const float regionMaxX = regionMinX + 3.0f * kSpacing;
    const float regionMinZ = kOriginZ + 2.0f * kSpacing - kSpacing * 0.5f;
    const float regionMaxZ = regionMinZ + 3.0f * kSpacing;

    const auto overlapsRegion = [&](const lpl::math::AABB<lpl::math::Fixed32> &b) {
        return !(b.max.x.toFloat() <= regionMinX || b.min.x.toFloat() >= regionMaxX ||
                 b.max.z.toFloat() <= regionMinZ || b.min.z.toFloat() >= regionMaxZ);
    };

    std::vector<lpl::core::u32> expected;
    for (std::size_t i = 0; i < centres.size(); ++i)
        if (overlapsRegion(boxAt(centres[i].x, centres[i].y, centres[i].z, kSpacing * 0.5f)))
            expected.push_back(static_cast<lpl::core::u32>(i));

    std::vector<lpl::core::u32> got;
    lpl::core::u32 visited = 0u;
    lpl::core::u32 pruned = 0u;
    tree.queryVisible([&](const lpl::math::AABB<lpl::math::Fixed32> &bound) { return overlapsRegion(bound); },
                      [&](lpl::core::u32 id) { got.push_back(id); }, &visited, &pruned);

    std::sort(got.begin(), got.end());
    std::printf("  region holds %zu objects; tree returned %zu (nodes %u, pruned %u)\n", expected.size(), got.size(),
                visited, pruned);

    check(got.size() == expected.size(), "no surplus and no loss");
    check(got == expected, "exactly the objects inside the region");
    check(pruned > 0u, "whole subtrees were rejected (the tree is doing work)");
    check(visited < static_cast<lpl::core::u32>(centres.size()), "fewer nodes tested than objects");
}

/**
 * @brief Checks that a region covering everything returns everything.
 *
 * @details A cull that prunes too eagerly fails here and nowhere else.
 */
static void checkEverythingComesBackFromAnAllAcceptingTest(const lpl::physics::Octree &tree,
                                                           const std::vector<lpl::math::Vec3<float>> &centres)
{
    std::vector<lpl::core::u32> all;
    tree.queryVisible([](const lpl::math::AABB<lpl::math::Fixed32> &) { return true; },
                      [&](lpl::core::u32 id) { all.push_back(id); });
    check(all.size() == centres.size(), "an all-accepting test returns the whole set");
}

/**
 * @brief The octree's hierarchy has to AGREE with its own node bounds.
 *
 * This test exists because the tree passed everything else while being wrong. A
 * broad-phase forgives a mis-shaped hierarchy: a bad node still gets descended,
 * every object in it is compared individually, and the pair list comes out right —
 * slower, but right. A CULLER does not forgive it, because a culler trusts the node
 * bounds to speak for a subtree. So the defect only became visible the day something
 * asked the tree to prune, and it showed up as thirty-one payloads returned for
 * eighteen visible boxes.
 *
 * Two things were wrong at once. @c encode3D biases its arguments itself, and the
 * caller had biased them already, so the sum wrapped the twenty-one-bit field and
 * the key's high bits were noise. And even unwrapped, the key was in absolute world
 * units while the node split read bit (20 - depth) as an octant of the tree's OWN
 * bounds; the two only agree for a world that happens to be the 2^21 cube on the
 * origin.
 *
 * What is checked here is therefore the invariant, not a signature: every object the
 * tree hands back must be inside the region asked for, and every object inside it
 * must come back. Exactly — no surplus, no loss. Plus the thing that makes a tree
 * worth its rebuild: that whole subtrees are actually rejected.
 */
int main()
{
    std::printf("== octree hierarchical cull ==\n");

    std::vector<lpl::math::Vec3<float>> centres;
    for (int z = 0; z < kSide; ++z)
        for (int x = 0; x < kSide; ++x)
            centres.push_back(
                {kOriginX + static_cast<float>(x) * kSpacing, 3.0f, kOriginZ + static_cast<float>(z) * kSpacing});

    const float spanX = static_cast<float>(kSide) * kSpacing;
    lpl::physics::Octree tree{lpl::math::AABB<lpl::math::Fixed32>{}, 4u};
    tree.setWorldBounds(lpl::math::AABB<lpl::math::Fixed32>{
        lpl::math::Vec3<lpl::math::Fixed32>{lpl::math::Fixed32::fromFloat(kOriginX - kSpacing),
                                            lpl::math::Fixed32::fromFloat(-64.0f),
                                            lpl::math::Fixed32::fromFloat(kOriginZ - kSpacing)},
        lpl::math::Vec3<lpl::math::Fixed32>{lpl::math::Fixed32::fromFloat(kOriginX + spanX),
                                            lpl::math::Fixed32::fromFloat(64.0f),
                                            lpl::math::Fixed32::fromFloat(kOriginZ + spanX)   }
    });

    for (std::size_t i = 0; i < centres.size(); ++i)
        tree.insert(static_cast<lpl::core::u32>(i), boxAt(centres[i].x, centres[i].y, centres[i].z, kSpacing * 0.5f));
    tree.rebuild();

    check(tree.count() == centres.size(), "every object is in the tree");

    checkCullReturnsExactlyTheRegion(tree, centres);

    checkEverythingComesBackFromAnAllAcceptingTest(tree, centres);

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
