/**
 * @file test_octree_cull.cpp
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
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-31
 * @copyright MIT License
 */

#include <algorithm>
#include <cstdio>
#include <lpl/math/AABB.hpp>
#include <lpl/physics/Octree.hpp>
#include <vector>

using namespace lpl;

static int failures = 0;

static void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++failures;
}

static math::AABB<math::Fixed32> boxAt(float x, float y, float z, float half)
{
    return math::AABB<math::Fixed32>{
        math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(x - half), math::Fixed32::fromFloat(y - half),
                                  math::Fixed32::fromFloat(z - half)},
        math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(x + half), math::Fixed32::fromFloat(y + half),
                                  math::Fixed32::fromFloat(z + half)}};
}

int main()
{
    std::printf("== octree hierarchical cull ==\n");

    // A world that is deliberately NOT centred on the origin and NOT a power-of-two
    // cube: those are the two cases the old absolute-grid key accidentally survived.
    constexpr int kSide = 8;
    constexpr float kSpacing = 24.0f;
    constexpr float kOriginX = 137.0f;
    constexpr float kOriginZ = -412.0f;

    std::vector<math::Vec3<float>> centres;
    for (int z = 0; z < kSide; ++z)
        for (int x = 0; x < kSide; ++x)
            centres.push_back({kOriginX + static_cast<float>(x) * kSpacing, 3.0f,
                               kOriginZ + static_cast<float>(z) * kSpacing});

    const float spanX = static_cast<float>(kSide) * kSpacing;
    physics::Octree tree{math::AABB<math::Fixed32>{}, 4u};
    tree.setWorldBounds(math::AABB<math::Fixed32>{
        math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(kOriginX - kSpacing), math::Fixed32::fromFloat(-64.0f),
                                  math::Fixed32::fromFloat(kOriginZ - kSpacing)},
        math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(kOriginX + spanX), math::Fixed32::fromFloat(64.0f),
                                  math::Fixed32::fromFloat(kOriginZ + spanX)}});

    for (std::size_t i = 0; i < centres.size(); ++i)
        tree.insert(static_cast<core::u32>(i), boxAt(centres[i].x, centres[i].y, centres[i].z, kSpacing * 0.5f));
    tree.rebuild();

    check(tree.count() == centres.size(), "every object is in the tree");

    // The region: a 3x3 window of the grid, well inside it, so nothing is decided by
    // a boundary case.
    const float regionMinX = kOriginX + 2.0f * kSpacing - kSpacing * 0.5f;
    const float regionMaxX = regionMinX + 3.0f * kSpacing;
    const float regionMinZ = kOriginZ + 2.0f * kSpacing - kSpacing * 0.5f;
    const float regionMaxZ = regionMinZ + 3.0f * kSpacing;

    const auto overlapsRegion = [&](const math::AABB<math::Fixed32> &b) {
        return !(b.max.x.toFloat() <= regionMinX || b.min.x.toFloat() >= regionMaxX ||
                 b.max.z.toFloat() <= regionMinZ || b.min.z.toFloat() >= regionMaxZ);
    };

    // Ground truth, computed without the tree.
    std::vector<core::u32> expected;
    for (std::size_t i = 0; i < centres.size(); ++i)
        if (overlapsRegion(boxAt(centres[i].x, centres[i].y, centres[i].z, kSpacing * 0.5f)))
            expected.push_back(static_cast<core::u32>(i));

    std::vector<core::u32> got;
    core::u32 visited = 0u;
    core::u32 pruned = 0u;
    tree.queryVisible([&](const math::AABB<math::Fixed32> &bound) { return overlapsRegion(bound); },
                      [&](core::u32 id) { got.push_back(id); }, &visited, &pruned);

    std::sort(got.begin(), got.end());
    std::printf("  region holds %zu objects; tree returned %zu (nodes %u, pruned %u)\n", expected.size(), got.size(),
                visited, pruned);

    check(got.size() == expected.size(), "no surplus and no loss");
    check(got == expected, "exactly the objects inside the region");
    check(pruned > 0u, "whole subtrees were rejected (the tree is doing work)");
    check(visited < static_cast<core::u32>(centres.size()), "fewer nodes tested than objects");

    // A region covering everything must return everything: a cull that prunes too
    // eagerly fails here and nowhere else.
    std::vector<core::u32> all;
    tree.queryVisible([](const math::AABB<math::Fixed32> &) { return true; },
                      [&](core::u32 id) { all.push_back(id); });
    check(all.size() == centres.size(), "an all-accepting test returns the whole set");

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
