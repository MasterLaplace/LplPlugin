#include <lpl/math/AABB.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/physics/Octree.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(octree);

namespace {

using Box = lpl::math::AABB<lpl::math::Fixed32>;

/** @brief Boxes in the tree; enough to split it several levels deep with a leaf of four. */
constexpr lpl::core::u32 kBoxes = 300u;

/** @brief Regions queried. */
constexpr lpl::core::u32 kQueries = 64u;

/** @brief Objects a leaf holds before it splits: small, so the queries cross many nodes. */
constexpr lpl::core::u32 kLeafCapacity = 4u;

/**
 * @brief A deterministic generator of words: the same sequence on every target.
 */
struct Words {
    lpl::core::u32 state = 0x2545F491u;

    /** @brief The next word of the sequence. */
    [[nodiscard]] lpl::core::u32 next() noexcept
    {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }

    /** @brief A Fixed32 in [@p low, @p low + @p span) whole units, in steps of 1/256. */
    [[nodiscard]] lpl::math::Fixed32 within(lpl::core::i32 low, lpl::core::u32 span) noexcept
    {
        return lpl::math::Fixed32::fromRaw(low * 65536 + static_cast<lpl::core::i32>((next() % (span * 256u)) << 8u));
    }

    /** @brief A box of sides up to @p size, somewhere in [-@p half, @p half) on each axis. */
    [[nodiscard]] Box box(lpl::core::i32 half, lpl::core::u32 size) noexcept
    {
        const lpl::math::Vec3<lpl::math::Fixed32> low{within(-half, static_cast<lpl::core::u32>(2 * half)),
                                                      within(-half, static_cast<lpl::core::u32>(2 * half)),
                                                      within(-half, static_cast<lpl::core::u32>(2 * half))};
        const lpl::math::Vec3<lpl::math::Fixed32> side{within(0, size), within(0, size), within(0, size)};
        return Box{
            low,
            {low.x + side.x, low.y + side.y, low.z + side.z}
        };
    }
};

/** @brief The boxes of the tree, kept to check a query against all of them. */
Box gBoxes[kBoxes];

} // namespace

/**
 * @brief A query reports every box the region intersects, each once, and no other, however deep the
 *        tree and wherever the region falls on its split planes.
 *
 * @details The order the query reports them in decides the order collisions resolve in, so it is
 *          folded and compared between the targets.
 */
LPL_TEST(a_query_reports_exactly_the_boxes_it_intersects)
{
    const Box world{
        {lpl::math::Fixed32::fromInt(-64), lpl::math::Fixed32::fromInt(-64), lpl::math::Fixed32::fromInt(-64)},
        {lpl::math::Fixed32::fromInt(64),  lpl::math::Fixed32::fromInt(64),  lpl::math::Fixed32::fromInt(64) }
    };
    lpl::physics::Octree tree{world, kLeafCapacity};
    Words words;

    for (lpl::core::u32 index = 0u; index < kBoxes; ++index)
    {
        gBoxes[index] = words.box(60, 6u);
        tree.insert(index, gBoxes[index]);
    }
    tree.rebuild();

    lpl::core::u32 reported = 0u;
    lpl::core::u32 missed = 0u;
    lpl::core::u32 wrong = 0u;
    lpl::core::u32 order = 0x811C9DC5u;
    for (lpl::core::u32 query = 0u; query < kQueries; ++query)
    {
        const Box region = words.box(64, 40u);
        bool seen[kBoxes] = {};

        tree.query(region, [&](lpl::core::u32 id) {
            const bool known = id < kBoxes;
            wrong += (!known || seen[id] || !gBoxes[id].intersects(region)) ? 1u : 0u;
            if (known)
                seen[id] = true;
            ++reported;
            order = (order ^ id) * 0x01000193u;
        });
        for (lpl::core::u32 index = 0u; index < kBoxes; ++index)
            missed += (gBoxes[index].intersects(region) && !seen[index]) ? 1u : 0u;
    }

    test.check(reported > kQueries, "the regions reach boxes");
    test.check(wrong == 0u, "no query reports a box twice, or one it does not intersect");
    test.check(missed == 0u, "and none leaves out a box it intersects");
    test.measure("reported", reported);
    test.measure("missed", missed);
    test.measureHexadecimal("order", order);
}
