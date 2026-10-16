#include <lpl/core/Constants.hpp>
#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Morton.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(world_partition);

namespace {

/** @brief Cell sides, raw: one unit and just over it, the engine's ten units, odd ones, the end of the word, and two
 * under one unit. */
constexpr lpl::core::i32 kCellSides[] = {65536, 65537, 655360, 3 * 65536 + 12345, 99999, 2147483647, 32768, 1};

/** @brief Coordinates, raw: zero, one unit either side of it, and the ends of the word. */
constexpr lpl::core::i32 kEdgeCoordinates[] = {0, 1, -1, 65535, -65535, 65536, -65536, 2147483647, -2147483647 - 1};

/**
 * @brief The grid coordinate the partition used to take: the Fixed32 quotient, its fraction dropped, biased.
 */
[[nodiscard]] lpl::core::i32 quotientCell(lpl::core::i32 coordinate, lpl::core::i32 side)
{
    return (lpl::math::Fixed32::fromRaw(coordinate) / lpl::math::Fixed32::fromRaw(side)).toInt() +
           lpl::core::kMortonBias;
}

} // namespace

/**
 * @brief A cell key is the one the Fixed32 quotient gives, on both sides of zero and of every cell's edge.
 *
 * @details For each cell side, coordinates at the ends of the word, at each side of the cell
 *          edges around zero, at each side of the remainder at which a negative coordinate's
 *          quotient rounds into the cell below, and from a linear congruential generator, are
 *          keyed by the partition and by the quotient, and the keys are folded so the two targets
 *          can be compared too.
 */
LPL_TEST(a_cell_key_is_the_one_the_fixed_point_quotient_gives)
{
    lpl::core::u32 mismatches = 0u;
    lpl::core::u32 keyed = 0u;
    lpl::core::u32 negative = 0u;
    lpl::core::u32 underRounding = 0u;
    lpl::core::u32 fold = 0x811C9DC5u;
    lpl::core::u32 state = 0x2545F491u;

    for (const lpl::core::i32 side : kCellSides)
    {
        lpl::ecs::WorldPartition partition{lpl::math::Fixed32::fromRaw(side), 16u};
        const auto keyAndCompare = [&](lpl::core::i32 coordinate) {
            const lpl::core::i32 across = coordinate / -3;
            const lpl::core::i32 above = coordinate / 7 + 5;
            const lpl::core::u64 key =
                partition.mortonForPosition({lpl::math::Fixed32::fromRaw(coordinate),
                                             lpl::math::Fixed32::fromRaw(across), lpl::math::Fixed32::fromRaw(above)});
            const lpl::core::u64 expected = lpl::math::morton::encode3D(
                quotientCell(coordinate, side), quotientCell(across, side), quotientCell(above, side));
            if (key != expected)
                ++mismatches;
            ++keyed;
            if (coordinate < 0)
            {
                ++negative;
                const lpl::core::u32 below = 0u - static_cast<lpl::core::u32>(coordinate);
                const lpl::core::u32 rest = below % static_cast<lpl::core::u32>(side);
                if (rest != 0u && rest < ((static_cast<lpl::core::u32>(side) + 0xFFFFu) >> 16u))
                    ++underRounding;
            }
            fold = (fold ^ static_cast<lpl::core::u32>(key)) * 0x01000193u;
            fold = (fold ^ static_cast<lpl::core::u32>(key >> 32u)) * 0x01000193u;
        };

        for (const lpl::core::i32 coordinate : kEdgeCoordinates)
            keyAndCompare(coordinate);

        const lpl::core::i32 rounding =
            static_cast<lpl::core::i32>((static_cast<lpl::core::u32>(side) + 0xFFFFu) >> 16u);
        for (lpl::core::i32 cells = 0; cells < 4; ++cells)
        {
            const lpl::core::i64 edge = static_cast<lpl::core::i64>(cells) * side;
            for (const lpl::core::i64 offset : {lpl::core::i64{-1}, lpl::core::i64{0}, lpl::core::i64{1},
                                                static_cast<lpl::core::i64>(rounding) - 1, lpl::core::i64{rounding}})
            {
                for (const lpl::core::i64 coordinate : {-(edge + offset), edge + offset})
                {
                    if (coordinate >= -2147483647LL - 1LL && coordinate <= 2147483647LL)
                        keyAndCompare(static_cast<lpl::core::i32>(coordinate));
                }
            }
        }

        for (lpl::core::u32 draw = 0u; draw < 2000u; ++draw)
        {
            state = state * 1664525u + 1013904223u;
            keyAndCompare(static_cast<lpl::core::i32>(state) >> (state & 31u));
        }
    }

    test.check(mismatches == 0u, "every key is the one the Fixed32 quotient gives");
    test.check(negative > keyed / 4u && underRounding > 0u,
               "and negative coordinates were keyed, some with a remainder too small to round into the cell below");

    test.measure("coordinates_keyed", keyed);
    test.measure("negative_coordinates", negative);
    test.measure("remainders_under_the_rounding", underRounding);
    test.measureHexadecimal("key_signature", fold);
}
