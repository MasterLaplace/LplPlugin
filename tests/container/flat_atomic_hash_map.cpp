#include <lpl/container/FlatAtomicHashMap.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(flat_atomic_hash_map);

namespace {

/** @brief Pool entries of the map under test: its table holds 16 entries, so keys 16 apart share a probe. */
constexpr lpl::core::u32 kCapacity = 8u;
constexpr lpl::core::u64 kFirst = 3u;
constexpr lpl::core::u64 kSecond = kFirst + 16u;
constexpr lpl::core::u64 kThird = kFirst + 32u;

} // namespace

/**
 * @brief A key inserted again after a removal earlier in its probe sequence is found, not stored twice.
 *
 * @details Insertion used to take the first tombstone of the sequence, before the entry that held the key:
 * the key was then stored twice, and removing it once left the other copy behind.
 */
LPL_TEST(a_key_past_a_tombstone_is_found_again)
{
    lpl::container::FlatAtomicHashMap<lpl::core::u32> map{kCapacity};

    *map.insert(kFirst) = 1u;
    *map.insert(kSecond) = 2u;
    test.check(map.remove(kFirst), "the first key of the sequence is removed");

    const lpl::core::u32 *again = map.insert(kSecond);
    test.check(again != nullptr && *again == 2u, "inserting the second key again returns its value");
    test.check(map.size() == 1u, "and stores it once");
    test.check(map.remove(kSecond) && map.get(kSecond) == nullptr, "so one removal leaves no copy behind");
}

/**
 * @brief A new key inserted past a tombstone is stored, and the key beyond the tombstone stays found.
 */
LPL_TEST(a_new_key_past_a_tombstone_is_stored)
{
    lpl::container::FlatAtomicHashMap<lpl::core::u32> map{kCapacity};

    *map.insert(kFirst) = 1u;
    *map.insert(kSecond) = 2u;
    test.check(map.remove(kFirst), "the first key of the sequence is removed");
    *map.insert(kThird) = 3u;

    test.check(map.size() == 2u, "two keys are stored");
    test.check(map.get(kSecond) != nullptr && *map.get(kSecond) == 2u, "the key beyond the tombstone keeps its value");
    test.check(map.get(kThird) != nullptr && *map.get(kThird) == 3u, "the new key holds its own");
    test.check(map.get(kFirst) == nullptr, "and the removed key is gone");
}
