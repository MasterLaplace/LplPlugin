#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(arena_allocator);

namespace {

[[nodiscard]] bool isAlignedOnEight(const void *pointer)
{
    return (reinterpret_cast<lpl::core::usize>(pointer) & 7u) == 0u;
}

} // namespace

/**
 * @brief Gate P1 arena: the arena hands out aligned blocks it owns, refuses what it cannot hold,
 *        and takes everything back on reset.
 */
LPL_TEST(allocates_aligned_blocks_and_resets)
{
    constexpr lpl::core::usize kCapacity = 256u;
    lpl::memory::ArenaAllocator arena(kCapacity);
    void *const first = arena.allocate(64u, 8u);
    void *const second = arena.allocate(32u, 8u);
    void *const third = arena.allocate(16u, 8u);
    int notInTheArena = 0;

    if (!test.check(first != nullptr && second != nullptr && third != nullptr, "three blocks fit in 256 bytes"))
        return;

    test.check(isAlignedOnEight(first) && isAlignedOnEight(second) && isAlignedOnEight(third),
               "each block is aligned on eight bytes");
    test.check(arena.ownsPtr(first) && arena.ownsPtr(third), "the arena owns the blocks it handed out");
    test.check(!arena.ownsPtr(&notInTheArena), "and not a variable on the stack");
    test.check(arena.used() == 64u + 32u + 16u, "the blocks take their sizes and no padding");
    test.check(arena.allocate(kCapacity, 8u) == nullptr, "a block larger than what is left is refused");

    arena.reset();
    test.check(arena.used() == 0u, "reset takes every byte back");
}
