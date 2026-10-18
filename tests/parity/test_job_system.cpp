#include <lpl/concurrency/ChaseLevDeque.hpp>
#include <lpl/concurrency/JobSystem.hpp>

#include <atomic>
#include <cstdio>

namespace {

int checks = 0;
int failures = 0;

void expect(bool condition, const char *what)
{
    ++checks;
    if (condition)
        return;
    ++failures;
    std::printf("  FAIL  %s\n", what);
}

constexpr lpl::core::u32 kDequeCapacity = 4u;
constexpr lpl::core::u32 kPendingJobs = 20'000u;

/**
 * @brief Pushes one item more than a deque holds, then checks that every item it kept pops once, in
 *        last-in first-out order.
 *
 * @details push used to write past a full deque, over the oldest item not yet taken, which then never came
 * out while the item written over it came out twice.
 */
void checkFullDequeKeepsItsItems()
{
    lpl::concurrency::ChaseLevDeque<int *> deque{kDequeCapacity};
    int items[kDequeCapacity + 1u] = {};

    for (lpl::core::u32 i = 0u; i < kDequeCapacity + 1u; ++i)
        static_cast<void>(deque.push(&items[i]));

    bool inOrder = true;
    for (lpl::core::u32 i = kDequeCapacity; i > 0u; --i)
        inOrder = inOrder && deque.pop() == &items[i - 1u];
    expect(inOrder, "a full deque pops every item it held, newest first");
    expect(deque.pop() == nullptr, "and nothing after them");
}

/**
 * @brief Destroys a job system right after a burst of jobs, then checks that every job ran and the handle
 *        reached zero.
 *
 * @details The destructor used to join the workers and drop the jobs still pending: they never ran, and a
 * thread waiting on their handle would have waited forever.
 */
void checkDestructionRunsThePendingJobs()
{
    std::atomic<lpl::core::u32> ran{0u};
    lpl::concurrency::JobHandle handle;
    {
        lpl::concurrency::JobSystem jobs{1u};
        for (lpl::core::u32 i = 0u; i < kPendingJobs; ++i)
            jobs.kickJob([&ran] { ran.fetch_add(1u, std::memory_order_relaxed); }, handle);
    }
    expect(ran.load() == kPendingJobs, "every job kicked before the destructor ran");
    expect(handle.counter.load() == 0, "and the handle reached zero");
}

} // namespace

/**
 * @brief A full work-stealing deque refuses an item rather than overwriting one, and a job system that is
 *        destroyed runs the jobs still pending.
 */
int main()
{
    checkFullDequeKeepsItsItems();
    checkDestructionRunsThePendingJobs();

    if (failures == 0)
        std::printf("ALL PASS (0 failures, %d checks)\n", checks);
    else
        std::printf("FAILED (%d failures, %d checks)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}
