#include <lpl/container/RingBuffer.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(ring_buffer);

namespace {

/** @brief How many elements of one kind were built, copied, moved and destroyed. */
struct Census {
    lpl::core::u32 constructed = 0u; /**< Every constructor, copies and moves included. */
    lpl::core::u32 copied = 0u;      /**< Copy constructions and copy assignments. */
    lpl::core::u32 moved = 0u;       /**< Move constructions and move assignments. */
    lpl::core::u32 destroyed = 0u;   /**< Destructor calls. */
};

/** @brief An element with no default constructor, which reports its whole life to a Census. */
class Counted final {
public:
    Counted(Census &census, lpl::core::u32 value) noexcept : _census(&census), _value(value) { ++_census->constructed; }

    Counted(const Counted &other) noexcept : _census(other._census), _value(other._value)
    {
        ++_census->constructed;
        ++_census->copied;
    }

    Counted(Counted &&other) noexcept : _census(other._census), _value(other._value)
    {
        ++_census->constructed;
        ++_census->moved;
    }

    Counted &operator=(const Counted &other) noexcept
    {
        _census = other._census;
        _value = other._value;
        ++_census->copied;
        return *this;
    }

    Counted &operator=(Counted &&other) noexcept
    {
        _census = other._census;
        _value = other._value;
        ++_census->moved;
        return *this;
    }

    ~Counted() { ++_census->destroyed; }

    [[nodiscard]] lpl::core::u32 value() const noexcept { return _value; }

private:
    Census *_census;
    lpl::core::u32 _value;
};

[[nodiscard]] bool popsInOrder(lpl::container::RingBuffer<lpl::core::u32, 8u> &ring, lpl::core::u32 first,
                               lpl::core::u32 count)
{
    lpl::core::u32 value = 0u;
    for (lpl::core::u32 i = 0u; i < count; ++i)
    {
        if (!ring.pop(value) || value != first + i)
            return false;
    }
    return true;
}

} // namespace

/**
 * @brief A ring of N slots holds N elements, refuses the next one and counts it, and hands them back
 *        oldest first.
 */
LPL_TEST(holds_as_many_elements_as_it_has_slots)
{
    lpl::container::RingBuffer<lpl::core::u32, 4u> ring;
    lpl::core::u32 value = 0u;

    test.check(ring.isEmpty() && ring.size() == 0u, "a new ring is empty");
    test.check(!ring.pop(value) && ring.front() == nullptr && !ring.popFront(), "and has nothing to pop");

    const bool filled = ring.push(10u) && ring.push(11u) && ring.push(12u) && ring.push(13u);
    test.check(filled && ring.size() == 4u && ring.isFull(), "four slots take four elements");
    test.check(!ring.push(14u) && ring.rejectedCount() == 1u, "a fifth is refused and counted");

    const bool inOrder = ring.pop(value) && value == 10u && ring.pop(value) && value == 11u && ring.pop(value) &&
                         value == 12u && ring.pop(value) && value == 13u;
    test.check(inOrder, "the four come back oldest first");
    test.check(ring.isEmpty() && !ring.pop(value), "and the ring is empty again");
}

/**
 * @brief Order holds across hundreds of wraps of the slots, whichever calls move the elements.
 */
LPL_TEST(keeps_the_order_across_many_wraps)
{
    lpl::container::RingBuffer<lpl::core::u32, 8u> ring;
    lpl::core::u32 next = 0u;
    lpl::core::u32 expected = 0u;
    bool inOrder = true;

    for (lpl::core::u32 round = 0u; round < 300u && inOrder; ++round)
    {
        for (lpl::core::u32 i = 0u; i < 5u; ++i)
            inOrder = ring.push(next++) && inOrder;
        inOrder = popsInOrder(ring, expected, 5u) && inOrder;
        expected += 5u;
    }
    test.check(inOrder && ring.isEmpty(), "1500 elements pushed and popped one by one, in order");

    lpl::core::u32 batch[7] = {};
    lpl::core::u32 drained[7] = {};
    for (lpl::core::u32 round = 0u; round < 300u && inOrder; ++round)
    {
        const lpl::core::u32 count = 3u + round % 5u;
        for (lpl::core::u32 i = 0u; i < count; ++i)
            batch[i] = next++;
        inOrder = ring.pushBulk(std::span<const lpl::core::u32>(batch, count)) == count && inOrder;
        inOrder = ring.drain(std::span<lpl::core::u32>(drained, count)) == count && inOrder;
        for (lpl::core::u32 i = 0u; i < count && inOrder; ++i)
            inOrder = drained[i] == expected++;
    }
    test.check(inOrder && ring.isEmpty(), "and in batches of 3 to 7 across the wrap of 8 slots");
}

/**
 * @brief Bulk calls move what fits: a bulk push stops at a full ring and counts the rest, a drain stops
 *        at its span, a consume at its maximum.
 */
LPL_TEST(bulk_calls_move_what_fits)
{
    lpl::container::RingBuffer<lpl::core::u32, 8u> ring;
    const lpl::core::u32 items[10] = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
    lpl::core::u32 drained[3] = {};
    lpl::core::u32 visited[8] = {};
    lpl::core::usize visitCount = 0u;

    test.check(ring.pushBulk(std::span<const lpl::core::u32>(items, 10u)) == 8u, "eight of ten fit in eight slots");
    test.check(ring.rejectedCount() == 2u && ring.isFull(), "the two left out are counted");

    const lpl::core::usize drainedCount = ring.drain(std::span<lpl::core::u32>(drained, 3u));
    test.check(drainedCount == 3u && drained[0] == 0u && drained[1] == 1u && drained[2] == 2u,
               "a drain into three places takes the three oldest");

    const lpl::core::usize consumed =
        ring.consume([&](lpl::core::u32 &value) noexcept { visited[visitCount++] = value; }, 2u);
    test.check(consumed == 2u && visitCount == 2u && visited[0] == 3u && visited[1] == 4u,
               "a consume of at most two visits the next two");

    test.check(ring.pushBulk(std::span<const lpl::core::u32>(items, 6u)) == 5u && ring.rejectedCount() == 3u,
               "five free slots take five of six, across the wrap");

    visitCount = 0u;
    const lpl::core::usize rest =
        ring.consume([&](lpl::core::u32 &value) noexcept { visited[visitCount++] = value; }, 100u);
    const bool restInOrder =
        rest == 8u && visited[0] == 5u && visited[1] == 6u && visited[2] == 7u && visited[3] == 0u && visited[7] == 4u;
    test.check(restInOrder && ring.isEmpty(), "a consume of at most a hundred takes the eight left, in order");
}

/**
 * @brief The ring constructs only what it is given: no slot is default-constructed, and emplace builds
 *        the element in its slot with no copy and no move.
 */
LPL_TEST(constructs_only_what_it_is_given)
{
    Census census;
    {
        lpl::container::RingBuffer<Counted, 8u> ring;
        test.check(census.constructed == 0u, "an empty ring of eight slots holds no element");

        const bool built = ring.emplace(census, 1u) && ring.emplace(census, 2u) && ring.emplace(census, 3u);
        test.check(built && census.constructed == 3u && census.copied == 0u && census.moved == 0u,
                   "three emplaced elements are three constructions, in place");

        Counted *const oldest = ring.front();
        test.check(oldest != nullptr && oldest->value() == 1u && ring.size() == 3u,
                   "front shows the oldest without removing it");
        test.check(ring.front() == oldest, "and shows it again until it is removed");
    }
    test.check(census.destroyed == 3u && census.destroyed == census.constructed,
               "a ring destroyed with three elements left destroys the three");
}

/**
 * @brief Every element the ring constructs is destroyed exactly once, whichever call removes it.
 */
LPL_TEST(destroys_every_element_exactly_once)
{
    Census census;
    lpl::core::u32 destroyedWhileTheRingLived = 0u;
    {
        Counted received(census, 0u);
        Counted drained[2] = {Counted(census, 0u), Counted(census, 0u)};
        lpl::core::u32 visitedSum = 0u;
        {
            lpl::container::RingBuffer<Counted, 8u> ring;

            for (lpl::core::u32 value = 1u; value <= 8u; ++value)
                ring.emplace(census, value);
            test.check(!ring.emplace(census, 9u) && ring.rejectedCount() == 1u, "a ninth element is refused");

            test.check(ring.pop(received) && received.value() == 1u, "pop moves the oldest out");
            test.check(ring.popFront() && ring.front() != nullptr && ring.front()->value() == 3u,
                       "popFront drops the next one");
            test.check(ring.drain(std::span<Counted>(drained, 2u)) == 2u && drained[1].value() == 4u,
                       "drain moves two out");
            test.check(ring.consume([&](Counted &element) noexcept { visitedSum += element.value(); }, 2u) == 2u &&
                           visitedSum == 5u + 6u,
                       "consume visits two in place");
            test.check(ring.size() == 2u, "two are left in the ring");
            destroyedWhileTheRingLived = census.destroyed;
        }
        test.check(census.destroyed == destroyedWhileTheRingLived + 2u, "the two left die with the ring");
    }
    test.check(census.destroyed == census.constructed, "every construction has its destruction");
}
