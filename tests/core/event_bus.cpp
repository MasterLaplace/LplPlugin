#include <lpl/core/EventBus.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(event_bus);

namespace {

/**
 * @brief An event the test publishes, with the value its subscriber adds up.
 */
struct Tally {
    lpl::core::u32 value; /**< What the subscriber adds. */
};

} // namespace

/**
 * @brief A subscriber receives what is published while it listens and nothing after it leaves, in
 *        ring 0 as on the host.
 */
LPL_TEST(a_subscriber_receives_what_is_published_while_it_listens)
{
    lpl::core::EventBus &bus = lpl::core::EventBus::instance();
    lpl::core::u32 received = 0u;
    const lpl::core::u32 subscription =
        bus.subscribe<Tally>([&received](const Tally &event) { received += event.value; });

    test.check(bus.hasSubscribers(), "the bus has a subscriber");
    bus.publish(Tally{3u});
    bus.publish(Tally{4u});
    bus.unsubscribe<Tally>(subscription);
    bus.publish(Tally{5u});

    test.check(received == 7u, "the subscriber receives the two events published while it listens");
    test.check(!bus.hasSubscribers(), "and the bus has none once it has left");
    test.measure("received", received);
}
