#include <lpl/core/EventBus.hpp>

#include <cstdio>

namespace {

/** @brief An event the test subscribes to and publishes. */
struct Ping {
    int value; /**< What a subscriber adds up. */
};

/** @brief An event of another type. */
struct Pong {
    int value; /**< Never added up. */
};

/**
 * @brief Prints @p claim as passed or failed, and counts it in @p failures when it failed.
 */
void check(bool passed, const char *claim, int &failures)
{
    std::printf("  %s %s\n", passed ? "PASS" : "FAIL", claim);
    if (!passed)
        ++failures;
}

} // namespace

/**
 * @brief Checks that the event bus knows whether anyone listens, through subscriptions, withdrawals
 *        and a clear, and that an event still reaches its subscribers and only them.
 *
 * A publisher reads EventBus::hasSubscribers to skip building an event nobody can receive, and
 * publish returns at once on it: a count that drifted from the handlers would drop events in
 * silence, or keep paying for them.
 */
int main()
{
    lpl::core::EventBus &bus = lpl::core::EventBus::instance();
    int failures = 0;
    int received = 0;

    check(!bus.hasSubscribers(), "nobody listens before any subscription", failures);

    const lpl::core::u32 ping = bus.subscribe<Ping>([&received](const Ping &event) { received += event.value; });
    check(bus.hasSubscribers(), "someone listens after a subscription", failures);

    bus.publish(Ping{3});
    bus.publish(Pong{5});
    check(received == 3, "an event reaches the subscribers of its type, and only them", failures);

    bus.unsubscribe<Pong>(ping);
    check(bus.hasSubscribers(), "withdrawing an id from another type withdraws nothing", failures);

    bus.unsubscribe<Ping>(ping);
    check(!bus.hasSubscribers(), "nobody listens once the only subscription is withdrawn", failures);

    bus.publish(Ping{7});
    check(received == 3, "an event published to nobody reaches nobody", failures);

    (void) bus.subscribe<Ping>([&received](const Ping &event) { received += event.value; });
    (void) bus.subscribe<Pong>([&received](const Pong &event) { received += event.value; });
    bus.publish(Pong{11});
    check(received == 14, "a second type's subscriber receives its events", failures);

    bus.clear();
    check(!bus.hasSubscribers(), "nobody listens after a clear", failures);

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
