#include <lpl/net/session/SessionManager.hpp>
#include <lpl/net/transport/ITransport.hpp>

#include <array>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

/** @brief Counts how it is called, and records every datagram it is handed. */
class CountingTransport final : public lpl::net::transport::ITransport {
public:
    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}

    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> data,
                                             const lpl::net::Endpoint *address) override
    {
        ++sendCalls;
        ++datagrams;
        destinations.push_back(address != nullptr ? *address : lpl::net::Endpoint{});
        return static_cast<lpl::core::u32>(data.size());
    }

    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }

    const char *name() const noexcept override { return "CountingTransport"; }

    lpl::core::u32 sendCalls{0};  /**< One-at-a-time calls. */
    lpl::core::u32 batchCalls{0}; /**< Batched calls. */
    lpl::core::u32 datagrams{0};  /**< Packets actually handed over, either way. */
    std::vector<lpl::net::Endpoint> destinations;
};

/**
 * @brief Same, but overrides the batch entry point the way a real batching transport does — this is what
 *        SocketTransport (sendmmsg) and KernelTransport (one ring kick) do.
 */
class BatchingTransport final : public lpl::net::transport::ITransport {
public:
    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}

    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> data,
                                             const lpl::net::Endpoint *address) override
    {
        ++sendCalls;
        ++datagrams;
        destinations.push_back(address != nullptr ? *address : lpl::net::Endpoint{});
        return static_cast<lpl::core::u32>(data.size());
    }

    lpl::core::Expected<lpl::core::u32> sendBatch(std::span<const lpl::net::transport::Datagram> batch) override
    {
        ++batchCalls;
        for (const auto &datagram : batch)
        {
            ++datagrams;
            destinations.push_back(datagram.address != nullptr ? *datagram.address : lpl::net::Endpoint{});
        }
        return static_cast<lpl::core::u32>(batch.size());
    }

    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }

    const char *name() const noexcept override { return "BatchingTransport"; }

    lpl::core::u32 sendCalls{0};
    lpl::core::u32 batchCalls{0};
    lpl::core::u32 datagrams{0};
    std::vector<lpl::net::Endpoint> destinations;
};

/**
 * @brief Connect @p count clients, each with its own address.
 */
void seed(lpl::net::session::SessionManager &sessions, lpl::core::u32 count)
{
    for (lpl::core::u32 i = 0; i < count; ++i)
    {
        auto joined = sessions.connect(i + 1);
        if (joined.has_value())
            joined.value()->setAddress(
                lpl::net::Endpoint::fromOctets(127, 0, 0, 1, static_cast<lpl::core::u16>(40000 + i)));
    }
}

/**
 * @brief A transport with no batching still receives every packet.
 *
 * @details The default sendBatch loops over send(), so a platform without a batching syscall (Windows, macOS)
 *          keeps working untouched.
 *
 * @param payload           The broadcast payload.
 * @param kClients          The number of clients connected.
 * @param expectedDatagrams The datagrams the broadcast must hand over.
 */
void checkNonBatchingTransportStillGetsEveryPacket(const std::vector<lpl::core::byte> &payload, lpl::core::u32 kClients,
                                                   lpl::core::u32 expectedDatagrams)
{
    lpl::net::session::SessionManager sessions;
    seed(sessions, kClients);
    CountingTransport transport;

    check(sessions.broadcastState(transport, payload).has_value(), "a packet that fills one datagram is accepted");

    check(transport.datagrams == expectedDatagrams, "non-batching transport still gets every datagram");
    check(transport.sendCalls == expectedDatagrams, "and pays one send() per datagram, as before");
    check(transport.batchCalls == 0, "it never sees a batch call it did not implement");
}

/**
 * @brief A batching transport gets the whole burst in ONE call.
 *
 * @param payload           The broadcast payload.
 * @param kClients          The number of clients connected.
 * @param expectedDatagrams The datagrams the broadcast must hand over.
 */
void checkBatchingTransportGetsTheWholeBurstInOneCall(const std::vector<lpl::core::byte> &payload,
                                                      lpl::core::u32 kClients, lpl::core::u32 expectedDatagrams)
{
    lpl::net::session::SessionManager sessions;
    seed(sessions, kClients);
    BatchingTransport transport;

    check(sessions.broadcastState(transport, payload).has_value(), "a packet that fills one datagram is accepted");

    check(transport.batchCalls == 1, "a batching transport is called exactly once for the whole broadcast");
    check(transport.sendCalls == 0, "and never falls back to one-at-a-time sends");
    check(transport.datagrams == expectedDatagrams, "while still delivering every datagram");
}

/**
 * @brief Every client is addressed once, none is dropped or duplicated.
 *
 * @param payload  The broadcast payload.
 * @param kClients The number of clients connected.
 */
void checkEveryClientIsAddressedOnce(const std::vector<lpl::core::byte> &payload, lpl::core::u32 kClients)
{
    lpl::net::session::SessionManager sessions;
    seed(sessions, kClients);
    BatchingTransport transport;

    check(sessions.broadcastState(transport, payload).has_value(), "a packet that fills one datagram is accepted");

    bool everyClientAddressed = true;
    for (lpl::core::u32 i = 0; i < kClients; ++i)
    {
        const auto expected = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, static_cast<lpl::core::u16>(40000 + i));
        lpl::core::u32 seen = 0;
        for (const auto &destination : transport.destinations)
        {
            if (destination == expected)
                ++seen;
        }
        everyClientAddressed = everyClientAddressed && seen == 1u;
    }
    check(everyClientAddressed, "each client receives the packet exactly once");
}

/**
 * @brief A state packet longer than one datagram is refused, and nothing is sent.
 *
 * @details broadcastState used to cut such a packet into raw slices, without a header per slice: the client read
 *          the first as a whole packet, truncated, and dropped the others.
 *
 * @param kClients The number of clients connected.
 */
void checkAPacketLongerThanADatagramIsRefused(lpl::core::u32 kClients)
{
    lpl::net::session::SessionManager sessions;
    seed(sessions, kClients);
    BatchingTransport transport;
    const std::vector<lpl::core::byte> tooLong(lpl::net::session::SessionManager::kMaxPayloadSize + 1u,
                                               lpl::core::byte{0xAB});

    const auto sent = sessions.broadcastState(transport, tooLong);

    check(!sent.has_value(), "a packet one byte longer than a datagram is refused");
    check(transport.batchCalls == 0 && transport.sendCalls == 0, "and touches the transport zero times");
}

/**
 * @brief No clients means no call at all.
 *
 * @param payload The broadcast payload.
 */
void checkNoClientsMeansNoCall(const std::vector<lpl::core::byte> &payload)
{
    lpl::net::session::SessionManager sessions;
    BatchingTransport transport;

    check(sessions.broadcastState(transport, payload).has_value(), "a packet that fills one datagram is accepted");

    check(transport.batchCalls == 0 && transport.sendCalls == 0,
          "a broadcast with no connected client touches the transport zero times");
}

/**
 * @brief receiveBatch: the symmetric receive path.
 *
 * @details A transport with a real recvmmsg drains a burst in one call; the default implementation loops
 *          receive(). Both must surface every queued packet. QueuedReceiveTransport has N packets queued and
 *          only implements receive(): the base receiveBatch must loop until the queue drains.
 */
void checkDefaultReceiveBatchDrainsEveryQueuedPacket()
{
    class QueuedReceiveTransport final : public lpl::net::transport::ITransport {
    public:
        lpl::core::Expected<void> open() override { return {}; }
        void close() override {}
        lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte>, const lpl::net::Endpoint *) override
        {
            return lpl::core::u32{0};
        }
        lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte> buffer,
                                                    lpl::net::Endpoint *from) override
        {
            ++receiveCalls;
            if (delivered >= total)
                return lpl::core::u32{0};
            ++delivered;
            if (from != nullptr)
                *from = lpl::net::Endpoint::fromOctets(10, 0, 0, 1, static_cast<lpl::core::u16>(50000 + delivered));
            buffer[0] = lpl::core::byte{0x01};
            return lpl::core::u32{1};
        }
        const char *name() const noexcept override { return "QueuedReceiveTransport"; }
        lpl::core::u32 total{5};
        lpl::core::u32 delivered{0};
        lpl::core::u32 receiveCalls{0};
    };

    QueuedReceiveTransport transport;
    std::array<lpl::core::byte, 64 * 8> storage{};
    std::array<lpl::net::transport::ReceiveSlot, 8> slots{};
    for (lpl::core::usize i = 0; i < slots.size(); ++i)
        slots[i].buffer = std::span<lpl::core::byte>{storage.data() + i * 8, 8};

    auto received = transport.receiveBatch(std::span<lpl::net::transport::ReceiveSlot>{slots.data(), slots.size()});
    check(received.has_value() && received.value() == 5, "the default receiveBatch drains every queued packet");
    check(slots[0].length == 1 && slots[4].length == 1, "each drained slot carries its length");
    check(slots[0].source != slots[1].source, "each slot carries its own sender");
}

} // namespace

/**
 * @brief Transport batching test.
 *
 * @details A broadcast is one packet per client, all ready at once. Handing them to the transport one at a
 *          time costs one syscall each; handed over together, a transport that can batch does (sendmmsg on
 *          Linux sockets, a single ring kick for the kernel module). This counts the CALLS, because an
 *          optimisation nobody measured is a claim, not a result. The payload fills one datagram exactly.
 */
int main()
{
    std::printf("== transport batching ==\n");

    constexpr lpl::core::u32 kClients = 5;

    const std::vector<lpl::core::byte> payload(lpl::net::session::SessionManager::kMaxPayloadSize,
                                               lpl::core::byte{0xAB});

    checkNonBatchingTransportStillGetsEveryPacket(payload, kClients, kClients);
    checkBatchingTransportGetsTheWholeBurstInOneCall(payload, kClients, kClients);
    checkEveryClientIsAddressedOnce(payload, kClients);
    checkAPacketLongerThanADatagramIsRefused(kClients);
    checkNoClientsMeansNoCall(payload);
    checkDefaultReceiveBatchDrainsEveryQueuedPacket();

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
