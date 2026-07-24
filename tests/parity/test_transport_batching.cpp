/*
** LplPlugin — transport batching test
**
** A broadcast is fragments × clients packets, all ready at once. Handing them
** to the transport one at a time costs one syscall each; handed over together,
** a transport that can batch does (sendmmsg on Linux sockets, a single ring
** kick for the kernel module). This counts the CALLS, because an optimisation
** nobody measured is a claim, not a result.
*/

#include <lpl/net/session/SessionManager.hpp>
#include <lpl/net/transport/ITransport.hpp>

#include <array>
#include <cstdio>
#include <vector>

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

/// Counts how it is called, and records every datagram it is handed.
class CountingTransport final : public net::transport::ITransport {
public:
    core::Expected<void> open() override { return {}; }
    void close() override {}

    core::Expected<core::u32> send(std::span<const core::byte> data, const net::Endpoint *address) override
    {
        ++sendCalls;
        ++datagrams;
        destinations.push_back(address != nullptr ? *address : net::Endpoint{});
        return static_cast<core::u32>(data.size());
    }

    core::Expected<core::u32> receive(std::span<core::byte>, net::Endpoint *) override { return core::u32{0}; }

    const char *name() const noexcept override { return "CountingTransport"; }

    core::u32 sendCalls{0};  ///< One-at-a-time calls.
    core::u32 batchCalls{0}; ///< Batched calls.
    core::u32 datagrams{0};  ///< Packets actually handed over, either way.
    std::vector<net::Endpoint> destinations;
};

/// Same, but overrides the batch entry point the way a real batching transport
/// does — this is what SocketTransport (sendmmsg) and KernelTransport (one ring
/// kick) do.
class BatchingTransport final : public net::transport::ITransport {
public:
    core::Expected<void> open() override { return {}; }
    void close() override {}

    core::Expected<core::u32> send(std::span<const core::byte> data, const net::Endpoint *address) override
    {
        ++sendCalls;
        ++datagrams;
        destinations.push_back(address != nullptr ? *address : net::Endpoint{});
        return static_cast<core::u32>(data.size());
    }

    core::Expected<core::u32> sendBatch(std::span<const net::transport::Datagram> batch) override
    {
        ++batchCalls;
        for (const auto &datagram : batch)
        {
            ++datagrams;
            destinations.push_back(datagram.address != nullptr ? *datagram.address : net::Endpoint{});
        }
        return static_cast<core::u32>(batch.size());
    }

    core::Expected<core::u32> receive(std::span<core::byte>, net::Endpoint *) override { return core::u32{0}; }

    const char *name() const noexcept override { return "BatchingTransport"; }

    core::u32 sendCalls{0};
    core::u32 batchCalls{0};
    core::u32 datagrams{0};
    std::vector<net::Endpoint> destinations;
};

/// Connect @p count clients, each with its own address.
void seed(net::session::SessionManager &sessions, core::u32 count)
{
    for (core::u32 i = 0; i < count; ++i)
    {
        auto joined = sessions.connect(i + 1);
        if (joined.has_value())
            joined.value()->setAddress(net::Endpoint::fromOctets(127, 0, 0, 1, static_cast<core::u16>(40000 + i)));
    }
}

} // namespace

int main()
{
    std::printf("== transport batching ==\n");

    constexpr core::u32 kClients = 5;

    // A payload big enough to need several fragments, so the burst is
    // fragments × clients rather than just one packet per client.
    const core::usize payloadSize = net::session::SessionManager::kMaxPayloadSize * 3 + 16;
    std::vector<core::byte> payload(payloadSize, core::byte{0xAB});

    const core::u32 expectedFragments = 4; // 3 full + 1 partial
    const core::u32 expectedDatagrams = expectedFragments * kClients;

    // --- a transport with no batching still receives every packet ----------- //
    // The default sendBatch loops over send(), so a platform without a batching
    // syscall (Windows, macOS) keeps working untouched.
    {
        net::session::SessionManager sessions;
        seed(sessions, kClients);
        CountingTransport transport;

        sessions.broadcastState(transport, payload);

        check(transport.datagrams == expectedDatagrams, "non-batching transport still gets every datagram");
        check(transport.sendCalls == expectedDatagrams, "and pays one send() per datagram, as before");
        check(transport.batchCalls == 0, "it never sees a batch call it did not implement");
    }

    // --- a batching transport gets the whole burst in ONE call -------------- //
    {
        net::session::SessionManager sessions;
        seed(sessions, kClients);
        BatchingTransport transport;

        sessions.broadcastState(transport, payload);

        check(transport.batchCalls == 1, "a batching transport is called exactly once for the whole broadcast");
        check(transport.sendCalls == 0, "and never falls back to one-at-a-time sends");
        check(transport.datagrams == expectedDatagrams, "while still delivering every datagram");
    }

    // --- every client is addressed, none is dropped or duplicated ----------- //
    {
        net::session::SessionManager sessions;
        seed(sessions, kClients);
        BatchingTransport transport;

        sessions.broadcastState(transport, payload);

        bool everyClientAddressed = true;
        for (core::u32 i = 0; i < kClients; ++i)
        {
            const auto expected = net::Endpoint::fromOctets(127, 0, 0, 1, static_cast<core::u16>(40000 + i));
            core::u32 seen = 0;
            for (const auto &destination : transport.destinations)
            {
                if (destination == expected)
                    ++seen;
            }
            everyClientAddressed = everyClientAddressed && seen == expectedFragments;
        }
        check(everyClientAddressed, "each client receives each fragment exactly once");
    }

    // --- no clients means no call at all ------------------------------------ //
    {
        net::session::SessionManager sessions;
        BatchingTransport transport;

        sessions.broadcastState(transport, payload);

        check(transport.batchCalls == 0 && transport.sendCalls == 0,
              "a broadcast with no connected client touches the transport zero times");
    }

    // --- receiveBatch: the symmetric receive path --------------------------- //
    // A transport with a real recvmmsg drains a burst in one call; the default
    // implementation loops receive(). Both must surface every queued packet.
    {
        // A transport that has N packets queued and only implements receive():
        // the base receiveBatch must loop until the queue drains.
        class QueuedReceiveTransport final : public net::transport::ITransport {
        public:
            core::Expected<void> open() override { return {}; }
            void close() override {}
            core::Expected<core::u32> send(std::span<const core::byte>, const net::Endpoint *) override
            {
                return core::u32{0};
            }
            core::Expected<core::u32> receive(std::span<core::byte> buffer, net::Endpoint *from) override
            {
                ++receiveCalls;
                if (delivered >= total)
                    return core::u32{0};
                ++delivered;
                if (from != nullptr)
                    *from = net::Endpoint::fromOctets(10, 0, 0, 1, static_cast<core::u16>(50000 + delivered));
                buffer[0] = core::byte{0x01};
                return core::u32{1};
            }
            const char *name() const noexcept override { return "QueuedReceiveTransport"; }
            core::u32 total{5};
            core::u32 delivered{0};
            core::u32 receiveCalls{0};
        };

        QueuedReceiveTransport transport;
        std::array<core::byte, 64 * 8> storage{};
        std::array<net::transport::ReceiveSlot, 8> slots{};
        for (core::usize i = 0; i < slots.size(); ++i)
            slots[i].buffer = std::span<core::byte>{storage.data() + i * 8, 8};

        auto received = transport.receiveBatch(std::span<net::transport::ReceiveSlot>{slots.data(), slots.size()});
        check(received.has_value() && received.value() == 5, "the default receiveBatch drains every queued packet");
        check(slots[0].length == 1 && slots[4].length == 1, "each drained slot carries its length");
        check(slots[0].source != slots[1].source, "each slot carries its own sender");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
