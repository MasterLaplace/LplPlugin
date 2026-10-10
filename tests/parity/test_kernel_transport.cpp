/**
 * @file test_kernel_transport.cpp
 * @brief The process side of the module's rings, against a file mapped as the module maps them.
 *
 * KernelTransport maps /dev/lpl0; here it maps a plain file of the same size instead, and the test
 * plays the module: it writes the header, reads what the transport publishes on the TX ring, and
 * publishes RX slots for the transport to read. A kick is an ioctl, which a plain file refuses, but
 * the transport counts it all the same. The module itself is built and run on a Linux kernel
 * elsewhere; this runs wherever the tests run.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-10-10
 * @copyright MIT License
 */

#include <lpl/core/Log.hpp>
#include <lpl/net/Endpoint.hpp>
#include <lpl/net/transport/KernelTransport.hpp>

#include "../../kernel/lpl_protocol.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;

void check(const char *label, bool ok)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok)
        ++failures;
}

/** @brief A plain file mapped like /dev/lpl0, with the test on the module's side of it. */
class FakeModule final {
public:
    FakeModule()
    {
        const char *directory = std::getenv("TMPDIR");
        std::string pattern = std::string(directory != nullptr ? directory : "/tmp") + "/lpl-kernel-transport-XXXXXX";

        _path.assign(pattern.begin(), pattern.end());
        _path.push_back('\0');
        _fd = ::mkstemp(_path.data());
        if (_fd >= 0 && ::ftruncate(_fd, static_cast<off_t>(mappedLength())) == 0)
        {
            void *mapped = ::mmap(nullptr, mappedLength(), PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
            _shm = mapped == MAP_FAILED ? nullptr : static_cast<LplSharedMemory *>(mapped);
        }
    }

    ~FakeModule()
    {
        if (_shm != nullptr)
            ::munmap(_shm, mappedLength());
        if (_fd >= 0)
        {
            ::close(_fd);
            ::unlink(_path.data());
        }
    }

    FakeModule(const FakeModule &) = delete;
    FakeModule &operator=(const FakeModule &) = delete;

    [[nodiscard]] bool ready() const noexcept { return _shm != nullptr; }
    [[nodiscard]] const char *path() const noexcept { return _path.data(); }
    [[nodiscard]] LplSharedMemory &shm() noexcept { return *_shm; }

    void writeHeader(uint32_t version) noexcept
    {
        _shm->header.magic = LPL_MAGIC;
        _shm->header.version = version;
        _shm->header.slots = LPL_RING_SLOTS;
        _shm->header.size = sizeof(LplSharedMemory);
    }

    void publishRx(uint32_t index, uint32_t sourceIp, uint16_t sourcePort, const std::string &payload) noexcept
    {
        LplRxPacket &slot = _shm->rx.packets[index & LPL_RING_MASK];

        slot.src_ip = sourceIp;
        slot.src_port = sourcePort;
        slot.length = static_cast<uint16_t>(payload.size());
        std::memcpy(slot.data, payload.data(), payload.size());
        __atomic_store_n(&_shm->rx.writer.write_index, index + 1u, __ATOMIC_RELEASE);
    }

private:
    [[nodiscard]] static std::size_t mappedLength() noexcept
    {
        const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
        return (sizeof(LplSharedMemory) + page - 1u) / page * page;
    }

    std::vector<char> _path;
    int _fd = -1;
    LplSharedMemory *_shm = nullptr;
};

[[nodiscard]] std::span<const lpl::core::byte> bytesOf(const std::string &text) noexcept
{
    return {reinterpret_cast<const lpl::core::byte *>(text.data()), text.size()};
}

[[nodiscard]] bool txSlotCarries(const LplSharedMemory &shm, uint32_t index, const std::string &payload) noexcept
{
    const LplTxPacket &slot = shm.tx.packets[index & LPL_RING_MASK];
    return slot.length == payload.size() && std::memcmp(slot.data, payload.data(), payload.size()) == 0;
}

void refusesAnotherProtocol()
{
    FakeModule module;
    lpl::net::transport::KernelTransport transport(module.path());

    check("a mapping without the protocol header is refused", module.ready() && !transport.open().has_value());
    module.writeHeader(LPL_PROTOCOL_VERSION - 1u);
    check("a mapping of another protocol version is refused", !transport.open().has_value());
    module.writeHeader(LPL_PROTOCOL_VERSION);
    check("a mapping of this version opens", transport.open().has_value());
}

void publishesTransmitsAndKicksOnlyASleeper()
{
    FakeModule module;
    module.writeHeader(LPL_PROTOCOL_VERSION);
    lpl::net::transport::KernelTransport transport(module.path());
    const lpl::net::Endpoint peer(0x7F000001u, 9999u);
    const std::string first = "first";
    const std::string second = "second datagram";

    check("the transport opens", transport.open().has_value());
    check("three sends are accepted", transport.send(bytesOf(first), &peer).has_value() &&
                                          transport.send(bytesOf(second), &peer).has_value() &&
                                          transport.send(bytesOf(first), &peer).has_value());
    check("and published: the TX write index is 3",
          __atomic_load_n(&module.shm().tx.writer.write_index, __ATOMIC_ACQUIRE) == 3u);
    check("each slot holds its bytes", txSlotCarries(module.shm(), 0u, first) &&
                                           txSlotCarries(module.shm(), 1u, second) &&
                                           txSlotCarries(module.shm(), 2u, first));
    check("each slot names its destination in host byte order, which the module converts",
          module.shm().tx.packets[0].dst_ip == 0x7F000001u && module.shm().tx.packets[0].dst_port == 9999u);
    check("an awake sender is not kicked", transport.kickCount() == 0u);

    __atomic_store_n(&module.shm().tx.wake.sleeping, 1u, __ATOMIC_RELEASE);
    const std::array datagrams{
        lpl::net::transport::Datagram{bytesOf(first),  &peer},
        lpl::net::transport::Datagram{bytesOf(second), &peer}
    };
    const auto accepted = transport.sendBatch(datagrams);
    check("a batch of two is accepted and published once",
          accepted.has_value() && *accepted == 2u &&
              __atomic_load_n(&module.shm().tx.writer.write_index, __ATOMIC_ACQUIRE) == 5u);
    check("and a sleeping sender is kicked once for the batch", transport.kickCount() == 1u);
}

void waitsForTheModuleWhenFull()
{
    FakeModule module;
    module.writeHeader(LPL_PROTOCOL_VERSION);
    lpl::net::transport::KernelTransport transport(module.path());
    const lpl::net::Endpoint peer(0x7F000001u, 9999u);
    const std::string payload = "x";
    std::vector<lpl::net::transport::Datagram> datagrams(LPL_RING_SLOTS + 1u,
                                                         lpl::net::transport::Datagram{bytesOf(payload), &peer});

    check("the transport opens", transport.open().has_value());
    const auto accepted = transport.sendBatch(datagrams);
    check("a batch one slot larger than the ring fills it", accepted.has_value() && *accepted == LPL_RING_SLOTS);
    check("a full ring refuses the next send", !transport.send(bytesOf(payload), &peer).has_value());

    __atomic_store_n(&module.shm().tx.reader.read_index, 10u, __ATOMIC_RELEASE);
    check("once the module has sent ten, a send goes through", transport.send(bytesOf(payload), &peer).has_value());
}

void receivesWhatTheModulePublished()
{
    FakeModule module;
    module.writeHeader(LPL_PROTOCOL_VERSION);
    lpl::net::transport::KernelTransport transport(module.path());
    std::array<lpl::core::byte, LPL_MAX_PACKET_SIZE> buffer{};
    std::array<lpl::core::byte, 2> small{};
    lpl::net::Endpoint from;

    check("the transport opens", transport.open().has_value());
    auto received = transport.receive(buffer, &from);
    check("an empty ring gives nothing", received.has_value() && *received == 0u);

    module.publishRx(0u, 0x0A000002u, 4242u, "alpha");
    module.publishRx(1u, 0x0A000003u, 4243u, "bravo charlie");
    received = transport.receive(buffer, &from);
    check("the first packet comes back whole",
          received.has_value() && *received == 5u && std::memcmp(buffer.data(), "alpha", 5u) == 0);
    check("with its sender", from.address() == 0x0A000002u && from.port() == 4242u);

    received = transport.receive(small, &from);
    check("a buffer too small is refused", !received.has_value());
    check("and leaves the packet in the ring",
          __atomic_load_n(&module.shm().rx.reader.read_index, __ATOMIC_ACQUIRE) == 1u);

    received = transport.receive(buffer, &from);
    check("the second packet comes back whole",
          received.has_value() && *received == 13u && std::memcmp(buffer.data(), "bravo charlie", 13u) == 0);
    check("with its own sender", from.address() == 0x0A000003u && from.port() == 4243u);
    check("and the module sees both read", __atomic_load_n(&module.shm().rx.reader.read_index, __ATOMIC_ACQUIRE) == 2u);
}

void receivesInOneBatch()
{
    FakeModule module;
    module.writeHeader(LPL_PROTOCOL_VERSION);
    lpl::net::transport::KernelTransport transport(module.path());
    std::array<std::array<lpl::core::byte, LPL_MAX_PACKET_SIZE>, 8> buffers{};
    std::array<lpl::net::transport::ReceiveSlot, 8> slots{};

    for (std::size_t i = 0u; i < slots.size(); ++i)
        slots[i].buffer = buffers[i];
    check("the transport opens", transport.open().has_value());
    for (uint32_t i = 0u; i < 5u; ++i)
        module.publishRx(i, 0x0A000000u + i, static_cast<uint16_t>(5000u + i), std::string(1u + i, 'a'));

    auto received = transport.receiveBatch(std::span(slots).first(3u));
    check("a batch of three takes the three oldest, in order", received.has_value() && *received == 3u &&
                                                                   slots[0].length == 1u && slots[2].length == 3u &&
                                                                   slots[2].source.port() == 5002u);
    check("and frees their slots at once", __atomic_load_n(&module.shm().rx.reader.read_index, __ATOMIC_ACQUIRE) == 3u);

    received = transport.receiveBatch(slots);
    check("a batch of eight takes the two left",
          received.has_value() && *received == 2u && slots[1].length == 5u && slots[1].source.address() == 0x0A000004u);
    received = transport.receiveBatch(slots);
    check("an empty ring gives a batch of none", received.has_value() && *received == 0u);
}

} // namespace

int main()
{
    lpl::core::Log::info("=== KernelTransport against a mapped file ===");

    refusesAnotherProtocol();
    publishesTransmitsAndKicksOnlyASleeper();
    waitsForTheModuleWhenFull();
    receivesWhatTheModulePublished();
    receivesInOneBatch();

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
