/**
 * @file KernelTransport.cpp
 * @brief Kernel module transport implementation (Zero-copy IPC).
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/transport/KernelTransport.hpp>

#include "../../../kernel/lpl_protocol.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lpl::net::transport {

struct KernelTransport::Impl {
    const char *devicePath;
    int fd{-1};
    LplSharedMemory *shm{nullptr};

    explicit Impl(const char *path) : devicePath{path} {}
};

KernelTransport::KernelTransport(const char *devicePath) : _impl{std::make_unique<Impl>(devicePath)} {}

KernelTransport::~KernelTransport() { close(); }

core::Expected<void> KernelTransport::open()
{
    if (_impl->fd >= 0)
    {
        return {}; // already open
    }

    _impl->fd = ::open(_impl->devicePath, O_RDWR);
    if (_impl->fd < 0)
    {
        int err = errno;
        std::string msg =
            std::string("KernelTransport: open('") + _impl->devicePath + "') failed: " + std::strerror(err);
        core::Log::error(msg);
        return core::makeError(core::ErrorCode::IoError, "Failed to open kernel device");
    }

    size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    size_t len = ((sizeof(LplSharedMemory) + page - 1) / page) * page;
    void *mapped = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, _impl->fd, 0);

    if (mapped == MAP_FAILED)
    {
        int err = errno;
        ::close(_impl->fd);
        _impl->fd = -1;
        core::Log::error("KernelTransport: mmap failed: %s", std::strerror(err));
        return core::makeError(core::ErrorCode::IoError, "Failed to mmap kernel device");
    }

    _impl->shm = static_cast<LplSharedMemory *>(mapped);

    core::Log::info("KernelTransport: opened device and mmap'd shared memory");
    return {};
}

void KernelTransport::close()
{
    if (_impl->shm && _impl->shm != MAP_FAILED)
    {
        ::munmap(_impl->shm, sizeof(LplSharedMemory));
        _impl->shm = nullptr;
    }

    if (_impl->fd >= 0)
    {
        ::close(_impl->fd);
        _impl->fd = -1;
    }
}

bool KernelTransport::pushSlot(std::span<const core::byte> data, const Endpoint *address) noexcept
{
    if (data.size() > LPL_MAX_PACKET_SIZE)
        return false;

    const uint32_t head = smp_load_acquire(&_impl->shm->tx.idx.head);
    const uint32_t tail = smp_load_acquire(&_impl->shm->tx.idx.tail);
    const uint32_t next = head + 1;

    if ((next - tail) > LPL_RING_SLOTS)
        return false; // ring full

    LplTxPacket *slot = &_impl->shm->tx.packets[head & LPL_RING_MASK];

    // The module expects network byte order, exactly as the legacy driver path
    // wrote it (htonl/htons on a host-order address). Leaving these at 0, as
    // this did before, addressed every packet to 0.0.0.0:0.
    if (address != nullptr && address->valid())
    {
        slot->dst_ip = htonl(address->address());
        slot->dst_port = htons(address->port());
    }
    else
    {
        slot->dst_ip = 0;
        slot->dst_port = 0;
    }

    slot->length = static_cast<uint16_t>(data.size());
    std::memcpy(slot->data, data.data(), data.size());

    smp_store_release(&_impl->shm->tx.idx.head, next);
    return true;
}

core::Expected<core::u32> KernelTransport::send(std::span<const core::byte> data, const Endpoint *address)
{
    if (!_impl->shm)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    if (data.size() > LPL_MAX_PACKET_SIZE)
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Packet too large");
    }

    if (!pushSlot(data, address))
    {
        return core::makeError(core::ErrorCode::IoError, "TX ring buffer full");
    }

    // Kick the kthread to wake up and send UDP
    ::ioctl(_impl->fd, LPL_IOCTL_KICK_TX);

    return static_cast<core::u32>(data.size());
}

core::Expected<core::u32> KernelTransport::sendBatch(std::span<const Datagram> datagrams)
{
    if (!_impl->shm)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    if (datagrams.empty())
    {
        return core::u32{0};
    }

    // The whole point of the ring: fill every slot first, then wake the module's
    // kthread ONCE. Its ioctl is a plain wake_up_interruptible and the thread
    // drains the entire ring, so one kick flushes N packets — this is the "kick"
    // the book describes. Kicking per packet, as this used to (and as the legacy
    // driver path did too), pays a syscall per datagram and throws that away.
    core::u32 accepted = 0;
    for (const auto &datagram : datagrams)
    {
        if (!pushSlot(datagram.data, datagram.address))
            break; // ring full or packet oversized: flush what we have
        ++accepted;
    }

    if (accepted > 0)
    {
        ::ioctl(_impl->fd, LPL_IOCTL_KICK_TX);
    }

    return accepted;
}

core::Expected<core::u32> KernelTransport::receive(std::span<core::byte> buffer, Endpoint * /*fromAddress*/)
{
    if (!_impl->shm)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    uint32_t head = smp_load_acquire(&_impl->shm->rx.idx.head);
    uint32_t tail = smp_load_acquire(&_impl->shm->rx.idx.tail);

    if (tail == head)
    {
        return core::u32{0}; // Empty
    }

    LplRxPacket *slot = &_impl->shm->rx.packets[tail & LPL_RING_MASK];

    if (slot->length > buffer.size())
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Buffer too small");
    }

    std::memcpy(buffer.data(), slot->data, slot->length);

    // Address handling would go here

    smp_store_release(&_impl->shm->rx.idx.tail, tail + 1);

    return static_cast<core::u32>(slot->length);
}

const char *KernelTransport::name() const noexcept { return "KernelTransport"; }

} // namespace lpl::net::transport
