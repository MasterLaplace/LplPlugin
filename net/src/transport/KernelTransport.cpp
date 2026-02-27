/**
 * @file KernelTransport.cpp
 * @brief Kernel module transport implementation (Zero-copy IPC).
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/net/transport/KernelTransport.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include "../../../kernel/lpl_protocol.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <cstring>
#include <cerrno>

namespace lpl::net::transport {

struct KernelTransport::Impl
{
    const char*      devicePath;
    int              fd{-1};
    LplSharedMemory* shm{nullptr};

    explicit Impl(const char* path) : devicePath{path} {}
};

KernelTransport::KernelTransport(const char* devicePath)
    : _impl{std::make_unique<Impl>(devicePath)}
{}

KernelTransport::~KernelTransport()
{
    close();
}

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
        std::string msg = std::string("KernelTransport: open('")
                          + _impl->devicePath + "') failed: "
                          + std::strerror(err);
        core::Log::error(msg);
        return core::makeError(core::ErrorCode::IoError,
                               "Failed to open kernel device");
    }

    size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    size_t len = ((sizeof(LplSharedMemory) + page - 1) / page) * page;
    void* mapped = ::mmap(nullptr, len,
                          PROT_READ | PROT_WRITE, MAP_SHARED, _impl->fd, 0);

    if (mapped == MAP_FAILED)
    {
        int err = errno;
        ::close(_impl->fd);
        _impl->fd = -1;
        core::Log::error("KernelTransport: mmap failed: %s", std::strerror(err));
        return core::makeError(core::ErrorCode::IoError,
                               "Failed to mmap kernel device");
    }

    _impl->shm = static_cast<LplSharedMemory*>(mapped);

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

core::Expected<core::u32> KernelTransport::send(
    std::span<const core::byte> data,
    const void* /*address*/)
{
    if (!_impl->shm)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    if (data.size() > LPL_MAX_PACKET_SIZE)
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Packet too large");
    }

    uint32_t head = smp_load_acquire(&_impl->shm->tx.idx.head);
    uint32_t tail = smp_load_acquire(&_impl->shm->tx.idx.tail);
    uint32_t next = head + 1;

    if ((next - tail) > LPL_RING_SLOTS)
    {
        return core::makeError(core::ErrorCode::IoError, "TX ring buffer full");
    }

    LplTxPacket* slot = &_impl->shm->tx.packets[head & LPL_RING_MASK];

    // Address handling would go here, mapping ITransport socket address to dst_ip/port
    slot->dst_ip   = 0;
    slot->dst_port = 0;
    slot->length   = static_cast<uint16_t>(data.size());
    std::memcpy(slot->data, data.data(), data.size());

    smp_store_release(&_impl->shm->tx.idx.head, next);

    // Kick the kthread to wake up and send UDP
    ::ioctl(_impl->fd, LPL_IOCTL_KICK_TX);

    return static_cast<core::u32>(data.size());
}

core::Expected<core::u32> KernelTransport::receive(
    std::span<core::byte> buffer,
    void* /*fromAddress*/)
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

    LplRxPacket* slot = &_impl->shm->rx.packets[tail & LPL_RING_MASK];

    if (slot->length > buffer.size())
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Buffer too small");
    }

    std::memcpy(buffer.data(), slot->data, slot->length);

    // Address handling would go here

    smp_store_release(&_impl->shm->rx.idx.tail, tail + 1);

    return static_cast<core::u32>(slot->length);
}

const char* KernelTransport::name() const noexcept
{
    return "KernelTransport";
}

} // namespace lpl::net::transport
