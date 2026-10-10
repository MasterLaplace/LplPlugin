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

#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lpl::net::transport {

struct KernelTransport::Impl {
    const char *devicePath;
    int fd{-1};
    LplSharedMemory *shm{nullptr};
    std::size_t mappedLength{0};     /**< Length passed to mmap, for munmap. */
    core::u32 txWriteIndex{0};       /**< Next TX slot this process fills; published to shm->tx.writer. */
    core::u32 txCachedReadIndex{0};  /**< The module's TX read index, as last read. */
    core::u32 rxReadIndex{0};        /**< Next RX slot this process reads; published to shm->rx.reader. */
    core::u32 rxCachedWriteIndex{0}; /**< The module's RX write index, as last read. */
    core::u64 kicks{0};              /**< System calls that woke the module's sender. */

    explicit Impl(const char *path) : devicePath{path} {}
};

namespace {

[[nodiscard]] bool speaksThisProtocol(const LplSharedMemory &shm) noexcept
{
    return shm.header.magic == LPL_MAGIC && shm.header.version == LPL_PROTOCOL_VERSION &&
           shm.header.slots == LPL_RING_SLOTS && shm.header.size == sizeof(LplSharedMemory);
}

} // namespace

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
    _impl->mappedLength = len;

    if (!speaksThisProtocol(*_impl->shm))
    {
        core::Log::error(std::string("KernelTransport: the module's mapping is protocol ") +
                         std::to_string(_impl->shm->header.version) + ", this build speaks " +
                         std::to_string(LPL_PROTOCOL_VERSION));
        close();
        return core::makeError(core::ErrorCode::InvalidState, "Kernel module speaks another protocol");
    }

    _impl->txWriteIndex = smp_load_acquire(&_impl->shm->tx.writer.write_index);
    _impl->txCachedReadIndex = smp_load_acquire(&_impl->shm->tx.reader.read_index);
    _impl->rxReadIndex = smp_load_acquire(&_impl->shm->rx.reader.read_index);
    _impl->rxCachedWriteIndex = _impl->rxReadIndex;

    core::Log::info("KernelTransport: opened device and mmap'd shared memory");
    return {};
}

void KernelTransport::close()
{
    if (_impl->shm && _impl->shm != MAP_FAILED)
    {
        ::munmap(_impl->shm, _impl->mappedLength);
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

    if (_impl->txWriteIndex - _impl->txCachedReadIndex >= LPL_RING_SLOTS)
    {
        _impl->txCachedReadIndex = smp_load_acquire(&_impl->shm->tx.reader.read_index);
        if (_impl->txWriteIndex - _impl->txCachedReadIndex >= LPL_RING_SLOTS)
            return false;
    }

    LplTxPacket *slot = &_impl->shm->tx.packets[_impl->txWriteIndex & LPL_RING_MASK];

    if (address != nullptr && address->valid())
    {
        slot->dst_ip = address->address();
        slot->dst_port = address->port();
    }
    else
    {
        slot->dst_ip = 0;
        slot->dst_port = 0;
    }

    slot->length = static_cast<uint16_t>(data.size());
    std::memcpy(slot->data, data.data(), data.size());

    ++_impl->txWriteIndex;
    return true;
}

void KernelTransport::publishTx() noexcept
{
    smp_store_release(&_impl->shm->tx.writer.write_index, _impl->txWriteIndex);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (__atomic_load_n(&_impl->shm->tx.wake.sleeping, __ATOMIC_RELAXED) != 0u)
    {
        ::ioctl(_impl->fd, LPL_IOCTL_KICK_TX);
        ++_impl->kicks;
    }
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

    publishTx();

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
        publishTx();
    }

    return accepted;
}

core::Expected<core::u32> KernelTransport::receive(std::span<core::byte> buffer, Endpoint *fromAddress)
{
    if (!_impl->shm)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    if (_impl->rxCachedWriteIndex == _impl->rxReadIndex)
    {
        _impl->rxCachedWriteIndex = smp_load_acquire(&_impl->shm->rx.writer.write_index);
        if (_impl->rxCachedWriteIndex == _impl->rxReadIndex)
            return core::u32{0};
    }

    const LplRxPacket *slot = &_impl->shm->rx.packets[_impl->rxReadIndex & LPL_RING_MASK];
    const core::u16 length = slot->length;

    if (length > buffer.size())
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Buffer too small");
    }

    std::memcpy(buffer.data(), slot->data, length);
    if (fromAddress != nullptr)
        *fromAddress = Endpoint(slot->src_ip, slot->src_port);
    ++_impl->rxReadIndex;
    smp_store_release(&_impl->shm->rx.reader.read_index, _impl->rxReadIndex);

    return static_cast<core::u32>(length);
}

core::Expected<core::u32> KernelTransport::receiveBatch(std::span<ReceiveSlot> slots)
{
    if (!_impl->shm)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    core::u32 ready = _impl->rxCachedWriteIndex - _impl->rxReadIndex;
    if (ready < slots.size())
    {
        _impl->rxCachedWriteIndex = smp_load_acquire(&_impl->shm->rx.writer.write_index);
        ready = _impl->rxCachedWriteIndex - _impl->rxReadIndex;
    }

    const core::u32 wanted = slots.size() < ready ? static_cast<core::u32>(slots.size()) : ready;
    core::u32 received = 0;

    for (; received < wanted; ++received)
    {
        const LplRxPacket *packet = &_impl->shm->rx.packets[(_impl->rxReadIndex + received) & LPL_RING_MASK];
        const core::u16 length = packet->length;
        ReceiveSlot &slot = slots[received];

        if (length > slot.buffer.size())
        {
            if (received == 0)
                return core::makeError(core::ErrorCode::InvalidArgument, "Buffer too small");
            break;
        }
        std::memcpy(slot.buffer.data(), packet->data, length);
        slot.source = Endpoint(packet->src_ip, packet->src_port);
        slot.length = length;
    }

    if (received != 0)
    {
        _impl->rxReadIndex += received;
        smp_store_release(&_impl->shm->rx.reader.read_index, _impl->rxReadIndex);
    }
    return received;
}

const char *KernelTransport::name() const noexcept { return "KernelTransport"; }

core::u64 KernelTransport::kickCount() const noexcept { return _impl->kicks; }

} // namespace lpl::net::transport
