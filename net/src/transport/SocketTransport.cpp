/**
 * @file SocketTransport.cpp
 * @brief POSIX UDP socket transport implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/transport/SocketTransport.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace lpl::net::transport {

namespace {

/// Endpoint (host byte order) → sockaddr_in (network byte order). This file is
/// the single translation point between the engine's platform-neutral Endpoint
/// and the BSD sockets representation.
[[nodiscard]] sockaddr_in toSockaddr(const Endpoint &endpoint) noexcept
{
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(endpoint.port());
    addr.sin_addr.s_addr = htonl(endpoint.address());
    return addr;
}

[[nodiscard]] Endpoint fromSockaddr(const sockaddr_in &addr) noexcept
{
    return Endpoint{ntohl(addr.sin_addr.s_addr), ntohs(addr.sin_port)};
}

} // namespace

struct SocketTransport::Impl {
    core::u16 port;
    int fd{-1};
    Endpoint defaultDest{}; ///< Used when send() is called with nullptr address.

    explicit Impl(core::u16 p) : port{p} {}
};

SocketTransport::SocketTransport(core::u16 port) : _impl{std::make_unique<Impl>(port)} {}

SocketTransport::~SocketTransport() { close(); }

core::Expected<void> SocketTransport::open()
{
    _impl->fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (_impl->fd < 0)
    {
        return core::makeError(core::ErrorCode::IoError, "socket() failed");
    }

    int flags = ::fcntl(_impl->fd, F_GETFL, 0);
    ::fcntl(_impl->fd, F_SETFL, flags | O_NONBLOCK);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(_impl->port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(_impl->fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        ::close(_impl->fd);
        _impl->fd = -1;
        return core::makeError(core::ErrorCode::IoError, "bind() failed");
    }

    core::Log::info("SocketTransport: bound to port");
    return {};
}

void SocketTransport::close()
{
    if (_impl->fd >= 0)
    {
        ::close(_impl->fd);
        _impl->fd = -1;
    }
}

core::Expected<core::u32> SocketTransport::send(std::span<const core::byte> data, const Endpoint *address)
{
    if (_impl->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Socket not open");
    }

    // Fall back to the pre-set default destination when no address is provided
    // (e.g. InputSendSystem calling sendInputs(..., nullptr, ...)).
    const Endpoint &target = (address != nullptr && address->valid()) ? *address : _impl->defaultDest;
    const sockaddr_in addr = toSockaddr(target);

    const auto sent =
        ::sendto(_impl->fd, data.data(), data.size(), 0, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));

    if (sent < 0)
    {
        return core::makeError(core::ErrorCode::IoError, "sendto() failed");
    }

    return static_cast<core::u32>(sent);
}

core::Expected<core::u32> SocketTransport::sendBatch(std::span<const Datagram> datagrams)
{
    if (_impl->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Socket not open");
    }

    if (datagrams.empty())
    {
        return core::u32{0};
    }

#ifdef __linux__
    // sendmmsg hands the whole burst to the kernel in ONE syscall. A broadcast
    // is fragments × clients packets, so this turns N syscalls into 1.
    //
    // Linux only, deliberately: Windows and macOS have no equivalent, and this
    // transport is precisely the portable path for them. They fall through to
    // the base implementation's loop, which is correct, just not batched.
    constexpr core::usize kMaxBurst = 64;

    core::u32 totalSent = 0;
    core::usize offset = 0;

    while (offset < datagrams.size())
    {
        const core::usize burst = std::min(kMaxBurst, datagrams.size() - offset);

        // These three arrays must outlive the syscall: mmsghdr points into them.
        sockaddr_in addresses[kMaxBurst]{};
        iovec iovecs[kMaxBurst]{};
        mmsghdr messages[kMaxBurst]{};

        for (core::usize i = 0; i < burst; ++i)
        {
            const auto &datagram = datagrams[offset + i];
            const Endpoint &target = (datagram.address != nullptr && datagram.address->valid()) ? *datagram.address :
                                                                                                  _impl->defaultDest;
            addresses[i] = toSockaddr(target);

            iovecs[i].iov_base = const_cast<core::byte *>(datagram.data.data());
            iovecs[i].iov_len = datagram.data.size();

            messages[i].msg_hdr.msg_name = &addresses[i];
            messages[i].msg_hdr.msg_namelen = sizeof(sockaddr_in);
            messages[i].msg_hdr.msg_iov = &iovecs[i];
            messages[i].msg_hdr.msg_iovlen = 1;
        }

        const int sent = ::sendmmsg(_impl->fd, messages, static_cast<unsigned int>(burst), 0);
        if (sent < 0)
        {
            // Report what did get out rather than losing the count entirely.
            return totalSent > 0 ? core::Expected<core::u32>{totalSent}
                                 : core::makeError(core::ErrorCode::IoError, "sendmmsg() failed");
        }

        totalSent += static_cast<core::u32>(sent);

        // A short send means the socket buffer filled: stop, do not spin.
        if (static_cast<core::usize>(sent) < burst)
            break;

        offset += burst;
    }

    return totalSent;
#else
    return ITransport::sendBatch(datagrams);
#endif
}

core::Expected<core::u32> SocketTransport::receive(std::span<core::byte> buffer, Endpoint *fromAddress)
{
    if (_impl->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Socket not open");
    }

    sockaddr_in addr{};
    socklen_t addrLen = sizeof(addr);

    const auto received =
        ::recvfrom(_impl->fd, buffer.data(), buffer.size(), 0, reinterpret_cast<sockaddr *>(&addr), &addrLen);

    if (received < 0)
    {
        if (errno == EWOULDBLOCK || errno == EAGAIN)
        {
            return core::u32{0};
        }
        return core::makeError(core::ErrorCode::IoError, "recvfrom() failed");
    }

    if (fromAddress != nullptr)
    {
        *fromAddress = fromSockaddr(addr);
    }

    return static_cast<core::u32>(received);
}

core::Expected<core::u32> SocketTransport::receiveBatch(std::span<ReceiveSlot> slots)
{
    if (_impl->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Socket not open");
    }

    if (slots.empty())
    {
        return core::u32{0};
    }

#ifdef __linux__
    // recvmmsg drains up to kMaxBurst waiting packets in ONE syscall — the
    // receive-side symmetry of sendmmsg. Linux only, on purpose: the base
    // implementation's loop covers Windows and macOS, which lack recvmmsg.
    constexpr core::usize kMaxBurst = 64;
    const core::usize burst = std::min(kMaxBurst, slots.size());

    sockaddr_in addresses[kMaxBurst]{};
    iovec iovecs[kMaxBurst]{};
    mmsghdr messages[kMaxBurst]{};

    for (core::usize i = 0; i < burst; ++i)
    {
        iovecs[i].iov_base = slots[i].buffer.data();
        iovecs[i].iov_len = slots[i].buffer.size();

        messages[i].msg_hdr.msg_name = &addresses[i];
        messages[i].msg_hdr.msg_namelen = sizeof(sockaddr_in);
        messages[i].msg_hdr.msg_iov = &iovecs[i];
        messages[i].msg_hdr.msg_iovlen = 1;
    }

    // Non-blocking (no MSG_WAITFORONE): returns what is already queued, or -1 /
    // EAGAIN when nothing is.
    const int count = ::recvmmsg(_impl->fd, messages, static_cast<unsigned int>(burst), 0, nullptr);
    if (count < 0)
    {
        if (errno == EWOULDBLOCK || errno == EAGAIN)
            return core::u32{0};
        return core::makeError(core::ErrorCode::IoError, "recvmmsg() failed");
    }

    for (int i = 0; i < count; ++i)
    {
        slots[i].length = messages[i].msg_len;
        slots[i].source = fromSockaddr(addresses[i]);
    }

    return static_cast<core::u32>(count);
#else
    return ITransport::receiveBatch(slots);
#endif
}

const char *SocketTransport::name() const noexcept { return "SocketTransport"; }

void SocketTransport::setDefaultDest(const Endpoint &endpoint) noexcept { _impl->defaultDest = endpoint; }

} // namespace lpl::net::transport
