// /////////////////////////////////////////////////////////////////////////////
/// @file SocketTransport.cpp
/// @brief POSIX UDP socket transport implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/transport/SocketTransport.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>

namespace lpl::net::transport {

struct SocketTransport::Impl
{
    core::u16 port;
    int       fd{-1};

    explicit Impl(core::u16 p) : port{p} {}
};

SocketTransport::SocketTransport(core::u16 port)
    : impl_{std::make_unique<Impl>(port)}
{}

SocketTransport::~SocketTransport()
{
    close();
}

core::Expected<void> SocketTransport::open()
{
    impl_->fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (impl_->fd < 0)
    {
        return core::makeError(core::ErrorCode::IoError, "socket() failed");
    }

    int flags = ::fcntl(impl_->fd, F_GETFL, 0);
    ::fcntl(impl_->fd, F_SETFL, flags | O_NONBLOCK);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(impl_->port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(impl_->fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        ::close(impl_->fd);
        impl_->fd = -1;
        return core::makeError(core::ErrorCode::IoError, "bind() failed");
    }

    core::Log::info("SocketTransport: bound to port");
    return {};
}

void SocketTransport::close()
{
    if (impl_->fd >= 0)
    {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

core::Expected<core::u32> SocketTransport::send(
    std::span<const core::byte> data,
    const void* address)
{
    if (impl_->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Socket not open");
    }

    const auto* addr = static_cast<const sockaddr_in*>(address);
    const auto sent = ::sendto(impl_->fd,
                               data.data(),
                               data.size(),
                               0,
                               reinterpret_cast<const sockaddr*>(addr),
                               sizeof(sockaddr_in));

    if (sent < 0)
    {
        return core::makeError(core::ErrorCode::IoError, "sendto() failed");
    }

    return static_cast<core::u32>(sent);
}

core::Expected<core::u32> SocketTransport::receive(
    std::span<core::byte> buffer,
    void* fromAddress)
{
    if (impl_->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Socket not open");
    }

    auto* addr = static_cast<sockaddr_in*>(fromAddress);
    socklen_t addrLen = sizeof(sockaddr_in);

    const auto received = ::recvfrom(impl_->fd,
                                     buffer.data(),
                                     buffer.size(),
                                     0,
                                     reinterpret_cast<sockaddr*>(addr),
                                     &addrLen);

    if (received < 0)
    {
        if (errno == EWOULDBLOCK || errno == EAGAIN)
        {
            return core::u32{0};
        }
        return core::makeError(core::ErrorCode::IoError, "recvfrom() failed");
    }

    return static_cast<core::u32>(received);
}

const char* SocketTransport::name() const noexcept
{
    return "SocketTransport";
}

} // namespace lpl::net::transport
