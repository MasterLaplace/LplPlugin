/**
 * @file SocketTransport.hpp
 * @brief Standard POSIX UDP socket transport.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_TRANSPORT_SOCKETTRANSPORT_HPP
    #define LPL_NET_TRANSPORT_SOCKETTRANSPORT_HPP

#include <lpl/net/transport/ITransport.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::net::transport {

/**
 * @class SocketTransport
 * @brief POSIX UDP socket-based transport (non-blocking).
 *
 * Binds to a local port on @ref open and uses @c sendto / @c recvfrom
 * for packet exchange.
 */
class SocketTransport final : public ITransport,
                              public core::NonCopyable<SocketTransport>
{
public:
    /**
     * @brief Constructs a socket transport bound to the given port.
     * @param port Local UDP port.
     */
    explicit SocketTransport(core::u16 port);
    ~SocketTransport() override;

    [[nodiscard]] core::Expected<void> open() override;
    void close() override;

    [[nodiscard]] core::Expected<core::u32> send(
        std::span<const core::byte> data,
        const void* address) override;

    [[nodiscard]] core::Expected<core::u32> receive(
        std::span<core::byte> buffer,
        void* fromAddress) override;

    [[nodiscard]] const char* name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net::transport

#endif // LPL_NET_TRANSPORT_SOCKETTRANSPORT_HPP
