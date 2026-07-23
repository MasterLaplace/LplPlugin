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
#    define LPL_NET_TRANSPORT_SOCKETTRANSPORT_HPP

#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>

namespace lpl::net::transport {

/**
 * @class SocketTransport
 * @brief POSIX UDP socket-based transport (non-blocking).
 *
 * Binds to a local port on @ref open and uses @c sendto / @c recvfrom
 * for packet exchange.
 */
class SocketTransport final : public ITransport, public core::NonCopyable<SocketTransport> {
public:
    /**
     * @brief Constructs a socket transport bound to the given port.
     * @param port Local UDP port.
     */
    explicit SocketTransport(core::u16 port);
    ~SocketTransport() override;

    [[nodiscard]] core::Expected<void> open() override;
    void close() override;

    [[nodiscard]] core::Expected<core::u32> send(std::span<const core::byte> data, const Endpoint *address) override;

    /**
     * @brief Batched send: one @c sendmmsg syscall on Linux.
     *
     * On any other platform (Windows, macOS) this falls back to the base
     * implementation's loop — there is no portable equivalent, and this
     * transport is exactly the path those platforms rely on.
     */
    [[nodiscard]] core::Expected<core::u32> sendBatch(std::span<const Datagram> datagrams) override;

    /**
     * @brief Batched receive: one @c recvmmsg syscall on Linux.
     *
     * Falls back to the base loop on Windows/macOS, which have no recvmmsg.
     */
    [[nodiscard]] core::Expected<core::u32> receiveBatch(std::span<ReceiveSlot> slots) override;

    [[nodiscard]] core::Expected<core::u32> receive(std::span<core::byte> buffer, Endpoint *fromAddress) override;

    [[nodiscard]] const char *name() const noexcept override;

    /**
     * @brief Sets a default destination address used when @c send() is called
     *        with a @c nullptr address.
     *
     * Called once after the initial handshake so that InputSendSystem,
     * BroadcastSystem, etc. do not need to carry a copy of the server address.
     *
     * @param endpoint The target.
     */
    void setDefaultDest(const Endpoint &endpoint) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net::transport

#endif // LPL_NET_TRANSPORT_SOCKETTRANSPORT_HPP
