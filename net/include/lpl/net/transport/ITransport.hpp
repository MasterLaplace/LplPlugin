/**
 * @file ITransport.hpp
 * @brief Abstract transport layer interface (Strategy pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_TRANSPORT_ITRANSPORT_HPP
#    define LPL_NET_TRANSPORT_ITRANSPORT_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/net/Endpoint.hpp>

#    include <cstddef>
#    include <functional>
#    include <span>

namespace lpl::net::transport {

/**
 * @struct Datagram
 * @brief One outbound packet: its bytes and where they go.
 *
 * Used by @ref ITransport::sendBatch. The spans are borrowed — they must stay
 * alive for the duration of the call and no longer.
 */
struct Datagram {
    std::span<const core::byte> data; ///< Packet bytes.
    const Endpoint *address;          ///< Destination, or nullptr for the default one.
};

/**
 * @struct ReceiveSlot
 * @brief One inbound packet's landing place for @ref ITransport::receiveBatch.
 *
 * The caller owns @c buffer and passes an array of these; the transport fills
 * @c source and @c length in place. Nothing is allocated in the receive path —
 * that matters precisely when the receive path is the bottleneck.
 */
struct ReceiveSlot {
    std::span<core::byte> buffer; ///< In: where to write this packet's bytes.
    Endpoint source;             ///< Out: who sent it.
    core::u32 length{0};         ///< Out: bytes written; 0 if the slot went unused.
};

/**
 * @class ITransport
 * @brief Strategy interface for the transport layer.
 *
 * Concrete implementations:
 *   - @c SocketTransport  — standard BSD/POSIX UDP sockets. The portable path:
 *     Windows, macOS and any Linux without the module use it.
 *   - @c KernelTransport  — zero-copy shared-memory rings via the LPL Linux
 *     kernel module. The fast path, tried first, with SocketTransport as the
 *     fallback when the module is not installed.
 */
class ITransport {
public:
    virtual ~ITransport() = default;

    /**
     * @brief Opens the transport (bind, ioctl, etc.).
     * @return OK on success.
     */
    [[nodiscard]] virtual core::Expected<void> open() = 0;

    /**
     * @brief Closes the transport.
     */
    virtual void close() = 0;

    /**
     * @brief Sends a packet to the given address.
     * @param data    Packet bytes.
     * @param address Destination, or nullptr to use the transport's default
     *                destination (see SocketTransport::setDefaultDest).
     * @return Number of bytes sent, or error.
     */
    [[nodiscard]] virtual core::Expected<core::u32> send(std::span<const core::byte> data, const Endpoint *address) = 0;

    /**
     * @brief Send several packets, giving the transport a chance to batch them.
     *
     * A broadcast is fragments × clients packets, all ready at the same instant.
     * Handed over one at a time that is one syscall each; handed over together,
     * a transport that can do better does better — @c SocketTransport issues a
     * single @c sendmmsg on Linux, @c KernelTransport fills its TX ring and
     * wakes the module's thread with ONE ioctl instead of one per packet.
     *
     * The default implementation simply loops over @ref send, so a transport
     * that has nothing to gain (or a platform with no batching syscall, such as
     * Windows or macOS) inherits correct behaviour and needs no override.
     *
     * @param datagrams The packets to send, in order.
     * @return How many were accepted by the transport, or an error.
     */
    [[nodiscard]] virtual core::Expected<core::u32> sendBatch(std::span<const Datagram> datagrams);

    /**
     * @brief Non-blocking receive.
     * @param buffer  Destination buffer.
     * @param[out] fromAddress Filled with sender address.
     * @return Number of bytes received (0 if nothing available), or error.
     */
    [[nodiscard]] virtual core::Expected<core::u32> receive(std::span<core::byte> buffer, Endpoint *fromAddress) = 0;

    /**
     * @brief Receive several packets, giving the transport a chance to batch.
     *
     * The receive analogue of @ref sendBatch, and the symmetric win: draining a
     * burst of clients one packet at a time is one syscall each; drained
     * together, a transport that can do better does — @c SocketTransport issues
     * a single @c recvmmsg on Linux.
     *
     * Non-blocking: it returns however many packets were waiting, up to
     * @p slots.size(), and 0 when none are. The default implementation loops
     * over @ref receive, so Windows and macOS — which have no recvmmsg — keep
     * working, just one syscall per packet.
     *
     * @param slots Caller-owned landing places; @c source and @c length are
     *              filled for the first N slots, where N is the return value.
     * @return How many packets were received, or an error.
     */
    [[nodiscard]] virtual core::Expected<core::u32> receiveBatch(std::span<ReceiveSlot> slots);

    /**
     * @brief Returns a human-readable name for this transport.
     */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::net::transport

#endif // LPL_NET_TRANSPORT_ITRANSPORT_HPP
