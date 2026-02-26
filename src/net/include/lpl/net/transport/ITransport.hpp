// /////////////////////////////////////////////////////////////////////////////
/// @file ITransport.hpp
/// @brief Abstract transport layer interface (Strategy pattern).
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

#include <cstddef>
#include <functional>
#include <span>

namespace lpl::net::transport {

// /////////////////////////////////////////////////////////////////////////////
/// @class ITransport
/// @brief Strategy interface for the transport layer.
///
/// Concrete implementations:
///   - @c SocketTransport  — standard BSD/POSIX UDP sockets.
///   - @c KernelTransport  — zero-copy via the LPL kernel module.
// /////////////////////////////////////////////////////////////////////////////
class ITransport
{
public:
    virtual ~ITransport() = default;

    /// @brief Opens the transport (bind, ioctl, etc.).
    /// @return OK on success.
    [[nodiscard]] virtual core::Expected<void> open() = 0;

    /// @brief Closes the transport.
    virtual void close() = 0;

    /// @brief Sends a packet to the given address.
    /// @param data    Packet bytes.
    /// @param address Opaque address (cast to sockaddr_in, etc.).
    /// @return Number of bytes sent, or error.
    [[nodiscard]] virtual core::Expected<core::u32> send(
        std::span<const core::byte> data,
        const void* address) = 0;

    /// @brief Non-blocking receive.
    /// @param buffer  Destination buffer.
    /// @param[out] fromAddress Filled with sender address.
    /// @return Number of bytes received (0 if nothing available), or error.
    [[nodiscard]] virtual core::Expected<core::u32> receive(
        std::span<core::byte> buffer,
        void* fromAddress) = 0;

    /// @brief Returns a human-readable name for this transport.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::net::transport
