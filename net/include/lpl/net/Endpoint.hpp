/**
 * @file Endpoint.hpp
 * @brief Transport-neutral network address (IPv4 + UDP port).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-17
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_ENDPOINT_HPP
#    define LPL_NET_ENDPOINT_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::net {

/**
 * @class Endpoint
 * @brief An IPv4 address and UDP port, in host byte order.
 *
 * The transport-facing address type. It carries no operating-system type, so
 * ITransport, the packet builders and the session layer stay free of
 * <netinet/in.h> / <winsock2.h> and compile on every target — POSIX, Windows
 * and the freestanding kernel. Each concrete transport converts an Endpoint to
 * whatever its platform speaks (SocketTransport to sockaddr_in, byte-swapping
 * on the way out) and is the only place allowed to know that representation.
 *
 * Host byte order is the invariant: callers never htons/htonl, transports
 * always do.
 */
class Endpoint {
public:
    constexpr Endpoint() noexcept = default;

    /**
     * @param address IPv4 address, host byte order (0x7F000001 == 127.0.0.1).
     * @param port UDP port, host byte order.
     */
    constexpr Endpoint(core::u32 address, core::u16 port) noexcept : _address{address}, _port{port} {}

    /** @brief Builds an endpoint from dotted-quad octets. */
    [[nodiscard]] static constexpr Endpoint fromOctets(core::u8 a, core::u8 b, core::u8 c, core::u8 d,
                                                       core::u16 port) noexcept
    {
        return Endpoint{(static_cast<core::u32>(a) << 24) | (static_cast<core::u32>(b) << 16) |
                            (static_cast<core::u32>(c) << 8) | static_cast<core::u32>(d),
                        port};
    }

    /**
     * @brief Parses a dotted-quad string ("127.0.0.1").
     *
     * Hand-rolled rather than inet_pton: this header must stay usable where no
     * sockets API exists.
     *
     * @param text Dotted-quad text, NUL-terminated.
     * @param port UDP port, host byte order.
     * @param[out] outEndpoint Filled on success, untouched on failure.
     * @return true if @p text is a well-formed dotted quad.
     */
    [[nodiscard]] static constexpr bool parse(const char *text, core::u16 port, Endpoint &outEndpoint) noexcept
    {
        if (text == nullptr)
            return false;

        core::u32 address = 0;
        core::u32 octet = 0;
        core::u32 digits = 0;
        core::u32 octetsSeen = 0;

        for (const char *cursor = text;; ++cursor)
        {
            const char character = *cursor;

            if (character >= '0' && character <= '9')
            {
                if (++digits > 3)
                    return false;
                octet = octet * 10 + static_cast<core::u32>(character - '0');
                if (octet > 255)
                    return false;
                continue;
            }

            if (character != '.' && character != '\0')
                return false;
            if (digits == 0)
                return false;

            address = (address << 8) | octet;
            ++octetsSeen;
            octet = 0;
            digits = 0;

            if (character == '\0')
                break;
            if (octetsSeen == 4)
                return false; // trailing dot
        }

        if (octetsSeen != 4)
            return false;

        outEndpoint = Endpoint{address, port};
        return true;
    }

    /** @brief IPv4 address, host byte order. */
    [[nodiscard]] constexpr core::u32 address() const noexcept { return _address; }

    /** @brief UDP port, host byte order. */
    [[nodiscard]] constexpr core::u16 port() const noexcept { return _port; }

    /** @brief A zero port means "unset"; port 0 is not a routable destination. */
    [[nodiscard]] constexpr bool valid() const noexcept { return _port != 0; }

    [[nodiscard]] friend constexpr bool operator==(const Endpoint &lhs, const Endpoint &rhs) noexcept
    {
        return lhs._address == rhs._address && lhs._port == rhs._port;
    }

private:
    core::u32 _address{0};
    core::u16 _port{0};
};

} // namespace lpl::net

#endif // LPL_NET_ENDPOINT_HPP
