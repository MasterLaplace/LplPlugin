/**
 * @file ITransport.cpp
 * @brief Default transport behaviour shared by every implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-22
 * @copyright MIT License
 */

#include <lpl/net/transport/ITransport.hpp>

namespace lpl::net::transport {

core::Expected<core::u32> ITransport::sendBatch(std::span<const Datagram> datagrams)
{
    // The portable fallback: hand the packets over one at a time. A transport
    // with a real batching primitive overrides this (SocketTransport uses
    // sendmmsg on Linux, KernelTransport fills its ring and kicks once); a
    // platform without one — Windows, macOS — inherits correct behaviour here.
    core::u32 sent = 0;
    for (const auto &datagram : datagrams)
    {
        if (!send(datagram.data, datagram.address))
            break;
        ++sent;
    }
    return sent;
}

core::Expected<core::u32> ITransport::receiveBatch(std::span<ReceiveSlot> slots)
{
    // Portable fallback: one receive() per slot until the socket runs dry. A
    // transport with recvmmsg (SocketTransport on Linux) overrides this; a
    // platform without it inherits correct, if unbatched, behaviour here.
    core::u32 received = 0;
    for (auto &slot : slots)
    {
        auto result = receive(slot.buffer, &slot.source);
        if (!result.has_value())
            return std::unexpected(result.error());
        if (result.value() == 0)
            break; // nothing more waiting
        slot.length = result.value();
        ++received;
    }
    return received;
}

} // namespace lpl::net::transport
