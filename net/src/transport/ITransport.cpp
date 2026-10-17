#include <lpl/net/transport/ITransport.hpp>

namespace lpl::net::transport {

core::Expected<core::u32> ITransport::sendBatch(std::span<const Datagram> datagrams)
{
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
    core::u32 received = 0;
    for (auto &slot : slots)
    {
        auto result = receive(slot.buffer, &slot.source);
        if (!result.has_value())
            return std::unexpected(result.error());
        if (result.value() == 0)
            break;
        slot.length = result.value();
        ++received;
    }
    return received;
}

} // namespace lpl::net::transport
