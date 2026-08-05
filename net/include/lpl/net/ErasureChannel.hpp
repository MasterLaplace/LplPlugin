/**
 * @file ErasureChannel.hpp
 * @brief A UDP channel that tolerates loss without retransmitting.
 *
 * Rollback netcode already absorbs latency; this absorbs loss. Fountain-coded
 * datagrams mean a receiver needs any sufficient subset rather than a specific one,
 * which removes a round trip from the tail of every dropped packet.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_NET_ERASURECHANNEL_HPP
#    define LPL_LPL_NET_ERASURECHANNEL_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::net {

// TODO(lot 3): declarations only — no implementation yet.

} // namespace lpl::net

#endif // LPL_LPL_NET_ERASURECHANNEL_HPP
