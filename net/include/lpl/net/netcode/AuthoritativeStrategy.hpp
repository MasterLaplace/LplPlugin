/**
 * @file AuthoritativeStrategy.hpp
 * @brief Server-authoritative netcode with client-side prediction.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_NETCODE_AUTHORITATIVESTRATEGY_HPP
#    define LPL_NET_NETCODE_AUTHORITATIVESTRATEGY_HPP

#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/net/netcode/INetcodeStrategy.hpp>

#    include <memory>

namespace lpl::net::netcode {

/**
 * @class AuthoritativeStrategy
 * @brief Classic server-authoritative model.
 *
 * Server processes all inputs and broadcasts state. Client predicts locally
 * and reconciles when the authoritative state arrives.
 */
class AuthoritativeStrategy final : public INetcodeStrategy, public core::NonCopyable<AuthoritativeStrategy> {
public:
    AuthoritativeStrategy();
    ~AuthoritativeStrategy() override;

    /**
     * @brief Queues an input for consumption at the next tick.
     *
     * Once a tick has run, an input whose sequence is at or below the last acknowledged sequence
     * has already been processed: it is rejected as stale, and the call still succeeds. Each
     * player's queue is sorted by sequence, so its inputs are processed in order.
     */
    [[nodiscard]] core::Expected<void> onInputReceived(core::u32 playerId, std::span<const core::byte> inputData,
                                                       core::u32 sequence) override;

    /**
     * @brief Does nothing: the server is the authority.
     *
     * In a distributed server mesh this would propagate state between nodes.
     */
    [[nodiscard]] core::Expected<void> onStateReceived(std::span<const core::byte> snapshotData,
                                                       core::u32 sequence) override;

    /**
     * @brief Consumes every pending input, then advances the server frame.
     *
     * The new server frame becomes the last acknowledged sequence.
     */
    void tick(core::f32 dt) override;

    [[nodiscard]] const char *name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net::netcode

#endif // LPL_NET_NETCODE_AUTHORITATIVESTRATEGY_HPP
