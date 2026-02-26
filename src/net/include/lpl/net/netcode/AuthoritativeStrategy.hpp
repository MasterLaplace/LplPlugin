// /////////////////////////////////////////////////////////////////////////////
/// @file AuthoritativeStrategy.hpp
/// @brief Server-authoritative netcode with client-side prediction.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/net/netcode/INetcodeStrategy.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::net::netcode {

// /////////////////////////////////////////////////////////////////////////////
/// @class AuthoritativeStrategy
/// @brief Classic server-authoritative model.
///
/// Server processes all inputs and broadcasts state. Client predicts locally
/// and reconciles when the authoritative state arrives.
// /////////////////////////////////////////////////////////////////////////////
class AuthoritativeStrategy final : public INetcodeStrategy,
                                     public core::NonCopyable<AuthoritativeStrategy>
{
public:
    AuthoritativeStrategy();
    ~AuthoritativeStrategy() override;

    [[nodiscard]] core::Expected<void> onInputReceived(
        core::u32 playerId,
        std::span<const core::byte> inputData,
        core::u32 sequence) override;

    [[nodiscard]] core::Expected<void> onStateReceived(
        std::span<const core::byte> snapshotData,
        core::u32 sequence) override;

    void tick(core::f32 dt) override;

    [[nodiscard]] const char* name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::net::netcode
