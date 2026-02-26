// /////////////////////////////////////////////////////////////////////////////
/// @file Reconciliation.hpp
/// @brief Server reconciliation: re-applies predicted inputs after
///        authoritative state correction.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/net/netcode/Prediction.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <functional>
#include <span>

namespace lpl::net::netcode {

// /////////////////////////////////////////////////////////////////////////////
/// @class Reconciliation
/// @brief Applies authoritative state, then re-applies unacknowledged
///        predicted inputs for smooth client-side correction.
// /////////////////////////////////////////////////////////////////////////////
class Reconciliation final : public core::NonCopyable<Reconciliation>
{
public:
    /// @brief Callback type for re-simulating a single input frame.
    using ResimulateCallback = std::function<void(std::span<const core::byte> inputData,
                                                   core::f32 dt)>;

    /// @brief Constructs a reconciliation module.
    /// @param prediction Reference to the prediction buffer.
    explicit Reconciliation(Prediction& prediction);
    ~Reconciliation();

    /// @brief Performs reconciliation.
    /// @param authoritativeState Bytes of the server-authoritative state.
    /// @param ackedSequence      Last acknowledged input sequence.
    /// @param dt                 Fixed delta-time per frame.
    /// @param applyState         Callable to apply raw state bytes.
    /// @param resimulate         Callable to re-simulate a single input.
    [[nodiscard]] core::Expected<void> reconcile(
        std::span<const core::byte> authoritativeState,
        core::u32 ackedSequence,
        core::f32 dt,
        const std::function<void(std::span<const core::byte>)>& applyState,
        const ResimulateCallback& resimulate);

private:
    Prediction& prediction_;
};

} // namespace lpl::net::netcode
