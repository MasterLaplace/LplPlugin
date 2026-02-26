// /////////////////////////////////////////////////////////////////////////////
/// @file Reconciliation.cpp
/// @brief Reconciliation implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/netcode/Reconciliation.hpp>

namespace lpl::net::netcode {

Reconciliation::Reconciliation(Prediction& prediction)
    : prediction_{prediction}
{}

Reconciliation::~Reconciliation() = default;

core::Expected<void> Reconciliation::reconcile(
    std::span<const core::byte> authoritativeState,
    core::u32 ackedSequence,
    core::f32 dt,
    const std::function<void(std::span<const core::byte>)>& applyState,
    const ResimulateCallback& resimulate)
{
    applyState(authoritativeState);

    prediction_.acknowledge(ackedSequence);

    const auto unacked = prediction_.getUnacknowledged(ackedSequence);
    for (const auto& input : unacked)
    {
        resimulate(input.data, dt);
    }

    return {};
}

} // namespace lpl::net::netcode
