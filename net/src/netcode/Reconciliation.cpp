/**
 * @file Reconciliation.cpp
 * @brief Reconciliation implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/netcode/Reconciliation.hpp>

namespace lpl::net::netcode {

Reconciliation::Reconciliation(Prediction& prediction)
    : _prediction{prediction}
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

    _prediction.acknowledge(ackedSequence);

    const auto unacked = _prediction.getUnacknowledged(ackedSequence);
    for (const auto& input : unacked)
    {
        resimulate(input.data, dt);
    }

    return {};
}

} // namespace lpl::net::netcode
