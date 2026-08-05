/**
 * @file Chronicle.cpp
 * @brief The run's account, in emission order.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/Chronicle.hpp>

namespace lpl::history {

namespace {

constexpr core::u32 kFnv1aPrime = 0x01000193u;

} // namespace

void Chronicle::record(const Fact &fact, const Attestation &attestation)
{
    Event event;
    event.fact = fact;
    event.attestation = attestation;
    _events.push_back(event);
}

const Event &Chronicle::at(core::u32 index) const noexcept
{
    static const Event kEmpty{};
    return index < _events.size() ? _events[index] : kEmpty;
}

core::u32 Chronicle::countByCause(Cause cause) const noexcept
{
    core::u32 total = 0u;
    for (core::usize i = 0u; i < _events.size(); ++i)
        total += _events[i].attestation.cause == cause ? 1u : 0u;
    return total;
}

core::u32 Chronicle::fold(core::u32 seed) const noexcept
{
    core::u32 hash = seed;
    const auto absorb = [&hash](core::u32 word) { hash = (hash ^ word) * kFnv1aPrime; };

    for (core::usize i = 0u; i < _events.size(); ++i)
    {
        const Event &e = _events[i];
        absorb(e.fact.subject);
        absorb(e.fact.predicate);
        absorb(e.fact.object);
        absorb(static_cast<core::u32>(e.fact.fromYear));
        absorb(static_cast<core::u32>(e.fact.toYear));
        absorb(static_cast<core::u32>(e.fact.sigma.raw()));
        absorb(static_cast<core::u32>(e.attestation.cause));
        absorb(e.attestation.agent);
    }
    return hash;
}

} // namespace lpl::history
