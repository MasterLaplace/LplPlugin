/**
 * @file Fact.cpp
 * @brief Trust scoring and Bayesian fusion, in fixed point.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/Fact.hpp>

namespace lpl::history {

namespace {

/**
 * @brief How much a source's distance from its event costs it, in [0, 1].
 *
 * A hyperbola rather than a threshold: 1 / (1 + years/32). Five years keeps 87 %,
 * fifty keeps 39 %, five hundred keeps 6 % and five thousand keeps 0.6 % — which is
 * the right shape, because the difference between an eyewitness and a grandchild is
 * enormous and the difference between five centuries and fifty is not.
 *
 * @param years Distance between the event and the writing.
 * @return The factor.
 */
[[nodiscard]] math::Fixed32 proximityFactor(core::u32 years) noexcept
{
    const math::Fixed32 scaled = math::Fixed32::fromInt(static_cast<core::i32>(years)) / math::Fixed32::fromInt(32);
    return math::Fixed32::one() / (math::Fixed32::one() + scaled);
}

/**
 * @brief What a kind of source is worth on its own, in [0, 1].
 *
 * Values, not a formula: this is a historiographical position and it belongs written
 * down where it can be argued with, rather than derived from an enum's index.
 *
 * @param kind The kind.
 * @return The factor.
 */
[[nodiscard]] math::Fixed32 kindFactor(SourceKind kind) noexcept
{
    switch (kind)
    {
    case SourceKind::Notarial: return math::Fixed32::fromRaw(60293);       // 0.92
    case SourceKind::Archaeology: return math::Fixed32::fromRaw(57672);    // 0.88
    case SourceKind::Administrative: return math::Fixed32::fromRaw(52429); // 0.80
    case SourceKind::Chronicle: return math::Fixed32::fromRaw(36044);      // 0.55
    case SourceKind::Panegyric: return math::Fixed32::fromRaw(13107);      // 0.20
    case SourceKind::Count: break;
    }
    return math::Fixed32::half();
}

} // namespace

math::Fixed32 trustworthiness(const SourceProfile &profile, const TrustWeights &weights) noexcept
{
    // Agreement saturates: the second independent source that says the same thing is
    // worth far more than the tenth, and a term that grew without bound would let a
    // crowd of weak sources outweigh a notarial act.
    const math::Fixed32 agreements = math::Fixed32::fromInt(static_cast<core::i32>(profile.independentAgreements));
    const math::Fixed32 consensus = agreements / (agreements + math::Fixed32::fromInt(3));

    const math::Fixed32 score = weights.temporalProximity * proximityFactor(profile.yearsAfterEvent) +
                                weights.sourceType * kindFactor(profile.kind) + weights.peerConsensus * consensus;

    if (score.raw() < 0)
        return math::Fixed32::zero();
    return score > math::Fixed32::one() ? math::Fixed32::one() : score;
}

math::Fixed32 fuseConfidence(math::Fixed32 a, math::Fixed32 b) noexcept
{
    const math::Fixed32 one = math::Fixed32::one();
    const math::Fixed32 left = a.raw() < 0 ? math::Fixed32::zero() : (a > one ? one : a);
    const math::Fixed32 right = b.raw() < 0 ? math::Fixed32::zero() : (b > one ? one : b);
    return one - (one - left) * (one - right);
}

bool contradicts(const Fact &a, const Fact &b) noexcept
{
    if (a.subject != b.subject || a.predicate != b.predicate)
        return false;
    if (a.object == b.object)
        return false;
    // Overlapping windows. Touching endpoints count: a claim covering [1200,1204] and
    // one covering [1204,1210] both assert something about 1204.
    return a.fromYear <= b.toYear && b.fromYear <= a.toYear;
}

} // namespace lpl::history
