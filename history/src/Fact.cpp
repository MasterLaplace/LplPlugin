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
 * @warning An unknown distance earns NOTHING here, and that is the conservative direction on
 * purpose. Proximity is a credit, and a credit nobody has evidenced must not be granted: the
 * alternative -- treating "we do not know when this was written" as "written on the day" --
 * is what happened while zero doubled as the unknown, and it scored every classical text as
 * an eyewitness account.
 *
 * The distance is saturated at 30000 years before the divide: a distance in the millions of
 * years is representable in the corpus (Pleiades dates occupation layers to 2.6 Ma) and would
 * overflow Q16.16 on the way in. Anything past it already keeps less than 0.11 %.
 *
 * @param years Distance between the event and the writing.
 * @return The factor.
 */
[[nodiscard]] math::Fixed32 proximityFactor(core::u32 years) noexcept
{
    if (years == kUnknownYearsAfterEvent)
        return math::Fixed32::zero();
    const core::u32 capped = years > 30000u ? 30000u : years;
    const math::Fixed32 scaled = math::Fixed32::fromInt(static_cast<core::i32>(capped)) / math::Fixed32::fromInt(32);
    return math::Fixed32::one() / (math::Fixed32::one() + scaled);
}

/**
 * @brief What a kind of source is worth on its own, in [0, 1].
 *
 * Values, not a formula: this is a historiographical position and it belongs written
 * down where it can be argued with, rather than derived from an enum's index. Notarial
 * 0.92, archaeology 0.88, administrative 0.80, chronicle 0.55, panegyric 0.20, each
 * written as its raw Q16.16 word.
 *
 * @param kind The kind.
 * @return The factor.
 */
[[nodiscard]] math::Fixed32 kindFactor(SourceKind kind) noexcept
{
    switch (kind)
    {
    case SourceKind::Notarial: return math::Fixed32::fromRaw(60293);
    case SourceKind::Archaeology: return math::Fixed32::fromRaw(57672);
    case SourceKind::Administrative: return math::Fixed32::fromRaw(52429);
    case SourceKind::Chronicle: return math::Fixed32::fromRaw(36044);
    case SourceKind::Panegyric: return math::Fixed32::fromRaw(13107);
    case SourceKind::Count: break;
    }
    return math::Fixed32::half();
}

} // namespace

math::Fixed32 trustworthiness(const SourceProfile &profile, const TrustWeights &weights) noexcept
{
    const math::Fixed32 agreements = math::Fixed32::fromInt(static_cast<core::i32>(profile.independentAgreements));
    const math::Fixed32 consensus = agreements / (agreements + math::Fixed32::fromInt(3));

    const math::Fixed32 score = weights.temporalProximity * proximityFactor(profile.yearsAfterEvent) +
                                weights.sourceType * kindFactor(profile.kind) + weights.peerConsensus * consensus;

    if (score.raw() < 0)
        return math::Fixed32::zero();
    return score > math::Fixed32::one() ? math::Fixed32::one() : score;
}

core::u32 distanceInYears(const SourceProfile &profile, core::i32 eventDay) noexcept
{
    if (profile.composedFrom != 0 || profile.composedTo != 0)
    {
        const core::i32 from = profile.composedFrom;
        const core::i32 to = profile.composedTo < from ? from : profile.composedTo;
        core::i32 gap = 0;
        if (eventDay < from)
            gap = from - eventDay;
        else if (eventDay > to)
            gap = eventDay - to;
        if (gap < 0)
            gap = -gap;
        const core::i32 width = to > from ? to - from : 0;
        const core::i64 days = static_cast<core::i64>(gap) + static_cast<core::i64>(width);
        return static_cast<core::u32>(days / 365);
    }
    return profile.yearsAfterEvent;
}

math::Fixed32 trustworthiness(const SourceProfile &profile, core::i32 eventDay, const TrustWeights &weights) noexcept
{
    SourceProfile dated = profile;
    dated.yearsAfterEvent = distanceInYears(profile, eventDay);
    return trustworthiness(dated, weights);
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
    return a.fromDay <= b.toDay && b.fromDay <= a.toDay;
}

} // namespace lpl::history
