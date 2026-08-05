/**
 * @file PossibleWorld.cpp
 * @brief Corpus to timeline: admission, fusion, contradiction.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/PossibleWorld.hpp>

namespace lpl::history {

bool Corpus::sourceProfile(core::u32 id, SourceProfile &outProfile) const noexcept
{
    for (core::usize i = 0u; i < sources.size(); ++i)
    {
        if (sources[i].id != id)
            continue;
        outProfile = sources[i];
        return true;
    }
    return false;
}

Timeline buildTimeline(const Corpus &corpus, const WorldView &view, FusionReport &outReport)
{
    outReport = FusionReport{};
    Timeline timeline;

    const auto admits = [&view](core::u32 source) {
        if (view.admittedSources.empty())
            return true;
        for (core::usize i = 0u; i < view.admittedSources.size(); ++i)
            if (view.admittedSources[i] == source)
                return true;
        return false;
    };

    // ── 1. Admission, and the weight this world puts on each claim ────────────
    //
    // A claim's weight is its own sigma tempered by how much its source is worth
    // believing. A confident assertion in a panegyric is a confident assertion in a
    // panegyric.
    lpl::pmr::vector<core::u32> index;
    lpl::pmr::vector<math::Fixed32> weight;
    for (core::usize i = 0u; i < corpus.facts.size(); ++i)
    {
        const Fact &fact = corpus.facts[i];
        if (!admits(fact.source))
        {
            ++outReport.rejectedSource;
            continue;
        }

        SourceProfile profile{};
        const math::Fixed32 trust = corpus.sourceProfile(fact.source, profile)
                                        ? trustworthiness(profile, view.weights)
                                        : math::Fixed32::half();
        index.push_back(static_cast<core::u32>(i));
        weight.push_back(fact.sigma * trust);
        ++outReport.admitted;
    }

    // ── 2. Fusion of independent agreement ────────────────────────────────────
    for (core::usize a = 0u; a < index.size(); ++a)
        for (core::usize b = a + 1u; b < index.size(); ++b)
        {
            const Fact &left = corpus.facts[index[a]];
            const Fact &right = corpus.facts[index[b]];
            if (left.source == right.source)
                continue; // the same source agreeing with itself is not evidence
            if (left.subject != right.subject || left.predicate != right.predicate || left.object != right.object)
                continue;
            // The windows have to OVERLAP. Two sources saying a settlement existed —
            // one in 1200 and one in 1900 — are not agreeing with each other, they are
            // describing two different states of the world, and fusing them would
            // manufacture confidence out of the mere reuse of a name.
            if (left.fromYear > right.toYear || right.fromYear > left.toYear)
                continue;

            const math::Fixed32 fused = fuseConfidence(weight[a], weight[b]);
            weight[a] = fused;
            weight[b] = fused;
            ++outReport.fused;
        }

    // ── 3. Contradiction: demote, never delete ────────────────────────────────
    for (core::usize a = 0u; a < index.size(); ++a)
        for (core::usize b = a + 1u; b < index.size(); ++b)
        {
            if (!contradicts(corpus.facts[index[a]], corpus.facts[index[b]]))
                continue;
            ++outReport.contradictions;

            // The less supported claim loses ground in proportion to how far ahead the
            // other is. It keeps a floor: a demoted claim that reached zero would be
            // indistinguishable from one nobody ever made, and the point is precisely
            // that the myth stays traceable.
            const core::usize loser = weight[a] < weight[b] ? a : b;
            const core::usize winner = loser == a ? b : a;
            const math::Fixed32 gap = weight[winner] - weight[loser];
            const math::Fixed32 floorValue = math::Fixed32::fromRaw(3277); // 0.05

            math::Fixed32 demoted = weight[loser] * (math::Fixed32::one() - gap);
            if (demoted < floorValue)
                demoted = floorValue;
            if (demoted < weight[loser])
            {
                weight[loser] = demoted;
                ++outReport.demoted;
            }
        }

    // ── The verdict: what the run must do about each ──────────────────────────
    for (core::usize i = 0u; i < index.size(); ++i)
    {
        Constraint constraint{};
        constraint.fact = corpus.facts[index[i]];
        constraint.confidence = weight[i];

        if (weight[i] >= view.forceThreshold)
        {
            constraint.kind = ConstraintKind::Force;
            ++outReport.forced;
        }
        else if (weight[i] >= view.seedThreshold)
        {
            constraint.kind = ConstraintKind::Seed;
            ++outReport.seeded;
        }
        else
        {
            constraint.kind = ConstraintKind::Score;
            ++outReport.scored;
        }
        timeline.add(constraint);
    }

    timeline.finalise();
    return timeline;
}

} // namespace lpl::history
