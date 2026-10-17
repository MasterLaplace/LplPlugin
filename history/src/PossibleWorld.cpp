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
        const math::Fixed32 trust = corpus.sourceProfile(fact.source, profile) ?
                                        trustworthiness(profile, fact.fromDay, view.weights) :
                                        math::Fixed32::half();
        index.push_back(static_cast<core::u32>(i));
        weight.push_back(fact.sigma * trust);
        ++outReport.admitted;
    }

    for (core::usize a = 0u; a < index.size(); ++a)
        for (core::usize b = a + 1u; b < index.size(); ++b)
        {
            const Fact &left = corpus.facts[index[a]];
            const Fact &right = corpus.facts[index[b]];
            if (left.source == right.source)
                continue;
            if (left.subject != right.subject || left.predicate != right.predicate || left.object != right.object)
                continue;
            if (left.fromDay > right.toDay || right.fromDay > left.toDay)
                continue;

            const math::Fixed32 fused = fuseConfidence(weight[a], weight[b]);
            weight[a] = fused;
            weight[b] = fused;
            ++outReport.fused;
        }

    for (core::usize a = 0u; a < index.size(); ++a)
        for (core::usize b = a + 1u; b < index.size(); ++b)
        {
            if (!contradicts(corpus.facts[index[a]], corpus.facts[index[b]]))
                continue;
            ++outReport.contradictions;

            const core::usize loser = weight[a] < weight[b] ? a : b;
            const core::usize winner = loser == a ? b : a;
            const math::Fixed32 gap = weight[winner] - weight[loser];
            const math::Fixed32 floorValue = math::Fixed32::fromRaw(3277);

            math::Fixed32 demoted = weight[loser] * (math::Fixed32::one() - gap);
            if (demoted < floorValue)
                demoted = floorValue;
            if (demoted < weight[loser])
            {
                weight[loser] = demoted;
                ++outReport.demoted;
            }
        }

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
