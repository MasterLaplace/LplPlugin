#include <lpl/history/Calendar.hpp>
#include <lpl/history/Chronicle.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Era.hpp>
#include <lpl/history/Parity.hpp>
#include <lpl/history/PossibleWorld.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(history);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;

[[nodiscard]] lpl::history::Fact factOfYear(lpl::core::u32 subject, lpl::core::i32 year)
{
    lpl::history::Fact fact;

    fact.subject = subject;
    fact.predicate = 11u;
    fact.object = 1u;
    fact.fromDay = lpl::history::firstDayOfYear(year);
    fact.toDay = lpl::history::lastDayOfYear(year);
    fact.sigma = lpl::math::Fixed32::half();
    return fact;
}

/**
 * @brief Five seed constraints, in 1200, 1201, 1207, 1250 and 1299.
 */
[[nodiscard]] lpl::history::Timeline fiveConstraintsInOneCentury()
{
    const lpl::core::i32 years[] = {1200, 1201, 1207, 1250, 1299};
    lpl::history::Timeline timeline;

    for (lpl::core::u32 index = 0u; index < 5u; ++index)
    {
        lpl::history::Constraint constraint;

        constraint.fact = factOfYear(1000u + index, years[index]);
        constraint.kind = lpl::history::ConstraintKind::Seed;
        timeline.add(constraint);
    }
    timeline.finalise();
    return timeline;
}

/**
 * @brief Walks the thirteenth century at @p daysPerTick and counts how many times each of the
 *        five constraints starts inside a tick.
 *
 * @return Whether every constraint started exactly once.
 */
[[nodiscard]] bool firesEachConstraintOnce(const lpl::history::Timeline &timeline, lpl::core::u32 daysPerTick)
{
    lpl::history::Era era;
    lpl::core::u32 timesFired[5] = {0u, 0u, 0u, 0u, 0u};

    era.startDay = lpl::history::firstDayOfYear(1200);
    era.endDay = lpl::history::lastDayOfYear(1299);
    era.daysPerTick = daysPerTick;
    for (lpl::core::u32 tick = 0u; tick < era.totalTicks(); ++tick)
    {
        lpl::core::i32 spanFrom = 0;
        lpl::core::i32 spanTo = 0;
        lpl::core::u32 first = 0u;
        lpl::core::u32 count = 0u;

        era.spanOfTick(tick, spanFrom, spanTo);
        if (!timeline.constraintsStartingIn(spanFrom, spanTo, first, count))
            continue;
        for (lpl::core::u32 index = 0u; index < count; ++index)
            ++timesFired[timeline.at(first + index).fact.subject - 1000u];
    }

    bool once = true;

    for (const lpl::core::u32 fired : timesFired)
        once = once && fired == 1u;
    return once;
}

} // namespace

LPL_TEST(a_source_is_worth_what_it_is_worth)
{
    lpl::history::SourceProfile charter;
    lpl::history::SourceProfile panegyric;

    charter.kind = lpl::history::SourceKind::Notarial;
    charter.yearsAfterEvent = 0u;
    charter.independentAgreements = 1u;
    panegyric.kind = lpl::history::SourceKind::Panegyric;
    panegyric.yearsAfterEvent = 40u;
    panegyric.independentAgreements = 0u;
    test.check(lpl::history::trustworthiness(charter) > lpl::history::trustworthiness(panegyric),
               "a notarial act of the same year beats a panegyric written forty years later");

    lpl::history::SourceProfile far = charter;
    lpl::history::SourceProfile further = charter;

    far.yearsAfterEvent = 500u;
    further.yearsAfterEvent = 5000u;

    const lpl::math::Fixed32 near = lpl::history::trustworthiness(charter);
    const lpl::math::Fixed32 distant = lpl::history::trustworthiness(far);
    const lpl::math::Fixed32 ancient = lpl::history::trustworthiness(further);

    test.check(near > distant && distant > ancient, "distance always costs something");
    test.check((near - distant) > (distant - ancient), "and the first centuries cost far more than the later ones");
}

/**
 * @brief A chronicle compiled around 1200 is ten years from an event of 1190 and eight hundred from
 *        one of 400: the distance belongs to the claim, not to the source.
 */
LPL_TEST(distance_belongs_to_the_claim)
{
    lpl::history::SourceProfile compiler;

    compiler.kind = lpl::history::SourceKind::Chronicle;
    compiler.composedFrom = lpl::history::firstDayOfYear(1195);
    compiler.composedTo = lpl::history::lastDayOfYear(1205);
    test.check(lpl::history::distanceInYears(compiler, lpl::history::firstDayOfYear(1190)) < 20u,
               "the chronicle is close to what it witnessed");
    test.check(lpl::history::distanceInYears(compiler, lpl::history::firstDayOfYear(400)) > 700u,
               "and far from what it repeats");
    test.check(lpl::history::trustworthiness(compiler, lpl::history::firstDayOfYear(1190)) >
                   lpl::history::trustworthiness(compiler, lpl::history::firstDayOfYear(400)),
               "so it is worth more about the first than the second");

    lpl::history::SourceProfile vague = compiler;

    vague.composedFrom = lpl::history::firstDayOfYear(1150);
    test.check(lpl::history::trustworthiness(vague, lpl::history::firstDayOfYear(1190)) <=
                   lpl::history::trustworthiness(compiler, lpl::history::firstDayOfYear(1190)),
               "a wider window of composition does not improve the score");

    lpl::history::SourceProfile stated;

    stated.kind = lpl::history::SourceKind::Chronicle;
    stated.yearsAfterEvent = 40u;
    test.check(lpl::history::distanceInYears(stated, lpl::history::firstDayOfYear(1000)) == 40u,
               "a stated distance is used when no window is given");
}

LPL_TEST(independent_agreement_raises_confidence)
{
    const lpl::math::Fixed32 half = lpl::math::Fixed32::half();
    const lpl::math::Fixed32 fused = lpl::history::fuseConfidence(half, half);

    test.check(fused > half, "two independent sources at one half are worth more than one");
    test.check(fused < lpl::math::Fixed32::one(), "but never certainty");
    test.check(lpl::history::fuseConfidence(lpl::math::Fixed32::one(), lpl::math::Fixed32::zero()) ==
                   lpl::math::Fixed32::one(),
               "certainty fused with ignorance stays certainty");
}

/**
 * @brief Gate P13 history: a chronicler says the king died in battle, his bones say dysentery. The
 *        consensus takes the bones, the chronicler's account stays reachable, and the timelines
 *        fold the same on both targets.
 */
LPL_TEST(the_consensus_wins_and_the_myth_stays_traceable)
{
    lpl::history::HistoryFoldResult folded{};

    lpl::history::foldHistoryState(folded);
    test.check(folded.contradictions == 1u, "the two accounts of the death contradict");
    test.check(folded.consensusObject == lpl::history::kObjectDysentery, "the consensus takes the bones");
    test.check(folded.demoted == 1u, "the chronicler's account is pushed down");
    test.check(folded.minorityReachable == 1u, "and is still there, as the evidence for the decision");
    test.check(folded.minoritySignature != folded.timelineSignature,
               "the world according to the chronicler alone is another world");
    test.check(folded.scoredClaims > 0u, "the timeline asks something of the run");

    test.measureHexadecimal("timeline_signature", folded.timelineSignature);
    test.measureHexadecimal("chronicle_signature", folded.chronicleSignature);
    test.measureHexadecimal("minority_signature", folded.minoritySignature);
    test.measure("constraints", folded.constraints);
    test.measure("scored_claims", folded.scoredClaims);
    test.measure("earned", folded.earned);
}

/**
 * @brief An event the timeline caused earns nothing: reproducing its own inputs is not a
 *        reconstruction. The same event, emitted by a system, earns the whole claim.
 */
LPL_TEST(only_what_was_not_caused_earns)
{
    lpl::history::Fact claim;
    lpl::history::Constraint scored;
    lpl::history::Timeline asked;
    lpl::history::Chronicle selfFulfilled;
    lpl::history::Chronicle honest;
    lpl::history::Attestation caused;
    lpl::history::Attestation emergent;
    lpl::history::Timeline nothing;

    claim.subject = 7u;
    claim.predicate = 8u;
    claim.object = 9u;
    claim.fromDay = 1300;
    claim.toDay = 1300;
    claim.sigma = lpl::math::Fixed32::half();
    scored.fact = claim;
    scored.kind = lpl::history::ConstraintKind::Score;
    asked.add(scored);
    asked.finalise();
    nothing.finalise();
    caused.cause = lpl::history::Cause::Constraint;
    emergent.cause = lpl::history::Cause::Emergent;
    selfFulfilled.record(claim, caused);
    honest.record(claim, emergent);

    const lpl::history::Divergence cheating = lpl::history::measureDivergence(selfFulfilled, asked);
    const lpl::history::Divergence real = lpl::history::measureDivergence(honest, asked);
    const lpl::history::Divergence vacuous = lpl::history::measureDivergence(honest, nothing);

    test.check(cheating.earned == 0u && cheating.selfFulfilled == 1u, "a caused event earns nothing, and says so");
    test.check(cheating.score == lpl::math::Fixed32::zero(), "so its score is zero");
    test.check(real.earned == 1u && real.score == lpl::math::Fixed32::one(), "an emergent event earns the whole claim");
    test.check(!vacuous.acceptable(lpl::math::Fixed32::zero()), "a timeline that asks nothing is not reconstructed");
}

/**
 * @brief A constraint fires exactly once whatever the clock rate, including at a century per step:
 *        firing on a year equal to the fact's skipped whole steps, and firing on every overlapping
 *        tick would fold differently at each rate.
 */
LPL_TEST(every_clock_rate_applies_each_constraint_once)
{
    const lpl::history::Timeline timeline = fiveConstraintsInOneCentury();

    test.check(firesEachConstraintOnce(timeline, 1u), "a day per tick applies each constraint once");
    test.check(firesEachConstraintOnce(timeline, 7u), "so does a week per tick");
    test.check(firesEachConstraintOnce(timeline, 365u), "a year per tick");
    test.check(firesEachConstraintOnce(timeline, 36525u), "and a century in one step");
}

LPL_TEST(file_order_does_not_change_the_timeline)
{
    lpl::history::Corpus corpus;
    lpl::history::Corpus reversed;
    lpl::history::WorldView view;
    lpl::history::FusionReport report{};
    lpl::history::FusionReport reversedReport{};

    lpl::history::parityCorpus(corpus);
    reversed.sources = corpus.sources;
    for (lpl::core::usize index = corpus.facts.size(); index > 0u; --index)
        reversed.facts.push_back(corpus.facts[index - 1u]);

    const lpl::history::Timeline forward = lpl::history::buildTimeline(corpus, view, report);
    const lpl::history::Timeline backward = lpl::history::buildTimeline(reversed, view, reversedReport);

    test.check(forward.fold(kFnv1aOffsetBasis) == backward.fold(kFnv1aOffsetBasis),
               "the same corpus typed backwards folds the same timeline");
}
