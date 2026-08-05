/**
 * @file test_history_parity.cpp
 * @brief SIM-022 end to end: the consensus wins, and the myth stays traceable.
 *
 * The example is small on purpose and it is the whole model at once. A chronicler
 * says the king died in battle; osteology says dysentery. They cannot both hold. The
 * default view must take the second — and the first must still be THERE, because a
 * pipeline that deletes the loser destroys the evidence for its own decision and no
 * one can afterwards retrace how the myth was built.
 *
 * The second claim tested here is the one that makes divergence a measurement rather
 * than a congratulation: an event the timeline CAUSED cannot count as agreement with
 * that timeline.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/Chronicle.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Era.hpp>
#include <lpl/history/Parity.hpp>
#include <lpl/history/PossibleWorld.hpp>

#include <cstdio>

namespace {

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== history: two sources, one past ==\n");

    // ── Trust scoring, and what it is for ─────────────────────────────────────
    std::printf("\n-- a source is worth what it is worth --\n");
    {
        history::SourceProfile charter;
        charter.kind = history::SourceKind::Notarial;
        charter.yearsAfterEvent = 0u;
        charter.independentAgreements = 1u;

        history::SourceProfile panegyric;
        panegyric.kind = history::SourceKind::Panegyric;
        panegyric.yearsAfterEvent = 40u;
        panegyric.independentAgreements = 0u;

        const math::Fixed32 trusted = history::trustworthiness(charter);
        const math::Fixed32 flattering = history::trustworthiness(panegyric);
        std::printf("    notarial act, same year : %.3f\n", static_cast<double>(trusted.toFloat()));
        std::printf("    panegyric, forty years  : %.3f\n", static_cast<double>(flattering.toFloat()));
        check(trusted > flattering, "a notarial act written the same year beats a panegyric written forty later");

        // Distance costs, and the cost is not linear: the difference between an
        // eyewitness and a grandchild is enormous, five centuries and fifty is not.
        history::SourceProfile near = charter;
        history::SourceProfile far = charter;
        far.yearsAfterEvent = 500u;
        history::SourceProfile further = charter;
        further.yearsAfterEvent = 5000u;
        const math::Fixed32 a = history::trustworthiness(near);
        const math::Fixed32 b = history::trustworthiness(far);
        const math::Fixed32 c = history::trustworthiness(further);
        check(a > b && b > c, "distance always costs something");
        check((a - b) > (b - c), "but the first centuries cost far more than the later ones");
    }

    // ── Fusion, and the word doing the work ───────────────────────────────────
    std::printf("\n-- independent agreement raises confidence --\n");
    {
        const math::Fixed32 half = math::Fixed32::half();
        const math::Fixed32 fused = history::fuseConfidence(half, half);
        std::printf("    0.50 fused with 0.50 = %.3f\n", static_cast<double>(fused.toFloat()));
        check(fused > half, "two independent sources at 0.5 are worth more than one");
        check(fused < math::Fixed32::one(), "but never certainty");
        check(history::fuseConfidence(math::Fixed32::one(), math::Fixed32::zero()) == math::Fixed32::one(),
              "certainty fused with ignorance stays certainty");
    }

    // ── The contradiction, and what happens to the loser ──────────────────────
    std::printf("\n-- SIM-022: the king --\n");
    history::HistoryFoldResult folded{};
    history::foldHistoryState(folded);

    std::printf("    consensus says he died of: %s\n",
                folded.consensusObject == history::kObjectDysentery
                    ? "dysentery (the bones)"
                    : (folded.consensusObject == history::kObjectBattle ? "battle (the chronicler)" : "?"));
    std::printf("    contradictions=%u demoted=%u constraints=%u\n", folded.contradictions, folded.demoted,
                folded.constraints);

    check(folded.contradictions == 1u, "the two accounts of his death are seen to contradict");
    check(folded.consensusObject == history::kObjectDysentery, "and the consensus view takes the bones");
    check(folded.demoted == 1u, "the chronicler's account is pushed down");
    check(folded.minorityReachable == 1u,
          "and is STILL THERE — a deleted loser would destroy the evidence for the decision");

    // The minority world is the same corpus through the same function with one entry
    // in admittedSources. If it folded the same, "possible worlds" would be a label.
    check(folded.minoritySignature != folded.timelineSignature,
          "the world according to the chronicler alone is a genuinely different world");

    // ── Divergence, and the rule that makes it a measurement ──────────────────
    std::printf("\n-- what the run had to earn --\n");
    std::printf("    scored=%u earned=%u score=%.3f\n", folded.scoredClaims, folded.earned,
                static_cast<double>(math::Fixed32::fromRaw(static_cast<core::i32>(folded.divergenceScore)).toFloat()));
    check(folded.scoredClaims > 0u, "the timeline asked something of the run");

    {
        // The decisive one, built by hand so the shape is visible. A chronicle whose
        // only matching event was CAUSED by the timeline must score zero: reproducing
        // your own inputs is not a reconstruction.
        history::Fact claim;
        claim.subject = 7u;
        claim.predicate = 8u;
        claim.object = 9u;
        claim.fromYear = 1300;
        claim.toYear = 1300;
        claim.sigma = math::Fixed32::half();

        history::Constraint scored;
        scored.fact = claim;
        scored.kind = history::ConstraintKind::Score;
        history::Timeline asked;
        asked.add(scored);
        asked.finalise();

        history::Chronicle selfFulfilled;
        history::Attestation caused;
        caused.cause = history::Cause::Constraint;
        selfFulfilled.record(claim, caused);
        const history::Divergence cheating = history::measureDivergence(selfFulfilled, asked);
        check(cheating.earned == 0u && cheating.selfFulfilled == 1u,
              "an event the timeline caused earns nothing, and says so");
        check(cheating.score == math::Fixed32::zero(), "so the score is zero");

        history::Chronicle honest;
        history::Attestation emergent;
        emergent.cause = history::Cause::Emergent;
        honest.record(claim, emergent);
        const history::Divergence real = history::measureDivergence(honest, asked);
        check(real.earned == 1u && real.score == math::Fixed32::one(),
              "the same event, emitted by a system, earns the whole claim");

        // An empty question must not score perfectly.
        history::Timeline nothing;
        nothing.finalise();
        const history::Divergence vacuous = history::measureDivergence(honest, nothing);
        check(!vacuous.acceptable(math::Fixed32::zero()), "a timeline that asks nothing is not reconstructed");
    }

    // ── Ordering is the contract ──────────────────────────────────────────────
    std::printf("\n-- the order is part of the contract --\n");
    {
        history::Corpus corpus;
        history::parityCorpus(corpus);
        history::WorldView view;
        history::FusionReport report{};
        const history::Timeline forward = history::buildTimeline(corpus, view, report);

        // Same facts, typed in backwards. A timeline that folded differently would make
        // the fold a property of a text file rather than of a corpus.
        history::Corpus reversed;
        reversed.sources = corpus.sources;
        for (core::usize i = corpus.facts.size(); i > 0u; --i)
            reversed.facts.push_back(corpus.facts[i - 1u]);
        history::FusionReport reversedReport{};
        const history::Timeline backward = history::buildTimeline(reversed, view, reversedReport);

        check(forward.fold(0x811C9DC5u) == backward.fold(0x811C9DC5u),
              "the same corpus in a different file order folds identically");
    }

    std::printf("\n-- signatures the kernel must reproduce --\n");
    std::printf("  timeline_sig  = 0x%08X\n", folded.timelineSignature);
    std::printf("  chronicle_sig = 0x%08X\n", folded.chronicleSignature);
    std::printf("  minority_sig  = 0x%08X\n", folded.minoritySignature);
    std::printf("  constraints   = %u\n", folded.constraints);
    std::printf("  scored        = %u\n", folded.scoredClaims);
    std::printf("  earned        = %u\n", folded.earned);

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
