/**
 * @file Parity.cpp
 * @brief SIM-022, end to end: the consensus wins and the myth stays traceable.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/Parity.hpp>

#include <lpl/history/Chronicle.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Era.hpp>

namespace lpl::history {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;

/**
 * @brief Adds one fact to a corpus.
 * @param corpus    Destination.
 * @param subject   Who.
 * @param predicate What of them.
 * @param object    The value.
 * @param fromYear  Window start.
 * @param toYear    Window end.
 * @param source    Who says so.
 * @param sigmaRaw  Confidence, raw Q16.16.
 */
void addFact(Corpus &corpus, core::u32 subject, core::u32 predicate, core::u32 object, core::i32 fromYear,
             core::i32 toYear, core::u32 source, core::i32 sigmaRaw)
{
    Fact fact;
    fact.subject = subject;
    fact.predicate = predicate;
    fact.object = object;
    fact.fromYear = fromYear;
    fact.toYear = toYear;
    fact.source = source;
    fact.sigma = math::Fixed32::fromRaw(sigmaRaw);
    corpus.facts.push_back(fact);
}

/**
 * @brief Adds one source profile.
 * @param corpus    Destination.
 * @param id        Identifier.
 * @param kind      What kind of source.
 * @param years     Years between the event and the writing.
 * @param agreement Independent sources that concur.
 */
void addSource(Corpus &corpus, core::u32 id, SourceKind kind, core::u32 years, core::u32 agreement)
{
    SourceProfile profile;
    profile.id = id;
    profile.kind = kind;
    profile.yearsAfterEvent = years;
    profile.independentAgreements = agreement;
    corpus.sources.push_back(profile);
}

} // namespace

void parityCorpus(Corpus &outCorpus)
{
    outCorpus.facts.clear();
    outCorpus.sources.clear();

    // The chronicler wrote forty years later and was praising someone. The osteologist
    // read the bones. They cannot both be right about what killed the king.
    addSource(outCorpus, kSourceChronicler, SourceKind::Panegyric, 40u, 0u);
    addSource(outCorpus, kSourceOsteology, SourceKind::Archaeology, 800u, 1u);
    addSource(outCorpus, kSourceCharter, SourceKind::Notarial, 0u, 1u);
    addSource(outCorpus, kSourceSurvey, SourceKind::Administrative, 10u, 1u);

    addFact(outCorpus, kSubjectKing, kPredicateDiedOf, kObjectBattle, 1204, 1204, kSourceChronicler, 29491); // 0.45
    addFact(outCorpus, kSubjectKing, kPredicateDiedOf, kObjectDysentery, 1204, 1204, kSourceOsteology, 60293); // 0.92

    // Two independent sources agreeing that the capital existed: the case fusion is
    // for, and the one that has to raise a confidence rather than merely keep it.
    addFact(outCorpus, kSubjectCapital, kPredicateExists, kObjectTrue, 1200, 1250, kSourceCharter, 55050); // 0.84
    addFact(outCorpus, kSubjectCapital, kPredicateExists, kObjectTrue, 1200, 1250, kSourceSurvey, 45875);  // 0.70

    // A settlement only the chronicler mentions, weakly. Nothing corroborates it, so it
    // stays a SCORED claim — something the run has to earn rather than something it is
    // handed. Without a claim of this kind the canonical case would score zero whatever
    // the run did, and a signature that is zero for every possible reason is a
    // signature that cannot detect a regression.
    addFact(outCorpus, kSubjectOutpost, kPredicateExists, kObjectTrue, 1225, 1225, kSourceChronicler, 26214); // 0.40
}

void foldHistoryState(HistoryFoldResult &out)
{
    Corpus corpus;
    parityCorpus(corpus);
    foldHistoryCorpus(corpus, out);
}

void foldHistoryCorpus(const Corpus &corpus, HistoryFoldResult &out)
{
    out = HistoryFoldResult{};

    // ── The consensus world: listens to everyone ──────────────────────────────
    WorldView consensus;
    FusionReport report{};
    const Timeline timeline = buildTimeline(corpus, consensus, report);

    out.timelineSignature = timeline.fold(kFnv1aOffsetBasis);
    out.constraints = timeline.size();
    out.contradictions = report.contradictions;
    out.demoted = report.demoted;

    // What the consensus believes killed him, and whether the loser survived. Both
    // matter: SIM-022's claim is not "the better source wins" but "the better source
    // wins AND the worse one is still there".
    math::Fixed32 best{};
    bool minorityPresent = false;
    for (core::u32 i = 0u; i < timeline.size(); ++i)
    {
        const Constraint &c = timeline.at(i);
        if (c.fact.predicate != kPredicateDiedOf)
            continue;
        if (c.confidence > best)
        {
            best = c.confidence;
            out.consensusObject = c.fact.object;
        }
        if (c.fact.object == kObjectBattle)
            minorityPresent = true;
    }
    out.minorityReachable = minorityPresent ? 1u : 0u;

    // ── The minority world: listens to the chronicler alone ───────────────────
    //
    // Not a fork of the code and not a different pipeline: the same corpus, the same
    // function, one entry in admittedSources. That is what "possible worlds" means
    // here, and folding it separately is what proves the two are genuinely different
    // beliefs rather than the same one relabelled.
    WorldView minority;
    minority.admittedSources.push_back(kSourceChronicler);
    FusionReport minorityReport{};
    const Timeline minorityTimeline = buildTimeline(corpus, minority, minorityReport);
    out.minoritySignature = minorityTimeline.fold(kFnv1aOffsetBasis);

    // ── The run: a chronicle, and what it earned ──────────────────────────────
    //
    // The events are emitted the way a system would emit them, one per year boundary
    // of the era, so the chronicle has a shape a real run would produce. What is being
    // exercised is the SCORING rule, not a simulation: a constrained event must not
    // count towards agreement with the timeline that caused it.
    const Era era{1200, 1250, 4u};
    Chronicle chronicle;
    for (core::u32 tick = 0u; tick < era.totalTicks(); ++tick)
    {
        if (!era.isYearBoundary(tick))
            continue;
        const core::i32 year = era.yearOfTick(tick);

        core::u32 first = 0u;
        core::u32 count = 0u;
        if (timeline.constraintsOfYear(year, first, count))
        {
            for (core::u32 i = 0u; i < count; ++i)
            {
                const Constraint &c = timeline.at(first + i);
                if (c.kind == ConstraintKind::Score)
                    continue; // a scored claim is not handed to the run
                Attestation attestation;
                attestation.cause = Cause::Constraint;
                attestation.agent = c.fact.source;
                chronicle.record(c.fact, attestation);
            }
        }

        // One emergent event: the outpost the systems kept alive. This is the only kind
        // that can earn anything, and it is deliberately the subject NOTHING corroborated
        // — the run reconstructing a claim that rests on one weak source is exactly the
        // case a divergence score is supposed to reward.
        if (year == 1225)
        {
            Fact emergent;
            emergent.subject = kSubjectOutpost;
            emergent.predicate = kPredicateExists;
            emergent.object = kObjectTrue;
            emergent.fromYear = year;
            emergent.toYear = year;
            emergent.sigma = math::Fixed32::one();
            Attestation attestation;
            attestation.cause = Cause::Emergent;
            attestation.agent = 1u;
            chronicle.record(emergent, attestation);
        }
    }

    out.chronicleSignature = chronicle.fold(kFnv1aOffsetBasis);

    const Divergence verdict = measureDivergence(chronicle, timeline);
    out.scoredClaims = verdict.scoredClaims;
    out.earned = verdict.earned;
    out.divergenceScore = static_cast<core::u32>(verdict.score.raw());
}

} // namespace lpl::history
