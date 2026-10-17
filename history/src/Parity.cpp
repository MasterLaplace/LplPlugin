#include <lpl/history/Parity.hpp>

#include <lpl/history/Calendar.hpp>

#include <lpl/history/Fold.hpp>

#include <lpl/history/Chronicle.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Era.hpp>

namespace lpl::history {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;

/**
 * @brief Adds one fact to a corpus.
 *
 * @warning Years IN, days out, and the widening is the honest part: a medieval corpus really
 * does date its claims to a year, so the window it deserves is the whole of that year. A
 * helper that pretended to know the day would be inventing precision no charter has.
 *
 * @param corpus    Destination.
 * @param subject   Who.
 * @param predicate What of them.
 * @param object    The value.
 * @param firstYear Window start, as a YEAR.
 * @param lastYear  Window end, as a year.
 * @param source    Who says so.
 * @param sigmaRaw  Confidence, raw Q16.16.
 */
void addFact(Corpus &corpus, core::u32 subject, core::u32 predicate, core::u32 object, core::i32 firstYear,
             core::i32 lastYear, core::u32 source, core::i32 sigmaRaw)
{
    Fact fact;
    fact.subject = subject;
    fact.predicate = predicate;
    fact.object = object;
    fact.fromDay = firstDayOfYear(firstYear);
    fact.toDay = lastDayOfYear(lastYear);
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

    addSource(outCorpus, kSourceChronicler, SourceKind::Panegyric, 40u, 0u);
    addSource(outCorpus, kSourceOsteology, SourceKind::Archaeology, 800u, 1u);
    addSource(outCorpus, kSourceCharter, SourceKind::Notarial, 0u, 1u);
    addSource(outCorpus, kSourceSurvey, SourceKind::Administrative, 10u, 1u);

    addFact(outCorpus, kSubjectKing, kPredicateDiedOf, kObjectBattle, 1204, 1204, kSourceChronicler, 29491);
    addFact(outCorpus, kSubjectKing, kPredicateDiedOf, kObjectDysentery, 1204, 1204, kSourceOsteology, 60293);

    addFact(outCorpus, kSubjectCapital, kPredicateExists, kObjectTrue, 1200, 1250, kSourceCharter, 55050);
    addFact(outCorpus, kSubjectCapital, kPredicateExists, kObjectTrue, 1200, 1250, kSourceSurvey, 45875);

    addFact(outCorpus, kSubjectOutpost, kPredicateExists, kObjectTrue, 1225, 1225, kSourceChronicler, 26214);
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

    WorldView consensus;
    FusionReport report{};
    const Timeline timeline = buildTimeline(corpus, consensus, report);

    out.timelineSignature = timeline.fold(kFnv1aOffsetBasis);
    out.constraints = timeline.size();
    out.contradictions = report.contradictions;
    out.demoted = report.demoted;

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

    WorldView minority;
    minority.admittedSources.push_back(kSourceChronicler);
    FusionReport minorityReport{};
    const Timeline minorityTimeline = buildTimeline(corpus, minority, minorityReport);
    out.minoritySignature = minorityTimeline.fold(kFnv1aOffsetBasis);

    const Era era = Era::ofYears(1200, 1250, 4u);
    Chronicle chronicle;
    for (core::u32 tick = 0u; tick < era.totalTicks(); ++tick)
    {
        core::i32 spanFrom = 0;
        core::i32 spanTo = 0;
        era.spanOfTick(tick, spanFrom, spanTo);
        const core::i32 year = yearOfDay(spanFrom);

        core::u32 first = 0u;
        core::u32 count = 0u;
        if (timeline.constraintsStartingIn(spanFrom, spanTo, first, count))
        {
            for (core::u32 i = 0u; i < count; ++i)
            {
                const Constraint &c = timeline.at(first + i);
                if (c.kind == ConstraintKind::Score)
                    continue;
                Attestation attestation;
                attestation.cause = Cause::Constraint;
                attestation.agent = c.fact.source;
                chronicle.record(c.fact, attestation);
            }
        }

        if (year == 1225)
        {
            Fact emergent;
            emergent.subject = kSubjectOutpost;
            emergent.predicate = kPredicateExists;
            emergent.object = kObjectTrue;
            emergent.fromDay = spanFrom;
            emergent.toDay = spanFrom;
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
