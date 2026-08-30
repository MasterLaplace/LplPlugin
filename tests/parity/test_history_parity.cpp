/**
 * @file test_history_parity.cpp
 * @brief SIM-022 end to end: the consensus wins, and the myth stays traceable.
 *
 * The example is small on purpose and it is the whole model at once. A chronicler
 * says the king died in battle; osteology says dysentery. They cannot both hold. The
 * default view must take the second -- and the first must still be THERE, because a
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

#include <lpl/engine/systems/Journey.hpp>
#include <lpl/history/Calendar.hpp>
#include <lpl/history/Chronicle.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Era.hpp>
#include <lpl/history/Parity.hpp>
#include <lpl/history/PossibleWorld.hpp>
#include <lpl/math/Geo.hpp>

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

    // ── The distance belongs to the CLAIM, not to the source ──────────────────
    std::printf("\n-- one chronicle, two claims, two distances --\n");
    {
        // @warning **What a single number on the source cannot express.** A chronicle compiled around
        // 1200 recounts an event of 1190 and one of 400. It is ten years from the first and eight
        // hundred from the second, and scoring it once is right about at most one of them.
        history::SourceProfile compiler;
        compiler.kind = history::SourceKind::Chronicle;
        compiler.composedFrom = history::firstDayOfYear(1195);
        compiler.composedTo = history::lastDayOfYear(1205);

        const core::u32 near = history::distanceInYears(compiler, history::firstDayOfYear(1190));
        const core::u32 far = history::distanceInYears(compiler, history::firstDayOfYear(400));
        std::printf("    from its own century: %u years, from antiquity: %u years\n", near, far);
        check(near < 20u, "it is close to what it witnessed");
        check(far > 700u, "and far from what it repeats");

        const math::Fixed32 onRecent = history::trustworthiness(compiler, history::firstDayOfYear(1190));
        const math::Fixed32 onAncient = history::trustworthiness(compiler, history::firstDayOfYear(400));
        check(onRecent > onAncient, "so the same chronicle is worth more about the first than the second");

        // @warning Measured from the NEAREST edge, so a source is given the benefit of its own
        // uncertainty about when it was written. Widening what is known about a manuscript must
        // never make the claims inside it score better -- and taking the far edge would do
        // exactly that in reverse, punishing a source for the vagueness of its own dating.
        history::SourceProfile vague = compiler;
        vague.composedFrom = history::firstDayOfYear(1150);
        check(history::trustworthiness(vague, history::firstDayOfYear(1190)) <=
                  history::trustworthiness(compiler, history::firstDayOfYear(1190)),
              "a wider composition window does not improve the score");

        // A source that states no window falls back on whatever distance it declares, so a corpus
        // that knows nothing new folds exactly as it did -- which is why the signatures above did
        // not move.
        history::SourceProfile stated;
        stated.kind = history::SourceKind::Chronicle;
        stated.yearsAfterEvent = 40u;
        check(history::distanceInYears(stated, history::firstDayOfYear(1000)) == 40u,
              "a stated distance is used when no window is given");
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
                folded.consensusObject == history::kObjectDysentery ?
                    "dysentery (the bones)" :
                    (folded.consensusObject == history::kObjectBattle ? "battle (the chronicler)" : "?"));
    std::printf("    contradictions=%u demoted=%u constraints=%u\n", folded.contradictions, folded.demoted,
                folded.constraints);

    check(folded.contradictions == 1u, "the two accounts of his death are seen to contradict");
    check(folded.consensusObject == history::kObjectDysentery, "and the consensus view takes the bones");
    check(folded.demoted == 1u, "the chronicler's account is pushed down");
    check(folded.minorityReachable == 1u,
          "and is STILL THERE -- a deleted loser would destroy the evidence for the decision");

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
        claim.fromDay = 1300;
        claim.toDay = 1300;
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

    // ── The gearing must not reach the history ────────────────────────────────
    std::printf("\n-- the same corpus at four different clock rates --\n");
    {
        // @warning **The claim the whole unit change exists to make.** A constraint used to fire when
        // the current year EQUALLED a fact's year, which is right only while the clock advances
        // one year at a time. Gear an era to cross a silent century in a step -- which is what a
        // gearing is FOR -- and every constraint inside that step was silently never applied,
        // producing a chronicle that looked complete. And firing on every overlapping tick would
        // have the opposite fault: 365 firings at a daily rate against one at a yearly one, so
        // the same corpus would fold differently depending on how fast it was read.
        //
        // So: exactly once, and the same set, at every rate.
        history::Timeline asked;
        const core::i32 kYears[] = {1200, 1201, 1207, 1250, 1299};
        for (core::u32 i = 0u; i < 5u; ++i)
        {
            history::Fact fact;
            fact.subject = 1000u + i;
            fact.predicate = 11u;
            fact.object = 1u;
            fact.fromDay = history::firstDayOfYear(kYears[i]);
            fact.toDay = history::lastDayOfYear(kYears[i]);
            fact.sigma = math::Fixed32::half();
            history::Constraint constraint;
            constraint.fact = fact;
            constraint.kind = history::ConstraintKind::Seed;
            asked.add(constraint);
        }
        asked.finalise();

        // Day by day, week by week, year by year, and a whole century in one step.
        const core::u32 kRates[] = {1u, 7u, 365u, 36525u};
        core::u32 firedPerRate[4] = {0u, 0u, 0u, 0u};
        bool everyConstraintFiredExactlyOnce = true;

        for (core::u32 r = 0u; r < 4u; ++r)
        {
            history::Era era;
            era.startDay = history::firstDayOfYear(1200);
            era.endDay = history::lastDayOfYear(1299);
            era.daysPerTick = kRates[r];

            core::u32 timesFired[5] = {0u, 0u, 0u, 0u, 0u};
            for (core::u32 tick = 0u; tick < era.totalTicks(); ++tick)
            {
                core::i32 spanFrom = 0;
                core::i32 spanTo = 0;
                era.spanOfTick(tick, spanFrom, spanTo);
                core::u32 first = 0u;
                core::u32 count = 0u;
                if (!asked.constraintsStartingIn(spanFrom, spanTo, first, count))
                    continue;
                for (core::u32 i = 0u; i < count; ++i)
                {
                    ++firedPerRate[r];
                    ++timesFired[asked.at(first + i).fact.subject - 1000u];
                }
            }
            for (core::u32 i = 0u; i < 5u; ++i)
            {
                if (timesFired[i] != 1u)
                    everyConstraintFiredExactlyOnce = false;
            }
            std::printf("    %6u days/tick : %u ticks, %u firings\n", kRates[r], era.totalTicks(), firedPerRate[r]);
        }

        check(firedPerRate[0] == 5u && firedPerRate[1] == 5u && firedPerRate[2] == 5u && firedPerRate[3] == 5u,
              "every rate applies all five constraints, including one that crosses a century in a step");
        check(everyConstraintFiredExactlyOnce, "and applies each exactly once, at every rate");
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

    // ── Gate P20 `journey` -- a body that walks, and what it earns ────────────
    std::printf("-- gate P20 journey: signatures the kernel must reproduce --\n");
    {
        lpl::engine::systems::JourneyFoldResult journey{};
        lpl::engine::systems::foldJourneyState(journey);

        std::printf("  journey_chronicle = 0x%08X\n", journey.chronicleSignature);
        std::printf("  journey_position  = 0x%08X\n", journey.positionSignature);
        std::printf("  journey_deed      = 0x%08X\n", journey.deedSignature);
        std::printf("  seeded            = %u\n", journey.seeded);
        std::printf("  forced            = %u\n", journey.forced);
        std::printf("  unplaceable       = %u\n", journey.unplaceable);
        std::printf("  arrivals          = %u\n", journey.arrivals);
        std::printf("  journey_scored    = %u\n", journey.scoredClaims);
        std::printf("  journey_earned    = %u\n", journey.earned);
        std::printf("  journey_score     = 0x%08X\n", journey.divergenceScore);
        std::printf("  journey_first     = %u\n", journey.firstArrival);
        std::printf("  journey_routed    = %u\n", journey.routedLegs);
        std::printf("  journey_avoided   = %u\n", journey.avoided);
        std::printf("  journey_road      = 0x%08X\n", journey.roadSignature);
        std::printf("  journey_waypoints = 0x%08X\n", journey.waypointSignature);
        std::printf("  journey_roadcells = %u\n", journey.roadCells);
        std::printf("  journey_roadpairs = %u\n", journey.roadPairs);
        std::printf("  journey_altchron  = 0x%08X\n", journey.alternateChronicle);
        std::printf("  journey_altarr    = %u\n", journey.alternateArrivals);
        std::printf("  journey_closedchron = 0x%08X\n", journey.closedChronicle);
        std::printf("  journey_closedarr = %u\n", journey.closedArrivals);
        std::printf("  journey_closedfirst = %u\n", journey.closedFirst);
        std::printf("  journey_wrappedroad = %u\n", journey.wrappedRoadCells);
        std::printf("  journey_polarroad = %u\n", journey.polarRoadCells);
        std::printf("  journey_altfirst  = %u\n", journey.alternateFirst);
        std::printf("  journey_cascaderoad = 0x%08X\n", journey.cascadeRoadSignature);
        std::printf("  journey_cascadecells = %u\n", journey.cascadeRoadCells);
        std::printf("  journey_cascadecoarse = %u\n", journey.cascadeCoarseExpanded);
        std::printf("  journey_cascadecorridor = %u\n", journey.cascadeCorridorCells);

        // @warning A constraint has to put somebody in the world, or everything below measures nothing.
        check(journey.seeded == 1u, "a constraint seeds a body");
        // @warning The claim of the whole gate: the run produced events NOBODY told it to produce.
        // Zero here and every signature below would still be stable and would prove nothing.
        check(journey.arrivals >= 2u, "and the body goes places on its own");
        // Both directions, deliberately: one claim the walk reaches and one it misses by two
        // years. A fixture where nothing is earned passes against a body that never moves; one
        // where everything is earned passes against a body teleported to its destination.
        check(journey.earned >= 1u, "some scored claims are earned");
        check(journey.earned < journey.scoredClaims, "and not all of them");
        // A place the gazetteer cannot locate must not become a position: the fixture holds one
        // such place, and nothing may walk to it.
        check(journey.unplaceable == 0u, "an unlocatable place is never walked to");
        // @warning Facts beat geometry. The corpus attests a road from the birthplace to place 302
        // while 301 sits nearer and unlinked, so this single number separates a walk that read
        // the corpus from one that only measured distances -- and every signature above is stable
        // either way.
        check(journey.firstArrival == 302u, "the walk follows the attested road, not the nearest place");
        // @warning And it follows the road's SHAPE, not just its destination. The fixture's route
        // doglegs north before turning east, so a body that ignored the resolver would reach the
        // same place having been somewhere else -- same arrivals, same first, different ground
        // covered. Without this number the gate proves roads exist and not that anybody walks one.
        check(journey.routedLegs >= 2u, "and walks the road's waypoints rather than the straight line");
        // @warning The ground is the only hard constraint, and until it was wired in the walk went
        // straight through rock. The fixture puts a boulder on the road, so a walk that consults
        // the ground and one that does not reach the same places by different paths -- and only
        // this number separates them.
        check(journey.avoided >= 1u, "and turns aside from ground it cannot cross");
        check(journey.divergenceScore == (65536u * journey.earned) / journey.scoredClaims,
              "the divergence score is the earned fraction");

        // @warning The attested link becomes ONE road, not one per direction: a resolver hands links
        // out both ways, and laying the pair twice would give a second road that took the first
        // one's own discount.
        check(journey.roadPairs == 1u, "the attested link becomes exactly one road");
        // @warning DERIVED from the fixture's geometry, not read off a run: the ridge fills columns
        // 0..39 of its row, so crossing it without climbing means reaching column 40 -- 28
        // columns east of the endpoints, the crossing cell, then 28 back west, with all the
        // northward travel absorbed into diagonals. 57 cells is therefore the shortest detour
        // that EXISTS, so this asserts the router found the provably optimal road rather than
        // merely a bent one. Measured against its own control: with the climb made free the same
        // fixture lays 28 cells, the straight line through the mountain -- which would fold to a
        // perfectly stable signature on both targets.
        check(journey.roadCells == 57u, "and it takes the shortest road that goes round the ridge");
        check(journey.roadSignature != 0u && journey.waypointSignature != 0u, "the road and its waypoints both fold");

        // @warning **Two worlds, and the assertion is that they DIFFER.** The same corpus, the same
        // seed, the same systems -- one entry more in `admittedSources`, and a body born
        // somewhere else lives a different life. If these folded alike, "possible worlds" would
        // be a label rather than a mechanism, which is exactly what the P13 minority signature
        // asserts of a timeline and this asserts of a WALK.
        check(journey.alternateChronicle != 0u, "the alternate world was walked at all");
        check(journey.alternateChronicle != journey.chronicleSignature,
              "and the world according to one source is a genuinely different world");
        // @warning It still has to be a real run, not an empty one: a world where nobody moves also
        // folds differently, and would satisfy the line above while proving nothing.
        check(journey.alternateArrivals >= 1u, "with a body that actually went somewhere");

        // @warning **The closed world must fold DIFFERENTLY, or `wrapWidth` did nothing.** Closing a
        // world changes every distance the walk measures, so a body that ignored it would produce
        // this exact chronicle again -- and would in truth be walking the long way round a planet,
        // on a road that looks like any other road.
        check(journey.closedChronicle != journey.chronicleSignature,
              "a closed world folds differently from an open one");
        check(journey.closedArrivals >= 1u, "and its body actually went somewhere too");

        // @warning **A closed routing grid must lay a SHORTER road**, because the two attested places
        // sit either side of its seam. A router that could not cross it paves most of the way round
        // instead -- a valid road, a long road, and nothing downstream can tell it was the wrong one.
        check(journey.wrappedRoadCells < journey.roadCells, "a closed grid paves a shorter road than an open one");
        check(journey.wrappedRoadCells > 0u, "and it paves one at all");

        // @warning And opening the poles must not make it WORSE. A road getting longer when a shortcut
        // is offered is what an inadmissible estimate looks like from outside: A* discards the cheap
        // route before evaluating it and hands back a plausible one.
        check(journey.polarRoadCells <= journey.wrappedRoadCells, "and offering the poles never makes it worse");

        // @warning **The cascade must lay the SAME road.** Planning on a summary and refining inside
        // the corridor it opens is a different search, so agreement is a property to prove -- and
        // proving it on a desktop proves it for a desktop, which is why both numbers cross to
        // ring 0. A summary that hid the pass would return a road that is found, valid, and 2.65
        // times the cost of the real one, without ever widening to say so (measured in
        // test-terrain-routes).
        check(journey.cascadeRoadSignature == journey.roadSignature,
              "a road planned coarse and refined fine is the road found flat");
        check(journey.cascadeRoadCells == journey.roadCells, "cell for cell");
        // @warning **Without this the two lines above are satisfied by no cascade at all.** A run that
        // fell back to a flat search folds an identical road signature and an identical count. Only
        // a coarse plan settling cells says the summary was built, planned on, and refined.
        check(journey.cascadeCoarseExpanded > 0u, "and a coarse plan actually ran");
        check(journey.cascadeCorridorCells > 0u, "opening a corridor for the fine search");
    }

    std::printf("-- Magellan: on a closed world the near place is the one across the seam\n");
    {
        // @warning **The check that stops `wrapWidth` being an inert field.** A body choosing "the
        // nearest place" with a plain subtraction sets off around the planet the long way, and that
        // is a perfectly ordinary voyage -- nothing downstream can tell it went the wrong way.
        const math::Fixed32 width = math::Fixed32::fromFloat(1000.0f);
        const math::Fixed32 west = math::Fixed32::fromFloat(995.0f);
        const math::Fixed32 east = math::Fixed32::fromFloat(5.0f);

        const math::Fixed32 openGap = math::foldOntoShorterWay(math::Fixed32{}, east - west);
        check(openGap < math::Fixed32::fromFloat(-900.0f), "an open world sees them far apart");

        const math::Fixed32 closedGap = math::foldOntoShorterWay(width, east - west);
        check(closedGap > math::Fixed32::zero() && closedGap < math::Fixed32::fromFloat(11.0f),
              "a closed world sees them as neighbours");

        check(math::foldOntoShorterWay(width, west - east) < math::Fixed32::zero(), "and the crossing has a direction");

        const math::Fixed32 inland = math::Fixed32::fromFloat(120.0f);
        check(math::foldOntoShorterWay(width, inland).raw() == inland.raw(), "an ordinary separation is left alone");

        // @warning **A separation can be WIDER than the world, and folding once is not enough.** A gap
        // of 300 across a world 120 wide came back as 180 -- still most of the way round, and wrong
        // in exactly the direction that sends a body the long way. `shortestDelta` never meets this
        // because it wraps both ends first; a caller holding two raw positions, as `planarDistance`
        // does, meets it immediately.
        const math::Fixed32 narrow = math::Fixed32::fromFloat(120.0f);
        const math::Fixed32 threeLaps = math::foldOntoShorterWay(narrow, math::Fixed32::fromFloat(300.0f));
        check(threeLaps.toInt() == 60, "a gap wider than the world folds all the way down");
        check(threeLaps > math::Fixed32::zero(), "and keeps its side");
        check(math::foldOntoShorterWay(narrow, math::Fixed32::fromFloat(-300.0f)).toInt() == -60,
              "the same going the other way");

        // @warning The SAME rule the cell-space caller uses, checked rather than assumed: two
        // implementations of "how far apart" would let the distance that CHOOSES a destination
        // disagree with the one that walks to it.
        math::GlobeWrap globe{};
        globe.columns = 1000;
        globe.rows = 400;
        core::i32 cellDx = 0;
        core::i32 cellDz = 0;
        math::shortestDelta(globe, 995, 0, 5, 0, cellDx, cellDz);
        check(cellDx == 10 && closedGap.toInt() == 10, "and cells agree with world units");
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
