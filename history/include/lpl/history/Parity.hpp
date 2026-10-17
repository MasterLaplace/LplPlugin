/**
 * @file Parity.hpp
 * @brief The constexpr corpus + recipe both sides replay.
 *
 * Same corpus, same seed, same fold on the host oracle and in ring 0.
 *
 * The corpus is SIM-022's example, because it is the smallest case in which the whole
 * model has to work at once: a chronicler says the king died in battle with a
 * confidence of 0.45, osteology says dysentery with 0.92, and the two cannot both
 * hold. The consensus view has to take the second, and the first has to remain
 * reachable — a pipeline that deleted the loser would destroy the evidence for its own
 * decision, and how the myth was built could never be traced.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_PARITY_HPP
#    define LPL_LPL_HISTORY_PARITY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/PossibleWorld.hpp>

namespace lpl::history {

/**
 * @brief Identifiers the canonical corpus uses.
 *
 * Small integers standing for what LplKnowledge would intern: the strings live where
 * a corpus is curated, and a module that runs in ring 0 trades in identifiers.
 */
enum : core::u32 {
    kSubjectKing = 1u,        /**< The king the two sources disagree about. */
    kSubjectCapital = 2u,     /**< A settlement both agree existed. */
    kSubjectOutpost = 3u,     /**< A settlement only the chronicler mentions. */
    kPredicateDiedOf = 10u,   /**< What killed him. */
    kPredicateExists = 11u,   /**< Whether a settlement is there. */
    kObjectBattle = 20u,      /**< The chronicler's answer. */
    kObjectDysentery = 21u,   /**< The osteologist's answer. */
    kObjectTrue = 22u,        /**< For an existence claim. */
    kSourceChronicler = 100u, /**< Wrote forty years later, to praise. */
    kSourceOsteology = 101u,  /**< Read the bones, eight centuries later. */
    kSourceCharter = 102u,    /**< A notarial act, written the same year. */
    kSourceSurvey = 103u,     /**< An administrative survey, a decade later. */
};

/**
 * @brief The canonical corpus both sides replay.
 *
 * Five facts and four sources, chosen so every mechanism the module has is exercised
 * once: a contradiction to resolve, an independent agreement to fuse, a strong source
 * that earns a Force and a weak one that only earns a Score.
 *
 * The chronicler wrote forty years later and was praising someone; he says the king died
 * in battle, at 0.45. The osteologist read the bones, and says dysentery at 0.92. They
 * cannot both be right about what killed the king.
 *
 * A charter (0.84) and a survey (0.70) agree that the capital existed from 1200 to 1250:
 * two independent sources agreeing, the case fusion is for, and the one that has to raise
 * a confidence rather than merely keep it.
 *
 * The outpost is a settlement only the chronicler mentions, weakly (0.40, in 1225). Nothing
 * corroborates it, so it stays a SCORED claim — something the run has to earn rather than
 * something it is handed. Without a claim of this kind the canonical case would score zero
 * whatever the run did, and a signature that is zero for every possible reason is a
 * signature that cannot detect a regression.
 *
 * @param outCorpus Receives the corpus.
 */
void parityCorpus(Corpus &outCorpus);

/**
 * @struct HistoryFoldResult
 * @brief The signatures the kernel must reproduce.
 */
struct HistoryFoldResult {
    core::u32 timelineSignature{0u};  /**< Fold of the consensus timeline. */
    core::u32 chronicleSignature{0u}; /**< Fold of the run's own account. */
    core::u32 minoritySignature{0u};  /**< Fold of the timeline a single-source world believes. */
    core::u32 constraints{0u};        /**< Constraints the consensus world holds. */
    core::u32 contradictions{0u};     /**< Pairs that could not both hold. */
    core::u32 demoted{0u};            /**< Claims a contradiction pushed down. */
    core::u32 consensusObject{0u};    /**< What the consensus believes killed the king. */
    core::u32 minorityReachable{0u};  /**< 1 when the losing claim is still in the timeline. */
    core::u32 scoredClaims{0u};       /**< Claims the run had to earn. */
    core::u32 earned{0u};             /**< Of those, the ones it did. */
    core::u32 divergenceScore{0u};    /**< Raw Q16.16 of the score. */
};

/**
 * @brief Builds the canonical world, runs it, and folds every stage.
 *
 * One function, which the engine test `history` runs on both targets.
 *
 * @param out Receives the signatures.
 */
void foldHistoryState(HistoryFoldResult &out);

/**
 * @brief Runs and folds a GIVEN corpus, rather than the canonical one.
 *
 * The same pipeline, with the corpus as a parameter. @ref foldHistoryState is this
 * function applied to @ref parityCorpus, which is what keeps gate P13 exactly where it
 * was when this split was made.
 *
 * It exists because a second gate needs it: LplKnowledge bakes this very corpus into a
 * `.lplknow` image, reads it back, and has to show that the timeline built from the
 * decoded corpus is bit-for-bit the timeline built from the corpus in memory. Without a
 * corpus-taking entry point that check would have had to re-implement the pipeline, and a
 * round-trip test whose two sides run different code proves nothing about the round trip.
 *
 * **The consensus world** listens to everyone. What it believes killed the king, and whether
 * the loser survived, are both folded: SIM-022's claim is not "the better source wins" but
 * "the better source wins AND the worse one is still there".
 *
 * **The minority world** listens to the chronicler alone. Not a fork of the code and not a
 * different pipeline: the same corpus, the same function, one entry in admittedSources. That
 * is what "possible worlds" means here, and folding it separately is what proves the two are
 * genuinely different beliefs rather than the same one relabelled.
 *
 * **The run** is a chronicle, and what it earned. The events are emitted the way a system would
 * emit them, tick by tick across the era, so the chronicle has a shape a real run would
 * produce; a scored claim is not handed to the run. What is being exercised is the SCORING
 * rule, not a simulation: a constrained event must not count towards agreement with the
 * timeline that caused it.
 *
 * An emergent event, on each tick that starts in 1225, is the outpost the systems kept alive.
 * It is the only kind that can earn anything, and it is deliberately the subject NOTHING
 * corroborated — the run reconstructing a claim that rests on one weak source is exactly the
 * case a divergence score is supposed to reward. The run knows the DAY it emitted it, so it
 * says the day. A scored claim dated to a whole year still matches it, because the two
 * intervals overlap -- which is the asymmetry the unit exists for: sources are vague, a
 * simulation is not.
 *
 * @param corpus What to believe.
 * @param out    Receives the signatures.
 */
void foldHistoryCorpus(const Corpus &corpus, HistoryFoldResult &out);

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_PARITY_HPP
