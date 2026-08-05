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
    kSubjectKing = 1u,        ///< The king the two sources disagree about.
    kSubjectCapital = 2u,     ///< A settlement both agree existed.
    kSubjectOutpost = 3u,     ///< A settlement only the chronicler mentions.
    kPredicateDiedOf = 10u,   ///< What killed him.
    kPredicateExists = 11u,   ///< Whether a settlement is there.
    kObjectBattle = 20u,      ///< The chronicler's answer.
    kObjectDysentery = 21u,   ///< The osteologist's answer.
    kObjectTrue = 22u,        ///< For an existence claim.
    kSourceChronicler = 100u, ///< Wrote forty years later, to praise.
    kSourceOsteology = 101u,  ///< Read the bones, eight centuries later.
    kSourceCharter = 102u,    ///< A notarial act, written the same year.
    kSourceSurvey = 103u,     ///< An administrative survey, a decade later.
};

/**
 * @brief The canonical corpus both sides replay.
 *
 * Four facts and four sources, chosen so every mechanism the module has is exercised
 * once: a contradiction to resolve, an independent agreement to fuse, a strong source
 * that earns a Force and a weak one that only earns a Score.
 *
 * @param outCorpus Receives the corpus.
 */
void parityCorpus(Corpus &outCorpus);

/**
 * @struct HistoryFoldResult
 * @brief The signatures the kernel must reproduce.
 */
struct HistoryFoldResult {
    core::u32 timelineSignature{0u};  ///< Fold of the consensus timeline.
    core::u32 chronicleSignature{0u}; ///< Fold of the run's own account.
    core::u32 minoritySignature{0u};  ///< Fold of the timeline a single-source world believes.
    core::u32 constraints{0u};        ///< Constraints the consensus world holds.
    core::u32 contradictions{0u};     ///< Pairs that could not both hold.
    core::u32 demoted{0u};            ///< Claims a contradiction pushed down.
    core::u32 consensusObject{0u};    ///< What the consensus believes killed the king.
    core::u32 minorityReachable{0u};  ///< 1 when the losing claim is still in the timeline.
    core::u32 scoredClaims{0u};       ///< Claims the run had to earn.
    core::u32 earned{0u};             ///< Of those, the ones it did.
    core::u32 divergenceScore{0u};    ///< Raw Q16.16 of the score.
};

/**
 * @brief Builds the canonical world, runs it, and folds every stage.
 *
 * One function, called by the host oracle and by the kernel smoke.
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
 * @param corpus What to believe.
 * @param out    Receives the signatures.
 */
void foldHistoryCorpus(const Corpus &corpus, HistoryFoldResult &out);

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_PARITY_HPP
