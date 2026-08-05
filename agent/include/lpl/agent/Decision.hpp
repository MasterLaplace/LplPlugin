/**
 * @file Decision.hpp
 * @brief The one decision seam, in a form both worlds can express.
 *
 * Two seams grew here before this file existed, and they were the same idea twice.
 * `agent::IPlanner` chose the next call for a hosted demon; `mind::IReasoner` chose
 * the next act for one in ring 0. Alongside them, `ToolRegistry`+`Dispatcher` and
 * `mind::IActionSurface` were both "what can be done now, and do it". Four concepts,
 * two implementations each, and nothing keeping them in step — which is exactly how
 * `apps/mapview` drifted until it carried bugs the engine no longer had.
 *
 * "One is hosted, the other freestanding" was never a reason for two of anything in
 * this project, and it is worth being explicit about that because it is the mistake
 * this file corrects. `ITransport` has a socket implementation and a kernel one.
 * `IMemoryBackend` has a Linux backend and a HAL. `ArenaAllocator` is a SINGLE
 * implementation, and the difference between host and kernel is not which arena but
 * where its block comes from. This is the same shape.
 *
 * WHAT IS SHARED, and it is the part that is genuinely universal: an act is text, a
 * decision is "given what is possible now and what has happened, produce the next
 * act", and a world is "tell me what is possible, and carry it out". None of those
 * three needs a heap, a string class or a JSON parser to be stated.
 *
 * ⚠ WHAT IS NOT SHARED, said here rather than discovered later. The hosted planner
 * decides partly from a critic's structured findings and from the last recipe as a
 * JSON DOCUMENT that it parses and overlays. That is not an accident of an API the
 * way `sockaddr` was for `net::Endpoint` — it is genuinely unbounded data, and it
 * cannot cross into ring 0. So it stays where the encoding stays: at the host's edge,
 * exactly as `SocketTransport` is the only place that knows about `htons`. A finding
 * reaches a shared decision the way every other fact about the world does, as an
 * @ref ActKind::Observation in the transcript.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_DECISION_HPP
#    define LPL_LPL_AGENT_DECISION_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::agent {

/**
 * @brief Bytes one act may occupy.
 *
 * Sized for the LONGER of the two encodings rather than for the shorter, because a
 * ring-0 act is a bare verb of a dozen bytes while a hosted one is a whole tool call
 * in JSON. Picking the small number would have made the shared type unusable by one
 * of its two users, which is how a shared type becomes two types again.
 *
 * The cost in ring 0 is a bigger transcript array in BSS and nothing else: every fold
 * over a transcript hashes @ref Act::bytes and the bytes actually used, never the
 * capacity, so this number can change without moving a single signature.
 */
inline constexpr core::u32 kActBytes = 512u;

/// Bytes the "what is possible now" alphabet may occupy.
inline constexpr core::u32 kAvailableBytes = 512u;

/**
 * @enum ActKind
 * @brief What one line of a transcript is.
 */
enum class ActKind : core::u8 {
    Thought,     ///< The demon reasoning to itself.
    Action,      ///< A move on the world.
    Observation, ///< What the world, or a critic, said back.
    Answer,      ///< The demon addressing the sovereign, done.
    Question,    ///< The demon addressing the sovereign, stuck.
};

/**
 * @struct Act
 * @brief One line of a transcript.
 */
struct Act {
    ActKind kind{ActKind::Thought}; ///< What this line is.
    char text[kActBytes]{};         ///< Its words: a verb, or a whole call.
    core::u32 bytes{0u};            ///< How many are in use.
    core::u32 step{0u};             ///< Which round produced it.
};

/**
 * @struct DecisionContext
 * @brief Everything a decision is allowed to depend on that both worlds can state.
 */
struct DecisionContext {
    const char *available{nullptr};    ///< Space-separated names of what can be done now.
    core::u32 availableBytes{0u};      ///< Bytes of @ref available.
    const Act *transcript{nullptr};    ///< What has happened, including observations.
    core::u32 transcriptLines{0u};     ///< Lines in @ref transcript.
    const char *goal{nullptr};         ///< What the sovereign asked for; may be null.
    core::u32 goalBytes{0u};           ///< Bytes of @ref goal.
    core::u32 turn{0u};                ///< Round within this budget, from zero.
    core::u32 turnsRemaining{0u};      ///< Including this one.

    /**
     * The last stretch of the budget, where a conclusion is required.
     *
     * Passed rather than derived from @ref turnsRemaining so the caller owns the
     * policy: a budget of nine turns divided by ten is zero, and a decider that
     * computed its own final stretch would never reach one on a small budget.
     */
    bool mustConclude{false};

    /**
     * Whether the world says the goal has been met.
     *
     * Asked of the world rather than inferred, and passed rather than left out: a
     * demon that has FINISHED and one that is STUCK both have nothing left to try,
     * and without this they are indistinguishable — a cautious one then asks for help
     * at the end of a job it has just completed.
     */
    bool satisfied{false};
};

/**
 * @brief Does this alphabet offer that name?
 *
 * Shared because both sides ask it and the answer has to be the same: the hosted
 * planner refuses a critic's suggestion for a capability the world is not offering,
 * and the ring-0 reasoner builds its grammar from the same list. Two scanners would
 * be two chances to disagree about what a word boundary is.
 *
 * @param alphabet Space-separated names.
 * @param bytes    Its length.
 * @param name     The name to look for.
 * @param nameBytes Its length.
 * @return true when @p name appears as a whole word.
 */
[[nodiscard]] constexpr bool alphabetOffers(const char *alphabet, core::u32 bytes, const char *name,
                                            core::u32 nameBytes) noexcept
{
    if (alphabet == nullptr || name == nullptr || nameBytes == 0u)
        return false;

    core::u32 cursor = 0u;
    while (cursor < bytes)
    {
        while (cursor < bytes && alphabet[cursor] == ' ')
            ++cursor;
        const core::u32 start = cursor;
        while (cursor < bytes && alphabet[cursor] != ' ')
            ++cursor;
        if (cursor - start == nameBytes)
        {
            bool match = true;
            for (core::u32 i = 0u; i < nameBytes && match; ++i)
                match = alphabet[start + i] == name[i];
            if (match)
                return true;
        }
        if (cursor == start)
            break;
    }
    return false;
}

/**
 * @class IDecider
 * @brief What chooses the next act.
 *
 * The seam a model plugs into. It never touches the world: acting is
 * @ref IWorldSurface's job, and keeping the two apart is what lets a decider be
 * swapped for a language model without any of the safety moving.
 */
class IDecider {
public:
    virtual ~IDecider() = default;

    /**
     * @brief Chooses the next act.
     * @param context What the decision may depend on.
     * @return The act; an @ref ActKind::Answer or @ref ActKind::Question ends the turn.
     */
    [[nodiscard]] virtual Act decide(const DecisionContext &context) noexcept = 0;

    /// A word for which policy this is, for a log line.
    [[nodiscard]] virtual const char *name() const noexcept { return "decider"; }
};

/**
 * @class IWorldSurface
 * @brief What the world allows, asked afresh every step.
 */
class IWorldSurface {
public:
    virtual ~IWorldSurface() = default;

    /**
     * @brief Writes the names of what can be done right now.
     *
     * Asked every step rather than once, and that is the difference between a grammar
     * and a menu: an action that has stopped being possible must stop being spellable.
     *
     * @param out      Receives a space-separated alphabet.
     * @param capacity Room in @p out.
     * @return Bytes written.
     */
    virtual core::u32 available(char *out, core::u32 capacity) noexcept = 0;

    /**
     * @brief Carries out one act and reports what happened.
     *
     * @param action      The act's text.
     * @param actionBytes How many bytes of it.
     * @param report      Receives what the world said back.
     * @param capacity    Room in @p report.
     * @param reportBytes Receives how much was written.
     * @return true when the act was legal and was carried out.
     */
    virtual bool perform(const char *action, core::u32 actionBytes, char *report, core::u32 capacity,
                         core::u32 *reportBytes) noexcept = 0;

    /**
     * @brief Has the goal been met?
     *
     * Asked of the world rather than of the demon. A demon that decided for itself
     * when it was finished would be marking its own work.
     *
     * @return true when the loop may stop.
     */
    [[nodiscard]] virtual bool satisfied() const noexcept = 0;
};

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_DECISION_HPP
