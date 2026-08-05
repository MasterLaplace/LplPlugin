/**
 * @file Transcript.hpp
 * @brief The record of a reason-act-observe loop.
 *
 * Serialised, it is a recipe: replaying it must rebuild the same world and fold
 * the same signature. That makes an agent session auditable and, more importantly,
 * testable.
 *
 * ── What this is NOT ─────────────────────────────────────────────────────────
 *
 * It is not a second journal. @c editor::CommandJournal already records every act
 * that changed the world, already replays them, and already declines to record a
 * query. Writing the acts down again here would give two histories that agree
 * until the day they do not — the duplication this project has paid for more than
 * once.
 *
 * So a turn stores what the journal has no room for — the THOUGHT that led to the
 * act, and the OBSERVATION that came back — plus the journal index the act landed
 * at. Rebuilding the world is the journal's job; explaining it is this one's.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_TRANSCRIPT_HPP
#    define LPL_LPL_AGENT_TRANSCRIPT_HPP

#    include <lpl/agent/ToolCall.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>
#    include <string_view>
#    include <vector>

namespace lpl::agent {

/// A turn whose act changed nothing, so the journal never recorded it.
inline constexpr core::u32 kNotJournalled = 0xFFFFFFFFu;

/**
 * @struct Turn
 * @brief One reason-act-observe step.
 */
struct Turn {
    core::u32 index{0u};                    ///< 0-based turn number.
    std::string thought;                    ///< Why, in the caller's own words. May be empty.
    std::string tool;                       ///< Which capability was invoked.
    std::string args;                       ///< Its arguments, as JSON.
    std::string observation;                ///< What came back: a report, or a refusal.
    bool ok{false};                         ///< Whether the act was accepted.
    core::u32 journalEntry{kNotJournalled}; ///< Where CommandJournal recorded it.
};

/**
 * @class Transcript
 * @brief The turns of one session, in order.
 */
class Transcript {
public:
    /**
     * @brief Appends a turn.
     * @param call         The validated call that was made.
     * @param observation  What came back.
     * @param ok           Whether it was accepted.
     * @param journalEntry Journal size after the act, or @ref kNotJournalled.
     */
    void record(const ToolCall &call, std::string observation, bool ok, core::u32 journalEntry);

    /**
     * @brief Appends a turn for a call that never validated, so it has no ToolDesc.
     * @param attempted The name of the tool that was attempted.
     * @param reason The reason for the refusal.
     */
    void recordRefusal(std::string_view attempted, std::string reason);

    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_turns.size()); }
    [[nodiscard]] const std::vector<Turn> &turns() const noexcept { return _turns; }
    [[nodiscard]] const Turn *last() const noexcept { return _turns.empty() ? nullptr : &_turns.back(); }

    /**
     * @brief How many times the most recent (tool, args) pair has just repeated.
     *
     * The anti-loop guard the research calls for: an agent that reissues the same
     * call and gets the same answer is not converging, it is stuck, and the only
     * useful thing left to do is stop. Counts the trailing run, so an earlier,
     * unrelated repetition does not arm it.
     */
    [[nodiscard]] core::u32 trailingRepeats() const noexcept;

    /**
     * @brief The session as a JSON document.
     * @return The JSON string representing the transcript.
     */
    [[nodiscard]] std::string toJson() const;

    void clear() noexcept { _turns.clear(); }

private:
    std::vector<Turn> _turns;
};

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_TRANSCRIPT_HPP
