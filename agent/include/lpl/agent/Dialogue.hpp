/**
 * @file Dialogue.hpp
 * @brief The master-to-demon channel.
 *
 * Distinct from tools on purpose. Tools are how the demon acts on its world;
 * dialogue is how it speaks to the sovereign, who is outside that world and is the
 * only thing in the system it cannot predict.
 *
 * That distinction is why an intent is not a tool call. A tool call is validated
 * against a grammar and either runs or is refused; an intent is a sentence, and
 * what to do about it is the demon's problem. Collapsing the two would make the
 * sovereign one more caller of the API — which is exactly what they are not.
 *
 * Transport-agnostic on purpose: intents arrive from a file, a pipe, a socket or a
 * ring buffer, and none of that changes what an intent IS. @ref Dialogue holds the
 * queue; something else fills it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_DIALOGUE_HPP
#    define LPL_LPL_AGENT_DIALOGUE_HPP

#    include <lpl/core/Types.hpp>

#    include <optional>
#    include <string>
#    include <string_view>
#    include <vector>

namespace lpl::agent {

/**
 * @struct Intent
 * @brief One thing the sovereign asked for, in their own words.
 */
struct Intent {
    std::string text;      ///< "make a world with rivers and a village"
    core::u32 sequence{0u}; ///< Order received, so a reply can name what it answers.
};

/**
 * @class Dialogue
 * @brief A two-way channel of sentences.
 */
class Dialogue {
public:
    /**
     * @brief Queues something the sovereign said.
     * @param text The text of the intent.
     */
    void offer(std::string_view text);

    /**
     * @brief Takes the next intent, if there is one.
     * @return The intent, or nothing when the sovereign has said nothing new.
     */
    [[nodiscard]] std::optional<Intent> poll();

    /**
     * @brief Queues something to say back.
     * @param text The text of the reply.
     */
    void say(std::string_view text);

    /**
     * @brief Gets all replies said back since the last @ref drainReplies.
     * @return A reference to the vector of replies.
     */
    [[nodiscard]] const std::vector<std::string> &replies() const noexcept { return _replies; }

    /**
     * @brief Takes the replies and forgets them.
     * @return The vector of replies.
     */
    [[nodiscard]] std::vector<std::string> drainReplies();

    [[nodiscard]] core::u32 pending() const noexcept { return static_cast<core::u32>(_intents.size() - _read); }
    [[nodiscard]] core::u32 received() const noexcept { return _received; }

private:
    std::vector<Intent> _intents;
    std::vector<std::string> _replies;
    std::size_t _read{0u};
    core::u32 _received{0u};
};

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_DIALOGUE_HPP
