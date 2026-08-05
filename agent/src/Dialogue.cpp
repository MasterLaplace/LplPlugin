/**
 * @file Dialogue.cpp
 * @brief Implementation of the master-to-demon channel.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Dialogue.hpp>

#include <utility>

namespace lpl::agent {

void Dialogue::offer(std::string_view text)
{
    Intent intent;
    intent.text = std::string{text};
    intent.sequence = _received++;
    _intents.push_back(std::move(intent));
}

std::optional<Intent> Dialogue::poll()
{
    if (_read >= _intents.size())
        return std::nullopt;
    // Read forward rather than erase from the front: the whole exchange stays
    // available afterwards, which is what makes a session auditable.
    return _intents[_read++];
}

void Dialogue::say(std::string_view text) { _replies.emplace_back(text); }

std::vector<std::string> Dialogue::drainReplies()
{
    std::vector<std::string> out;
    out.swap(_replies);
    return out;
}

} // namespace lpl::agent
