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
