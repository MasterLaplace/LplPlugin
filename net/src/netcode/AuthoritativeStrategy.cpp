#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/netcode/AuthoritativeStrategy.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace lpl::net::netcode {

/** @brief Per-player input entry. */
struct InputEntry {
    core::u32 sequence;
    std::vector<core::byte> data;
};

struct AuthoritativeStrategy::Impl {
    core::u32 serverFrame{0};
    core::u32 lastAckedSequence{0};

    std::unordered_map<core::u32, std::vector<InputEntry>> pendingInputs; /**< Per-player queue, consumed each tick. */

    std::unordered_map<core::u32, std::vector<InputEntry>> consumedInputs; /**< Inputs the last tick consumed. */
};

AuthoritativeStrategy::AuthoritativeStrategy() : _impl{std::make_unique<Impl>()} {}

AuthoritativeStrategy::~AuthoritativeStrategy() = default;

core::Expected<void> AuthoritativeStrategy::onInputReceived(core::u32 playerId, std::span<const core::byte> inputData,
                                                            core::u32 sequence)
{
    if (sequence <= _impl->lastAckedSequence && _impl->lastAckedSequence > 0)
    {
        return {};
    }

    InputEntry entry;
    entry.sequence = sequence;
    entry.data.assign(inputData.begin(), inputData.end());

    _impl->pendingInputs[playerId].push_back(std::move(entry));

    auto &queue = _impl->pendingInputs[playerId];
    std::sort(queue.begin(), queue.end(),
              [](const InputEntry &a, const InputEntry &b) { return a.sequence < b.sequence; });

    return {};
}

core::Expected<void> AuthoritativeStrategy::onStateReceived(std::span<const core::byte> snapshotData,
                                                            core::u32 sequence)
{
    (void) snapshotData;
    (void) sequence;
    return {};
}

void AuthoritativeStrategy::tick(core::f32 dt)
{
    (void) dt;

    _impl->consumedInputs.clear();
    _impl->consumedInputs.swap(_impl->pendingInputs);

    _impl->serverFrame++;
    _impl->lastAckedSequence = _impl->serverFrame;
}

const char *AuthoritativeStrategy::name() const noexcept { return "AuthoritativeStrategy"; }

} // namespace lpl::net::netcode
