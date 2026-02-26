// /////////////////////////////////////////////////////////////////////////////
/// @file InputManager.cpp
/// @brief InputManager implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/input/InputManager.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::input {

struct InputManager::Impl
{
    std::vector<std::unique_ptr<IInputSource>> sources;
    InputState                                  state{};
    NeuralInputState                            neuralState{};
};

InputManager::InputManager()
    : impl_{std::make_unique<Impl>()}
{}

InputManager::~InputManager() = default;

void InputManager::addSource(std::unique_ptr<IInputSource> source)
{
    impl_->sources.push_back(std::move(source));
}

core::Expected<void> InputManager::init()
{
    for (auto& source : impl_->sources)
    {
        auto result = source->init();
        if (!result.has_value())
        {
            return result;
        }
    }
    return {};
}

core::Expected<void> InputManager::poll()
{
    for (auto& source : impl_->sources)
    {
        auto result = source->poll();
        if (!result.has_value())
        {
            core::Log::warn("InputManager: poll failed for source");
        }
    }
    ++impl_->state.sequence;
    return {};
}

void InputManager::shutdown()
{
    for (auto& source : impl_->sources)
    {
        source->shutdown();
    }
}

const InputState& InputManager::currentState() const noexcept
{
    return impl_->state;
}

const NeuralInputState& InputManager::currentNeuralState() const noexcept
{
    return impl_->neuralState;
}

} // namespace lpl::input
