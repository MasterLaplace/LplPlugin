// /////////////////////////////////////////////////////////////////////////////
/// @file InputManager.hpp
/// @brief Aggregates all input sources and produces per-tick snapshots.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/input/IInputSource.hpp>
#include <lpl/input/InputState.hpp>
#include <lpl/input/NeuralInputState.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>
#include <vector>

namespace lpl::input {

// /////////////////////////////////////////////////////////////////////////////
/// @class InputManager
/// @brief Façade that polls all registered input sources and merges results
///        into a unified InputState + optional NeuralInputState per tick.
// /////////////////////////////////////////////////////////////////////////////
class InputManager final : public core::NonCopyable<InputManager>
{
public:
    InputManager();
    ~InputManager();

    /// @brief Registers a new input source.
    void addSource(std::unique_ptr<IInputSource> source);

    /// @brief Initializes all registered sources.
    [[nodiscard]] core::Expected<void> init();

    /// @brief Polls all sources and produces the current-tick snapshot.
    [[nodiscard]] core::Expected<void> poll();

    /// @brief Shuts down all sources.
    void shutdown();

    /// @brief Returns the latest classical input state.
    [[nodiscard]] const InputState& currentState() const noexcept;

    /// @brief Returns the latest neural input state (may be un-validated).
    [[nodiscard]] const NeuralInputState& currentNeuralState() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::input
