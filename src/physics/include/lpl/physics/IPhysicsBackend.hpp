// /////////////////////////////////////////////////////////////////////////////
/// @file IPhysicsBackend.hpp
/// @brief Abstract physics backend interface (Strategy pattern).
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

namespace lpl::physics {

// /////////////////////////////////////////////////////////////////////////////
/// @class IPhysicsBackend
/// @brief Strategy interface for physics step implementations.
///
/// Concrete backends:
///   - @c CpuPhysicsBackend  — single-threaded CPU reference.
///   - GPU backends via the @c gpu module (CUDA / Vulkan Compute).
// /////////////////////////////////////////////////////////////////////////////
class IPhysicsBackend
{
public:
    virtual ~IPhysicsBackend() = default;

    /// @brief Initializes the backend (allocates buffers, etc.).
    [[nodiscard]] virtual core::Expected<void> init() = 0;

    /// @brief Executes one fixed-tick physics step.
    /// @param dt Delta-time in seconds (typically 1/144).
    [[nodiscard]] virtual core::Expected<void> step(core::f32 dt) = 0;

    /// @brief Tears down the backend.
    virtual void shutdown() = 0;

    /// @brief Returns a human-readable name for this backend.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::physics
