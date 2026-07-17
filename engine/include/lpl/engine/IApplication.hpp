/**
 * @file IApplication.hpp
 * @brief Injected application/simulation payload seam.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-17
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_IAPPLICATION_HPP
#    define LPL_ENGINE_IAPPLICATION_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>

namespace lpl::platform {
class IPlatform;
} // namespace lpl::platform

namespace lpl::ecs {
class Registry;
} // namespace lpl::ecs

namespace lpl::memory {
class ArenaAllocator;
} // namespace lpl::memory

namespace lpl::engine {

class Config;
class Engine;

/**
 * @struct AppContext
 * @brief The engine services handed to an application payload.
 *
 * Everything an application may touch, and nothing else: it reaches the host
 * or the kernel only through @c platform, exactly as the engine does. Owned by
 * the engine and guaranteed to outlive the application.
 */
struct AppContext {
    platform::IPlatform &platform; ///< Clock / display / input / GPU memory.
    ecs::Registry &registry;       ///< The engine's entity registry.
    memory::ArenaAllocator &arena; ///< Per-frame scratch; reset every frame.
    const Config &config;          ///< The active configuration.
    Engine &engine;                ///< Owner; for requestShutdown().
};

/**
 * @class IApplication
 * @brief Strategy interface for the game/simulation the engine runs.
 *
 * The second dependency-injection seam, alongside platform::IPlatform: the
 * platform says *where* the engine runs, the application says *what* it runs.
 * The engine owns the loop, the ECS and the subsystems but holds no game logic,
 * so the same payload runs unchanged on a desktop host and in the freestanding
 * kernel, and swapping the game means swapping the injected object.
 *
 * Determinism: @ref fixedStep is the authoritative path and must stay Fixed32
 * and bit-identical across targets. @ref render is non-authoritative (float)
 * and must never feed a result back into simulation state.
 */
class IApplication {
public:
    virtual ~IApplication() = default;

    /**
     * @brief Set up the payload.
     * @param context Engine services; outlives the application.
     * @return Success, or the error that prevented start-up.
     */
    [[nodiscard]] virtual core::Expected<void> init(AppContext &context) = 0;

    /**
     * @brief Advance the authoritative simulation by one fixed step.
     * @param dt Fixed timestep in seconds; never varies at runtime.
     */
    virtual void fixedStep(core::f32 dt) = 0;

    /**
     * @brief Draw the current state. Non-authoritative.
     * @param alpha Interpolation factor in [0,1) between the last two steps.
     */
    virtual void render(core::f64 alpha) = 0;

    /** @brief Release payload resources. */
    virtual void shutdown() {}

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::engine

#endif // LPL_ENGINE_IAPPLICATION_HPP
