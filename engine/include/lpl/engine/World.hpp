/**
 * @file World.hpp
 * @brief A self-contained, running game instance.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-07-21
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_WORLD_HPP
#    define LPL_ENGINE_WORLD_HPP

#    include <lpl/concurrency/IJobSystem.hpp>
#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/SystemScheduler.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/memory.hpp>

namespace lpl::platform {
class IPlatform;
} // namespace lpl::platform

namespace lpl::memory {
class ArenaAllocator;
} // namespace lpl::memory

namespace lpl::engine {

class Config;
class Engine;
class ResourceManager;

/**
 * @struct WorldContext
 * @brief The engine services a World's hooks receive.
 *
 * A World owns its game state (registry, scheduler, spatial index); everything
 * that belongs to the host rather than the game — the platform, the shared asset
 * cache, the per-frame arena — is handed in here, so the World stays free of any
 * host or kernel header and the same World runs on a desktop and in ring 0.
 */
struct WorldContext {
    platform::IPlatform &platform; ///< Clock / display / input / GPU memory.
    ResourceManager &resources;    ///< Shared asset cache (load-once).
    memory::ArenaAllocator &arena; ///< Per-frame scratch; reset every frame.
    const Config &config;          ///< The active configuration.
    /// The hosting Engine, or nullptr when this World is hosted by a Server
    /// (which owns many Worlds and has no single Engine). Always check it.
    Engine *engine;
};

/**
 * @class World
 * @brief One running game instance: an entity Registry, the scheduler that
 *        steps it, an optional spatial index, and the game's lifecycle hooks.
 *
 * This is Flakkari's @c Game made self-contained and free of any singleton, and
 * it is the seam that replaced the old IApplication: the game IS the World, so
 * the game's entities live in the World's own registry instead of in a private
 * one the engine never sees. The engine hosts exactly one World (client / solo /
 * kernel); a server build owns many above the engine. The same type serves both
 * because it holds no global state and knows nothing about players or the
 * network — sessions and netcode live in the server layer.
 *
 * A game customises the World by subclassing and overriding the @c on* hooks
 * (default: an empty world that just runs its scheduler). The base is concrete
 * and instantiable on its own — an empty World the host can populate directly.
 *
 * Determinism: the registry is authoritative Fixed32 and folds bit-identically
 * across the Linux oracle and the kernel. @c onFixedStep is the authoritative
 * path; @c onRender is non-authoritative and must never feed state back.
 */
class World : public core::NonCopyable<World> {
public:
    World() : _jobSystem{}, _registry{}, _scheduler{_jobSystem} {}
    virtual ~World() = default;

    [[nodiscard]] ecs::Registry &registry() noexcept { return _registry; }
    [[nodiscard]] const ecs::Registry &registry() const noexcept { return _registry; }
    [[nodiscard]] ecs::SystemScheduler &scheduler() noexcept { return _scheduler; }

    /**
     * @brief Creates the spatial broad-phase index, if not already present.
     *
     * The spatial partition is a WORLD facility, not a network one: any game
     * with a large map (a solo Minecraft-like as much as a server) enables it,
     * and a game that does its own broad-phase (samples::CubePile keeps an
     * octree inside its physics backend) simply never calls this, so it costs
     * nothing. Idempotent — returns the existing index on repeat calls.
     *
     * @param cellSize Side length of a cubic cell (world units).
     * @param cellCapacity Spatial cells budgeted up front (see WorldPartition).
     */
    ecs::WorldPartition &enableSpatialPartition(math::Fixed32 cellSize,
                                                core::u32 cellCapacity = ecs::WorldPartition::kDefaultCellCapacity)
    {
        if (!_spatial)
            _spatial = lpl::pmr::make_unique<ecs::WorldPartition>(cellSize, cellCapacity);
        return *_spatial;
    }

    /**
     * @brief The spatial index, or nullptr if @ref enableSpatialPartition was never called.
     * @return The spatial partition, or nullptr if it was never created.
     */
    [[nodiscard]] ecs::WorldPartition *spatialPartition() noexcept { return _spatial.get(); }

    /**
     * @brief Finalise the system graph. Call once, after every system is
     *        registered on @ref scheduler and before the first tick.
     */
    [[nodiscard]] core::Expected<void> build() { return _scheduler.buildGraph(); }

    /**
     * @brief Advance the scheduler one step (the default @ref onFixedStep).
     * @param dt Delta time since the last frame.
     */
    void tick(core::f32 dt) { _scheduler.tick(dt); }

    // ---- Game lifecycle hooks (override in a subclass) -------------------- //

    /**
     * @brief Set the world up: populate the registry, register systems, load
     *        assets. Runs once before the loop.
     * @param context Host services (platform, resources, arena, …).
     */
    [[nodiscard]] virtual core::Expected<void> onInit(WorldContext &context)
    {
        (void) context;
        return {};
    }

    /**
     * @brief Advance the authoritative simulation one fixed step.
     * @param dt Fixed timestep in seconds. Default: run the scheduler.
     */
    virtual void onFixedStep(core::f32 dt) { tick(dt); }

    /**
     * @brief Draw the current state (non-authoritative). Default: nothing.
     * @param context Host services (for the display / input backends).
     * @param alpha Interpolation factor in [0,1) between the last two steps.
     */
    virtual void onRender(WorldContext &context, core::f64 alpha)
    {
        (void) context;
        (void) alpha;
    }

    /**
     * @brief Release game resources. Runs once at shutdown.
     */
    virtual void onShutdown() {}

    /**
     * @brief Human-readable name of the game.
     * @return The name of the game.
     */
    [[nodiscard]] virtual const char *name() const noexcept { return "World"; }

private:
    concurrency::InlineJobSystem _jobSystem;            ///< Backs the scheduler (declared first).
    ecs::Registry _registry;                            ///< Authoritative entity state.
    ecs::SystemScheduler _scheduler;                    ///< Steps the registry; takes _jobSystem.
    lpl::pmr::unique_ptr<ecs::WorldPartition> _spatial; ///< Optional broad-phase (on demand).
};

} // namespace lpl::engine

#endif // LPL_ENGINE_WORLD_HPP
