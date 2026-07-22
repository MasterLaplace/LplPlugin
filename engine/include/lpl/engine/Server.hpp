/**
 * @file Server.hpp
 * @brief Multi-instance game server: owns N Worlds and the shared net session.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-21
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SERVER_HPP
#    define LPL_ENGINE_SERVER_HPP

#    include <lpl/core/Platform.hpp>

// Multi-instance hosting is a SERVER BUILD concern. A client, a solo game and
// the kernel each run exactly one World and never pay for any of this; only a
// build with a sockets stack compiles the server layer at all.
#    ifdef LPL_HAS_NET

#        include <lpl/core/Expected.hpp>
#        include <lpl/core/NonCopyable.hpp>
#        include <lpl/core/Types.hpp>
#        include <lpl/engine/Config.hpp>
#        include <lpl/engine/ResourceManager.hpp>
#        include <lpl/engine/World.hpp>
#        include <lpl/std/memory.hpp>
#        include <lpl/std/vector.hpp>

namespace lpl::engine {

/**
 * @class Server
 * @brief Hosts many game instances at once, over one shared network session.
 *
 * This is Flakkari's GameManager, minus the singletons: it owns the transport,
 * the session manager and the netcode strategy, plus N Worlds — and each World
 * may be a different game, because a World is just an instance of "some game"
 * (see engine::World). Instances are addressed by an opaque id handed out at
 * registration.
 *
 * The asset cache is owned HERE and shared by reference with every World, so a
 * texture used by ten instances of the same game is loaded once. It is
 * thread-safe, which matters as soon as instances are ticked in parallel.
 *
 * Division of labour, deliberately: the Server routes and keeps sessions; a
 * World simulates and knows nothing about players or sockets. That keeps a
 * World's state purely deterministic and foldable, whether it runs here, in a
 * solo client, or in ring 0.
 */
class Server final : public core::NonCopyable<Server> {
public:
    /** @brief Identifies a hosted instance; never reused while it is alive. */
    using WorldId = core::u32;

    /** @brief Value returned by @ref addWorld when the instance was refused. */
    static constexpr WorldId kInvalidWorldId = ~WorldId{0};

    explicit Server(Config config);
    ~Server();

    /**
     * @brief Open the shared network session (transport, sessions, netcode).
     * @return Success, or the error that prevented the server from listening.
     */
    [[nodiscard]] core::Expected<void> init();

    /**
     * @brief Host one more game instance.
     * @param world The instance; ownership transfers. Its onInit runs here.
     * @return Its id, or @ref kInvalidWorldId if it failed to initialise.
     */
    [[nodiscard]] WorldId addWorld(lpl::pmr::unique_ptr<World> world);

    /**
     * @brief Stop hosting an instance and destroy it (its onShutdown runs).
     * @return true if @p id referred to a live instance.
     */
    bool removeWorld(WorldId id);

    /** @brief The instance behind @p id, or nullptr if it is gone. */
    [[nodiscard]] World *world(WorldId id) noexcept;

    /** @brief Number of live instances. */
    [[nodiscard]] core::usize worldCount() const noexcept;

    /**
     * @brief Advance every hosted instance by one fixed step.
     * @param dt Fixed timestep in seconds.
     */
    void tick(core::f32 dt);

    /** @brief Run until @ref requestShutdown (drives @ref tick). */
    void run();

    /** @brief Ask the loop to stop after the current iteration. */
    void requestShutdown() noexcept;

    /** @brief Close the session and destroy every instance. */
    void shutdown();

    /** @brief The shared, load-once asset cache handed to every World. */
    [[nodiscard]] ResourceManager &resources() noexcept;

private:
    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine

#    endif // LPL_HAS_NET

#endif // LPL_ENGINE_SERVER_HPP
