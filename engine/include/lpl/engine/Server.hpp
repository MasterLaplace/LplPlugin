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
#        include <lpl/engine/EventQueue.hpp>
#        include <lpl/engine/ResourceManager.hpp>
#        include <lpl/net/session/SessionManager.hpp>
#        include <lpl/serial/ReplayRecorder.hpp>
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
     *
     * Runs the instance's onInit, gives it the server-side systems and resolves
     * its schedule. @ref init must have succeeded first: the systems are wired
     * to the shared transport.
     *
     * @param world The instance; ownership transfers. Its onInit runs here.
     * @return Its id, or @ref kInvalidWorldId if the server is not initialised,
     *         the instance failed to initialise, or its schedule has a cycle.
     */
    [[nodiscard]] WorldId addWorld(lpl::pmr::unique_ptr<World> world);

    /**
     * @brief Stop hosting an instance and destroy it (its onShutdown runs).
     * @return true if @p id referred to a live instance.
     */
    bool removeWorld(WorldId id);

    /**
     * @brief The instance behind @p id, or nullptr if it is gone.
     */
    [[nodiscard]] World *world(WorldId id) noexcept;

    /**
     * @brief Number of live instances.
     */
    [[nodiscard]] core::usize worldCount() const noexcept;

    /**
     * @brief Advance every hosted instance by one fixed step.
     *
     * Receives first, then runs the netcode, then steps the instances — the
     * order the legacy server's schedule had (NetworkReceive was its first
     * PreSwap system). The instance steps run in parallel when
     * Config::serverWorkerThreads is non-zero, sequentially otherwise.
     *
     * @param dt Fixed timestep in seconds.
     */
    void tick(core::f32 dt);

    /**
     * @brief Run until @ref requestShutdown (drives @ref tick).
     */
    void run();

    /**
     * @brief Ask the loop to stop after the current iteration.
     */
    void requestShutdown() noexcept;

    /**
     * @brief Close the session and destroy every instance.
     */
    void shutdown();

    /**
     * @brief The shared, load-once asset cache handed to every World.
     */
    [[nodiscard]] ResourceManager &resources() noexcept;

    /**
     * @brief The event queues packets addressed to @p id are decoded into.
     *
     * Each instance gets its own set, so one game's inputs can never reach
     * another's systems. Wire an instance's networking systems (session, input
     * processing, broadcast) to these.
     *
     * @return The queues, or nullptr if @p id is not a live instance.
     */
    [[nodiscard]] EventQueues *queues(WorldId id) noexcept;

    /**
     * @brief The connected clients of instance @p id.
     *
     * Sessions are per-instance, not per-server: broadcasting the world state
     * walks the sessions of ONE manager, so sharing a manager across instances
     * would send game A's state to game B's players. Flakkari kept its player
     * list inside each Game for the same reason.
     *
     * @return The session manager, or nullptr if @p id is not a live instance.
     */
    [[nodiscard]] net::session::SessionManager *sessions(WorldId id) noexcept;

    /**
     * @brief Outcome of comparing a client's reported digest (see §6.4).
     */
    enum class DesyncVerdict {
        Match,      ///< The client agrees with us on that tick.
        Diverged,   ///< Same tick, different state: this client is desynced.
        TickUnknown ///< That tick has fallen out of our history, or is ahead.
    };

    /**
     * @brief Digest of instance @p id's authoritative state at the last tick.
     * @return The digest, or 0 if @p id is not a live instance.
     */
    [[nodiscard]] core::u64 stateHash(WorldId id) const noexcept;

    /**
     * @brief Compare a digest a client reported for one of ITS past ticks.
     *
     * A client's report always describes a tick we have already stepped, so the
     * comparison cannot use the current digest: the server keeps a short
     * history and looks the tick up in it. A verdict of Diverged means that
     * client's simulation no longer matches ours — a determinism bug, memory
     * corruption, or a cheat — and the book's answer is to force a full
     * resynchronisation of that client.
     *
     * @param id     The instance the client is playing in.
     * @param tick   The tick the client hashed.
     * @param digest What the client computed for it.
     */
    [[nodiscard]] DesyncVerdict checkClientHash(WorldId id, core::u64 tick, core::u64 digest) const noexcept;

    /**
     * @brief Ticks of digest history kept per instance for @ref checkClientHash.
     */
    static constexpr core::usize kStateHashHistory = 128;

    /**
     * @brief How many client reports have disagreed with us so far.
     *
     * Non-zero means at least one client's simulation diverged from the
     * authoritative one — a determinism bug, corruption, or a cheat.
     */
    [[nodiscard]] core::u64 desyncCount() const noexcept;

    /**
     * @brief How many client reports have agreed with us.
     */
    [[nodiscard]] core::u64 matchedReportCount() const noexcept;

    /**
     * @brief How many reports named a tick outside our history.
     */
    [[nodiscard]] core::u64 staleReportCount() const noexcept;

    /**
     * @brief How many ticks failed to drain the socket within their budget.
     *
     * Non-zero means the server is receiving faster than it processes: the
     * kernel's RX buffer is absorbing the overflow and will drop past its
     * limit. Raise Config::maxPacketsPerTick, add receive threads (RSS), or shed
     * clients. Zero means the receive path is keeping up. See §6.1.
     */
    [[nodiscard]] core::u64 backpressureEvents() const noexcept;

    /**
     * @brief The most recent tick at which backpressure was observed, or 0.
     */
    [[nodiscard]] core::u64 lastBackpressureTick() const noexcept;

    /**
     * @brief Snapshots kept for instance @p id, for post-mortem diagnosis.
     *
     * Populated only when Config::replaySnapshotInterval is non-zero.
     *
     * @return The recorder, or nullptr if @p id is not a live instance.
     */
    [[nodiscard]] const serial::ReplayRecorder *replay(WorldId id) const noexcept;

    /**
     * @brief Number of ticks this server has stepped its instances.
     */
    [[nodiscard]] core::u64 currentTick() const noexcept;

    /**
     * @brief The instance new clients are placed in (the first one hosted,
     *        unless overridden). kInvalidWorldId while none is hosted.
     */
    [[nodiscard]] WorldId defaultWorld() const noexcept;

    /**
     * @brief Choose which instance new clients join.
     */
    void setDefaultWorld(WorldId id) noexcept;

    /**
     * @brief Bind a sender to an instance, so its later packets route there.
     * @return false if @p id is not a live instance.
     */
    bool routeSenderToWorld(const net::Endpoint &sender, WorldId id);

    /**
     * @brief The instance a sender is bound to, or kInvalidWorldId.
     */
    [[nodiscard]] WorldId worldForSender(const net::Endpoint &sender) const noexcept;

private:
    /**
     * @brief Drain the shared transport and fan each packet out to its instance.
     */
    void pumpNetwork();

    /**
     * @brief Give a freshly hosted instance the server-side systems.
     *
     * Session, input processing, movement, physics, broadcast and monitoring,
     * each bound to THIS instance's queues, sessions and input state — the
     * legacy server's PreSwap/PostSwap chain, minus NetworkReceiveSystem, which
     * @ref pumpNetwork replaces (N instances must not race for one socket).
     */
    void registerInstanceSystems(WorldId id);

    /**
     * @brief Fold each instance's authoritative state into its digest history.
     */
    void recordStateHashes();

    /**
     * @brief Compare every client digest report decoded this tick.
     */
    void consumeStateHashReports();

    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine

#    endif // LPL_HAS_NET

#endif // LPL_ENGINE_SERVER_HPP
