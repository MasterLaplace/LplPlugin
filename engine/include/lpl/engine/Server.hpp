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

#    ifdef LPL_HAS_NET

#        include <lpl/core/Expected.hpp>
#        include <lpl/core/NonCopyable.hpp>
#        include <lpl/core/Types.hpp>
#        include <lpl/engine/Config.hpp>
#        include <lpl/engine/EventQueue.hpp>
#        include <lpl/engine/ResourceManager.hpp>
#        include <lpl/engine/World.hpp>
#        include <lpl/net/session/SessionManager.hpp>
#        include <lpl/serial/ReplayRecorder.hpp>
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
 *
 * Multi-instance hosting is a SERVER BUILD concern. A client, a solo game and the
 * kernel each run exactly one World and never pay for any of this; only a build with
 * a sockets stack (LPL_HAS_NET) compiles the server layer at all.
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
     * to the shared transport, and hosting before it would hand them a null socket
     * that only faults on the first tick, far from the call that caused it.
     *
     * No Engine hosts this World — the server does — so its context carries a null
     * engine handle, which WorldContext documents as legitimate. The schedule is
     * resolved once every system is registered (the game's in onInit, the server's
     * after it): without it the instance ticks an empty wave list and nothing it
     * registered would ever run. Engine::run builds its graph at the same point, and
     * the legacy server called core.buildSchedule() there too.
     *
     * @param world The instance; ownership transfers. Its onInit runs here.
     * @return Its id, or @ref kInvalidWorldId if the server is not initialised,
     *         the instance failed to initialise, or its schedule has a cycle.
     */
    [[nodiscard]] WorldId addWorld(lpl::pmr::unique_ptr<World> world);

    /**
     * @brief Stop hosting an instance and destroy it (its onShutdown runs).
     *
     * The World owns the systems, and those systems hold references to the queues,
     * sessions, input state and physics backend: the World is destroyed first. The
     * senders bound to it are unrouted, or they would route into a null slot.
     *
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
     * The shared socket is drained ONCE and each packet fanned out to its instance.
     * The instances never touch the transport: the server routes, a World simulates.
     * The digest reports decoded by the drain are checked before the instances
     * advance, so a verdict always refers to the tick history as the client saw it.
     *
     * Instances are independent — each owns its registry, scheduler, spatial index,
     * physics backend, queues, sessions and input state — so they can be stepped
     * concurrently. What they share is either thread-safe (the asset cache),
     * read-only for the whole step (the transport's fd; sendto is atomic per
     * datagram), or already done: the drain and the netcode run first,
     * single-threaded, so no worker touches the socket for reading.
     *
     * Checked under ThreadSanitizer (test-server-routing, 4 workers): every report
     * sits on the JobSystem's work-stealing transfer, never on instance state — no
     * Registry, WorldPartition, SessionManager, EventQueues or InputManager appears
     * in any of them. Running the same test with a single worker, where nothing can
     * be stolen, reports nothing at all, which pins those reports to
     * ChaseLevDeque::steal: it orders the handover with std::atomic_thread_fence,
     * which TSan does not model. That is a pre-existing property of the job system,
     * not of this fan-out — and it is why serverWorkerThreads defaults to 0
     * (sequential) rather than on.
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
     *
     * The workers stop before the instances they step are destroyed. Then the same
     * ordering rule as @ref removeWorld: the systems die with their World, and they
     * reference everything cleared afterwards.
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
        Match,      /**< The client agrees with us on that tick. */
        Diverged,   /**< Same tick, different state: this client is desynced. */
        TickUnknown /**< That tick has fallen out of our history, or is ahead. */
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
     * A history slot is reused every kStateHashHistory ticks, so holding the tick
     * number alongside the digest is what distinguishes "the tick the client means"
     * from "a much older tick that happens to land on the same slot".
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
     * @brief A captured divergence, for post-mortem diagnosis (§6.4.2).
     *
     * When a client's digest disagrees with ours for a tick still in history, the
     * server records the tick, both digests, and who reported it. Combined with
     * the periodic snapshot the server kept for that tick (@ref replay), this is
     * enough to reconstruct the divergent state and hunt the source — a rounding
     * error, a race, a logic bug, or a cheat.
     */
    struct DesyncReport {
        WorldId instance{0};       /**< Which instance the reporter was playing in. */
        core::u64 tick{0};         /**< The tick that diverged. */
        core::u64 serverDigest{0}; /**< The authoritative digest we held for it. */
        core::u64 clientDigest{0}; /**< What the client reported for it. */
        net::Endpoint source{};    /**< Who reported the divergence. */
    };

    /**
     * @brief The most recent divergence captured, if any.
     * @param out Filled with the latest DesyncReport on success.
     * @return false if no divergence has been recorded yet.
     */
    [[nodiscard]] bool lastDesyncReport(DesyncReport &out) const noexcept;

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
     *
     * It receives in bursts of one recvmmsg's worth. An unknown sender is placed in
     * the default instance — that is the join, and it is remembered so every later
     * packet from it goes to the same game; with nothing hosted yet the packet is
     * dropped rather than guessed. A full burst means the socket may still have more
     * and a short one means it ran dry, so the drain only keeps looping while the
     * bursts stay full.
     *
     * Stopping at the budget with the socket still returning full bursts means the
     * server did NOT keep up this tick, and the kernel's RX buffer is now absorbing
     * the overflow (and will silently drop past its limit): that tick counts as a
     * backpressure event, the metric that makes the otherwise invisible wall visible.
     * Its tick is recorded as the one about to be stepped, since the tick counter is
     * bumped only after the instances step.
     */
    void pumpNetwork();

    /**
     * @brief Give a freshly hosted instance the server-side systems.
     *
     * Session, input processing, movement, physics, broadcast and monitoring,
     * each bound to THIS instance's queues, sessions and input state — the
     * legacy server's PreSwap/PostSwap chain (NetworkReceive -> Session ->
     * InputProcessing -> Movement -> Physics, then Broadcast and the monitor after
     * the swap), minus NetworkReceiveSystem, which
     * @ref pumpNetwork replaces (N instances must not race for one socket, each
     * stealing datagrams meant for the others). The scheduler orders by phase, so
     * registration order is free.
     *
     * Every one of these systems indexes entities through the spatial grid, so the
     * instance gets one whether or not its game asked for a broad-phase. Its cell size
     * comes from fromFloat, NOT Fixed32{10}: the raw-representation constructor would
     * give a cell size of 10/65536 and put every entity in its own cell.
     *
     * Physics: the legacy server stepped the world every tick (its "Physics" PreSwap
     * system). A hosted World has no Engine to register it, so without this an
     * instance would receive inputs and broadcast a state that never moved.
     *
     * The broadcast is interest-managed when a radius is set and full otherwise. A
     * positive radius sends each client only what is near its avatar, as
     * spawn/despawn/delta — the O(clients × N) wall goes away (§2.6). Zero keeps the
     * full-state fallback, unchanged.
     */
    void registerInstanceSystems(WorldId id);

    /**
     * @brief Fold each instance's authoritative state into its digest history.
     *
     * Deliberately AFTER the fan-out has joined: every instance is at rest and this
     * walk is single-threaded, so it never races a step in flight. It also keeps a
     * periodic snapshot (§6.5), so a divergence reported later can be diagnosed
     * post-mortem against the state the server actually held.
     */
    void recordStateHashes();

    /**
     * @brief Compare every client digest report decoded this tick.
     *
     * A client's report names a tick it has already simulated, so this is where the
     * digest history earns its keep: checkClientHash looks that tick up rather than
     * comparing against the state held right now. A divergence is logged and counted:
     * the book's answer is a forced resynchronisation of that client, and acting on it
     * is the netcode strategy's call. It is also captured for post-mortem (§6.4.2):
     * the tick, both digests, and who reported it. The server's own full state for
     * that tick lives in its ReplayRecorder (replay(id)), so this is enough to
     * reconstruct and compare the two states after the fact. A report naming an
     * unknown tick is normal for a very late report, or a client ahead of the server.
     */
    void consumeStateHashReports();

    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine

#    endif // LPL_HAS_NET

#endif // LPL_ENGINE_SERVER_HPP
