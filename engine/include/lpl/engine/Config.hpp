/**
 * @file Config.hpp
 * @brief Engine configuration (Builder pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_ENGINE_CONFIG_HPP
#    define LPL_ENGINE_CONFIG_HPP

#    include <lpl/core/Constants.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/string.hpp>

namespace lpl::engine {

/** @brief Immutable engine configuration. */
class Config {
public:
    /** @brief Fluent builder for Config. */
    class Builder {
    public:
        Builder &tickRate(core::u32 hz) noexcept;
        Builder &maxEntities(core::u32 n) noexcept;
        Builder &maxChunks(core::u32 n) noexcept;
        Builder &worldCellCapacity(core::u32 n) noexcept;
        Builder &serverMode(bool enabled) noexcept;
        Builder &headless(bool enabled) noexcept;
        Builder &arenaSize(core::usize bytes) noexcept;
        Builder &worldArenaSize(core::usize bytes) noexcept;
        Builder &enableBci(bool enabled) noexcept;
        Builder &enablePhysics(bool enabled) noexcept;
        Builder &enableNetworking(bool enabled) noexcept;
        Builder &enableRendering(bool enabled) noexcept;
        Builder &enableRealTimeGuard(bool enabled) noexcept;
        Builder &enableGpu(bool enabled) noexcept;
        Builder &serverAddress(pmr::string addr) noexcept;
        Builder &serverPort(core::u16 port) noexcept;
        Builder &serverWorkerThreads(core::u32 n) noexcept;
        Builder &replaySnapshotInterval(core::u32 ticks) noexcept;
        Builder &maxPacketsPerTick(core::u32 n) noexcept;
        Builder &interestRadius(math::Fixed32 radius) noexcept;
        Builder &keyframeInterval(core::u32 ticks) noexcept;
        Builder &bandwidthBudgetBytes(core::u32 bytes) noexcept;
        Builder &lodNearRadius(math::Fixed32 radius) noexcept;
        Builder &lodFarInterval(core::u32 ticks) noexcept;
        Builder &sessionTimeoutMs(core::f64 ms) noexcept;

        [[nodiscard]] Config build() const noexcept;

    private:
        core::u32 _tickRate{core::kTickRate};
        core::u32 _maxEntities{core::kMaxEntities};
        core::u32 _maxChunks{core::kMaxChunks};
        core::u32 _worldCellCapacity{ecs::WorldPartition::kDefaultCellCapacity};
        core::u32 _serverWorkerThreads{0};
        core::u32 _replaySnapshotInterval{0};
        core::u32 _maxPacketsPerTick{256};
        core::u32 _keyframeInterval{60};
        core::u32 _bandwidthBudgetBytes{0};
        core::u32 _lodFarInterval{4};
        core::f64 _sessionTimeoutMs{30000.0};
        math::Fixed32 _interestRadius{math::Fixed32::zero()};
        math::Fixed32 _lodNearRadius{math::Fixed32::zero()};
        bool _serverMode{false};
        bool _headless{false};
        core::usize _arenaSize{64 * 1024 * 1024};
        core::usize _worldArenaSize{64 * 1024 * 1024};
        bool _enableBci{false};
        bool _enableGpu{false};
        bool _enablePhysics{true};
        bool _enableNetworking{true};
        bool _enableRendering{true};
        bool _enableRealTimeGuard{false};
        pmr::string _serverAddress{"127.0.0.1"};
        core::u16 _serverPort{4242};
    };

    [[nodiscard]] core::u32 tickRate() const noexcept { return _tickRate; }
    [[nodiscard]] core::u32 maxEntities() const noexcept { return _maxEntities; }
    [[nodiscard]] core::u32 maxChunks() const noexcept { return _maxChunks; }

    /**
     * @brief Returns the number of spatial cells budgeted by WorldPartition.
     * @return Number of spatial cells budgeted by WorldPartition.
     */
    [[nodiscard]] core::u32 worldCellCapacity() const noexcept { return _worldCellCapacity; }
    [[nodiscard]] bool serverMode() const noexcept { return _serverMode; }

    /**
     * @brief Worker threads used to tick hosted instances in parallel.
     *
     * Zero (the default) ticks them one after another on the calling thread.
     * Only engine::Server reads this; a client, a solo game and the kernel each
     * run exactly one instance and never fan out. Instances share nothing
     * mutable — queues, sessions, input state, registry, scheduler, spatial
     * index and physics backend are all per-instance — so a worker per instance
     * is sound; what they DO share (the asset cache, the socket, the log) is
     * either thread-safe or written once before the fan-out.
     *
     * @return Number of workers, or 0 for a sequential tick.
     */
    [[nodiscard]] core::u32 serverWorkerThreads() const noexcept { return _serverWorkerThreads; }

    /**
     * @brief Ticks between state snapshots kept for post-mortem diagnosis.
     *
     * Zero (the default) records nothing. When set, engine::Server stores a
     * snapshot every N ticks per instance, so a desync reported by a client can
     * be compared against the state the server actually held (§6.4.2).
     *
     * @return The interval in ticks, or 0 when snapshotting is off.
     */
    [[nodiscard]] core::u32 replaySnapshotInterval() const noexcept { return _replaySnapshotInterval; }

    /**
     * @brief Packets the server drains from its socket per tick, per receive.
     *
     * A hard cap is what keeps a receive burst bounded, but a cap fixed in the
     * source is an invisible ceiling: at N clients each sending an input a tick,
     * anything past maxPacketsPerTick per tick silently backs up in the kernel's
     * RX buffer. Raise it for a big server, and watch Server::backpressureEvents
     * to know whether it is high enough. Default 256.
     *
     * @return The per-tick receive budget.
     */
    [[nodiscard]] core::u32 maxPacketsPerTick() const noexcept { return _maxPacketsPerTick; }

    /**
     * @brief Interest radius for area-of-interest (AOI) broadcasting.
     *
     * Zero (the default) disables AOI: the server sends every client the full
     * world state each tick, the O(clients × N) fallback (systems::BroadcastSystem).
     * A positive radius switches an instance to systems::AoiBroadcastSystem, which
     * sends each client only the entities within @c interestRadius of its own
     * avatar, as spawn / despawn / delta — the lever that lets client count scale
     * past the broadcast wall (§2.6). Non-authoritative: it changes only what each
     * client receives, never the folded state.
     *
     * @return The radius in world units, or zero when AOI is off.
     */
    [[nodiscard]] math::Fixed32 interestRadius() const noexcept { return _interestRadius; }

    /**
     * @brief Ticks between full re-sends of an entity in a client's interest set.
     *
     * Between keyframes, the AOI broadcast sends an in-range entity as a field
     * delta: only the fields that changed against the last state that client was
     * told, so an unchanged field costs one bit of absence (§6.2.5). The keyframe
     * periodically re-sends the whole entity, so a delta lost on the unreliable
     * channel self-heals within this many ticks. 1 disables delta compression
     * (every send is full); higher trades staleness-on-loss for bandwidth. Only
     * used when @c interestRadius is positive. Default 60 (~1 s at 60 Hz).
     *
     * @return The keyframe interval in ticks.
     */
    [[nodiscard]] core::u32 keyframeInterval() const noexcept { return _keyframeInterval; }

    /**
     * @brief Per-client, per-tick byte budget for the AOI delta stream (§6.2.7).
     *
     * When positive, the entities that changed inside a client's interest radius
     * are ranked by a relevancy priority (proximity to the client's avatar plus
     * how long each has gone unsent) and only the highest-priority ones up to this
     * many bytes are sent that tick; the rest age and are sent on a later tick, so
     * nothing starves. This is the adaptive lever that caps a client's outbound
     * bandwidth. 0 (the default) means unlimited: every changed entity is sent.
     * Only used when @c interestRadius is positive.
     *
     * @return The budget in bytes, or 0 for unlimited.
     */
    [[nodiscard]] core::u32 bandwidthBudgetBytes() const noexcept { return _bandwidthBudgetBytes; }

    /**
     * @brief Radius of the full-rate near ring for network LOD (§6.2.6).
     *
     * Entities within this distance of a client's avatar replicate every tick;
     * those farther (but still inside @c interestRadius) replicate once every
     * @c lodFarInterval ticks. Zero (the default) disables LOD — every in-range
     * entity is full-rate. Only used when @c interestRadius is positive and this
     * is smaller than it.
     *
     * @return The near-ring radius in world units, or zero when LOD is off.
     */
    [[nodiscard]] math::Fixed32 lodNearRadius() const noexcept { return _lodNearRadius; }

    /**
     * @brief Update interval (ticks) for entities beyond the near ring (§6.2.6).
     * @return The far-ring update cadence in ticks (clamped to >= 1 in use).
     */
    [[nodiscard]] core::u32 lodFarInterval() const noexcept { return _lodFarInterval; }

    /**
     * @brief Idle timeout after which a client session (and its avatar) is reaped.
     *
     * A client that sends nothing for this long is treated as gone: its session,
     * its entity, its input state and its spatial-index entry are all removed, so
     * a disconnect does not leak an avatar that lingers and is broadcast forever.
     * The input stream is the heartbeat (a connected client sends its input every
     * tick). Zero disables timeout reaping. Default 30 s.
     *
     * @return The idle timeout in milliseconds, or 0 when reaping is off.
     */
    [[nodiscard]] core::f64 sessionTimeoutMs() const noexcept { return _sessionTimeoutMs; }
    [[nodiscard]] bool headless() const noexcept { return _headless; }

    /**
     * @brief Per-frame scratch arena; reset every frame.
     * @return Size in bytes of the per-frame scratch arena.
     */
    [[nodiscard]] core::usize arenaSize() const noexcept { return _arenaSize; }

    /**
     * @brief Persistent arena backing the World's ECS storage. NEVER reset while the
     *        World lives — chunks allocated from it must outlive the frame.
     * @return Size in bytes of the persistent arena backing the World's ECS storage.
     */
    [[nodiscard]] core::usize worldArenaSize() const noexcept { return _worldArenaSize; }
    [[nodiscard]] bool enableBci() const noexcept { return _enableBci; }

    /**
     * @brief Tests whether physics is enabled.
     * @return True if physics is enabled, false otherwise.
     */
    [[nodiscard]] bool enablePhysics() const noexcept { return _enablePhysics; }
    /**
     * @brief Tests whether networking is enabled.
     * @return True if networking is enabled, false otherwise.
     */
    [[nodiscard]] bool enableNetworking() const noexcept { return _enableNetworking; }
    /**
     * @brief Tests whether rendering is enabled.
     * @return True if rendering is enabled, false otherwise.
     */
    [[nodiscard]] bool enableRendering() const noexcept { return _enableRendering; }

    /**
     * @brief Runs the authoritative tick inside a platform real-time section, where a
     *        backend that enforces it makes heap allocation FAIL. Off by default: turn
     *        it on only once the tick is known to be allocation-free, or the first
     *        allocation takes the process (or the kernel) down.
     * @return True if the real-time guard is enabled, false otherwise.
     */
    [[nodiscard]] bool enableRealTimeGuard() const noexcept { return _enableRealTimeGuard; }
    [[nodiscard]] bool enableGpu() const noexcept { return _enableGpu; }
    [[nodiscard]] const pmr::string &serverAddress() const noexcept { return _serverAddress; }
    [[nodiscard]] core::u16 serverPort() const noexcept { return _serverPort; }

private:
    friend class Builder;

    core::u32 _tickRate{core::kTickRate};
    core::u32 _maxEntities{core::kMaxEntities};
    core::u32 _maxChunks{core::kMaxChunks};
    core::u32 _worldCellCapacity{ecs::WorldPartition::kDefaultCellCapacity};
    core::u32 _serverWorkerThreads{0};
    core::u32 _replaySnapshotInterval{0};
    core::u32 _maxPacketsPerTick{256};
    core::u32 _keyframeInterval{60};
    core::u32 _bandwidthBudgetBytes{0};
    core::u32 _lodFarInterval{4};
    core::f64 _sessionTimeoutMs{30000.0};
    math::Fixed32 _interestRadius{math::Fixed32::zero()};
    math::Fixed32 _lodNearRadius{math::Fixed32::zero()};
    bool _serverMode{false};
    bool _headless{false};
    core::usize _arenaSize{64 * 1024 * 1024};
    core::usize _worldArenaSize{64 * 1024 * 1024};
    bool _enableBci{false};
    bool _enableGpu{false};
    bool _enablePhysics{true};
    bool _enableNetworking{true};
    bool _enableRendering{true};
    bool _enableRealTimeGuard{false};
    pmr::string _serverAddress{"127.0.0.1"};
    core::u16 _serverPort{4242};
};

} // namespace lpl::engine

#endif // LPL_ENGINE_CONFIG_HPP
