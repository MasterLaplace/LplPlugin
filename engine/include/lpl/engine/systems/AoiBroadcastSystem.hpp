/**
 * @file AoiBroadcastSystem.hpp
 * @brief Area-of-interest server broadcast: each client gets only its neighbours.
 *
 * Runs in the Network phase (after physics and the buffer swap), like
 * BroadcastSystem — but instead of sending every client the whole world, it
 * sends each client only the entities inside its interest radius, and as a
 * per-session delta: EntitySpawn for entities that just entered the radius,
 * EntityDestroy for those that left, StateDelta for those that stayed. That
 * breaks the O(clients × N) broadcast wall (see the book's §2.6).
 *
 * Non-authoritative: it changes only what each client receives, never the
 * folded authoritative state. The wire snapshot is float, converted from the
 * authoritative Fixed32 transform at this boundary.
 *
 * Server-side only. Selected over BroadcastSystem when Config::interestRadius is
 * positive; a zero radius keeps the full-broadcast fallback.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-23
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_AOIBROADCASTSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_AOIBROADCASTSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class AoiBroadcastSystem
 * @brief Per-client interest-managed state broadcast (server).
 */
class AoiBroadcastSystem final : public ecs::ISystem {
public:
    /**
     * @param sessionManager Active sessions; each session's bound entity is its
     *        centre of interest.
     * @param transport      Transport for sending (batched at end of tick).
     * @param world          Spatial index the radius query runs against.
     * @param registry       Registry the entity snapshots are read from.
     * @param interestRadius Radius around each client's avatar to send (world
     *        units). Must be positive; a zero radius means AOI is off and the
     *        Server registers BroadcastSystem instead.
     */
    AoiBroadcastSystem(net::session::SessionManager &sessionManager,
                       std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                       ecs::Registry &registry, math::Fixed32 interestRadius);
    ~AoiBroadcastSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_AOIBROADCASTSYSTEM_HPP
