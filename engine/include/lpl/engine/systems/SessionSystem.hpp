/**
 * @file SessionSystem.hpp
 * @brief ECS system that handles client connections (server-side).
 *
 * Drains ConnectEvent queue, creates player entities, sends MSG_WELCOME,
 * and registers the client endpoint for future broadcasts.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_SESSIONSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_SESSIONSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/input/InputManager.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class SessionSystem
 * @brief Processes connection events and creates player entities (server).
 */
class SessionSystem final : public ecs::ISystem {
public:
    /**
     * @param sessionManager Session registry.
     * @param queues         Event queues (reads connects).
     * @param transport      Transport to send welcome packets.
     * @param inputManager   Input manager to create per-entity input state.
     * @param world          World partition to spawn entities.
     * @param registry       ECS registry for entity creation.
     * @param sessionTimeoutMs Idle timeout after which a session and its avatar
     *                         are reaped; 0 disables timeout reaping.
     */
    SessionSystem(net::session::SessionManager &sessionManager, EventQueues &queues,
                  std::shared_ptr<net::transport::ITransport> transport, input::InputManager &inputManager,
                  ecs::WorldPartition &world, ecs::Registry &registry, core::f64 sessionTimeoutMs = 0.0);
    ~SessionSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief Handles this tick's departures, acks and arrivals.
     *
     * Departures first, so a slot freed this tick can be reused: explicit disconnects
     * (the client asked to leave), then the idle timeout — a client that stopped sending
     * is treated as gone. Input is the heartbeat (InputProcessingSystem touches the
     * session every tick a client sends), so the timeout only fires for clients that
     * truly went silent. Snapshot acks then advance each client's confirmed baseline
     * sequence, so the AOI broadcast stops resending what that client has applied
     * (§6.2.5).
     *
     * Arrivals keep legacy parity (SessionManager::handleConnections): a client is known
     * by its endpoint, so a handshake from an address that already has a session is a
     * retransmission — common on UDP — and must NOT spawn a second player. The previous
     * check asked isDuplicate() about a freshly minted id, which by construction never
     * collides, so it could never fire. A new client gets an entity with the full
     * component set, as the legacy SessionManager::connect() built a full EntitySnapshot
     * (Position, Velocity, Mass, AABB, Health), spawned at {0, 10, 0} (the legacy
     * default) with zero velocity, unit mass, a {1, 2, 1} box, 100 health, and awake.
     *
     * ONE network identity: the entity's ECS id. The legacy keyed session, input and the
     * welcome on a single publicId; the modern port had split it into a separate player
     * counter (session/input) and the ECS id (broadcast/hash), so the input a client
     * sent — tagged with the welcome id — never matched the entity MovementSystem drives
     * (keyed by the ECS id), and the client could not find its own avatar in a snapshot.
     * Keying everything on entityId.raw() closes the loop: welcome → client input →
     * InputProcessing → Movement all name the same entity. When the session cannot be
     * opened, the entity is destroyed again: a session id collision would leak it.
     *
     * The avatar is bound to the session: it is the client's centre of interest.
     * AoiBroadcastSystem reads it back (Session::boundEntity) to know where to run the
     * radius query; without the bind the session has a null entity and AOI could not
     * place the client in the world. The welcome (HandshakeAck) then carries the
     * entity id.
     */
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_SESSIONSYSTEM_HPP
