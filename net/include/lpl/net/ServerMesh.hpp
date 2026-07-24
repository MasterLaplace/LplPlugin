/**
 * @file ServerMesh.hpp
 * @brief Multi-server mesh topology for horizontal scaling.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_SERVERMESH_HPP
#    define LPL_NET_SERVERMESH_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>
#    include <string>
#    include <vector>

namespace lpl::net {

/**
 * @struct MeshNode
 * @brief Describes a peer server in the mesh.
 */
struct MeshNode {
    core::u32 nodeId;
    std::string address;
    core::u16 port;
    bool alive{true};
};

/**
 * @class ServerMesh
 * @brief Manages inter-server communication for world-partition handoff
 *        and entity migration (server meshing, book §6.2.8).
 *
 * Each node owns a region of the world; an entity crossing a boundary is handed
 * off to the node that now owns its region (@ref migrateEntity), and nodes track
 * each other's liveness with periodic heartbeats (@ref heartbeat / @ref
 * onHeartbeatAck) so a dead peer is noticed and its regions can be reassigned.
 * Transport is the same UDP path the game clients use; only the packet types
 * differ (protocol::PacketType::NodeHeartbeat / EntityMigrate).
 */
class ServerMesh final : public core::NonCopyable<ServerMesh> {
public:
    /**
     * @brief Constructs a mesh with the given transport.
     * @param transport Reference to ITransport used for inter-node comms.
     */
    explicit ServerMesh(transport::ITransport &transport);
    ~ServerMesh();

    /**
     * @brief Registers a peer node.
     * @param node MeshNode to register.
     * @return Expected<void> indicating success or failure.
     */
    [[nodiscard]] core::Expected<void> addNode(MeshNode node);

    /**
     * @brief Removes a peer node.
     * @param nodeId ID of the node to remove.
     * @return Expected<void> indicating success or failure.
     */
    [[nodiscard]] core::Expected<void> removeNode(core::u32 nodeId);

    /**
     * @brief Sends entity migration data to a peer.
     * @param targetNodeId ID of the node to migrate the entity to.
     * @param entityData Serialized data of the entity to migrate.
     * @return Expected<void> indicating success or failure.
     */
    [[nodiscard]] core::Expected<void> migrateEntity(core::u32 targetNodeId, std::span<const core::byte> entityData);

    /**
     * @brief Sends a heartbeat to every known node and ages their liveness.
     *
     * Each call sends one NodeHeartbeat to each node and counts a missed beat
     * against it; a node that misses @ref kMaxMissedHeartbeats consecutive beats
     * without an ack is marked not alive, so its regions can be reassigned. An
     * ack (@ref onHeartbeatAck) resets that node's counter.
     */
    void heartbeat();

    /**
     * @brief Records that @p nodeId answered a heartbeat: reset its miss counter
     *        and mark it alive again.
     */
    void onHeartbeatAck(core::u32 nodeId) noexcept;

    /**
     * @brief Total heartbeats sent across all nodes, for telemetry/tests.
     * @return Number of heartbeats sent.
     */
    [[nodiscard]] core::u64 heartbeatsSent() const noexcept;

    /**
     * @brief Total entity migrations sent, for telemetry/tests.
     * @return Number of entity migrations sent.
     */
    [[nodiscard]] core::u64 migrationsSent() const noexcept;

    /**
     * @brief Consecutive missed heartbeats after which a node is declared dead.
     * @return Maximum number of missed heartbeats.
     */
    static constexpr core::u32 kMaxMissedHeartbeats = 3;

    /**
     * @brief Returns the list of known mesh nodes.
     * @return Span of known mesh nodes.
     */
    [[nodiscard]] std::span<const MeshNode> nodes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net

#endif // LPL_NET_SERVERMESH_HPP
