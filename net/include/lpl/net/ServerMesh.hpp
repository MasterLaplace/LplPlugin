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
    #define LPL_NET_SERVERMESH_HPP

#include <lpl/net/transport/ITransport.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>
#include <lpl/core/Expected.hpp>

#include <memory>
#include <string>
#include <vector>

namespace lpl::net {

/**
 * @struct MeshNode
 * @brief Describes a peer server in the mesh.
 */
struct MeshNode
{
    core::u32   nodeId;
    std::string address;
    core::u16   port;
    bool        alive{true};
};

/**
 * @class ServerMesh
 * @brief Manages inter-server communication for world-partition handoff
 *        and entity migration.
 */
class ServerMesh final : public core::NonCopyable<ServerMesh>
{
public:
    /**
     * @brief Constructs a mesh with the given transport.
     * @param transport Reference to ITransport used for inter-node comms.
     */
    explicit ServerMesh(transport::ITransport& transport);
    ~ServerMesh();

    /** @brief Registers a peer node. */
    [[nodiscard]] core::Expected<void> addNode(MeshNode node);

    /** @brief Removes a peer node. */
    [[nodiscard]] core::Expected<void> removeNode(core::u32 nodeId);

    /** @brief Sends entity migration data to a peer. */
    [[nodiscard]] core::Expected<void> migrateEntity(
        core::u32 targetNodeId,
        std::span<const core::byte> entityData);

    /** @brief Runs a heartbeat / health-check cycle. */
    void heartbeat();

    /** @brief Returns the list of known mesh nodes. */
    [[nodiscard]] std::span<const MeshNode> nodes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net

#endif // LPL_NET_SERVERMESH_HPP
