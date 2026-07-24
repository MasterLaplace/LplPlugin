/**
 * @file ServerMesh.cpp
 * @brief ServerMesh implementation stub.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/Endpoint.hpp>
#include <lpl/net/ServerMesh.hpp>
#include <lpl/net/protocol/Protocol.hpp>

#include <algorithm>
#include <cstring>

namespace lpl::net {

struct ServerMesh::Impl {
    transport::ITransport &transport;
    std::vector<MeshNode> nodes;
    std::vector<core::u32> missedBeats; ///< Index-aligned with @c nodes.
    std::vector<core::byte> scratch;    ///< Reused outbound packet buffer.
    core::u64 heartbeatsSent{0};
    core::u64 migrationsSent{0};

    explicit Impl(transport::ITransport &t) : transport{t} {}

    /// Index of @p nodeId in @c nodes, or -1.
    [[nodiscard]] core::isize indexOf(core::u32 nodeId) const
    {
        for (core::usize i = 0; i < nodes.size(); ++i)
            if (nodes[i].nodeId == nodeId)
                return static_cast<core::isize>(i);
        return -1;
    }

    /// Builds header + payload into @c scratch and sends it to @p node.
    [[nodiscard]] core::Expected<void> sendTo(const MeshNode &node, protocol::PacketType type,
                                              std::span<const core::byte> payload)
    {
        Endpoint endpoint{};
        if (!Endpoint::parse(node.address.c_str(), node.port, endpoint))
            return core::makeError(core::ErrorCode::InvalidArgument, "Mesh node has an unparseable address");

        scratch.resize(sizeof(protocol::PacketHeader) + payload.size());
        auto &header = *reinterpret_cast<protocol::PacketHeader *>(scratch.data());
        header.magic = protocol::kProtocolMagic;
        header.version = protocol::kProtocolVersion;
        header.type = type;
        header.flags = 0;
        header.padding = 0;
        header.sequence = 0;
        header.payloadSize = static_cast<core::u32>(payload.size());
        if (!payload.empty())
            std::memcpy(scratch.data() + sizeof(header), payload.data(), payload.size());

        auto sent = transport.send(std::span<const core::byte>{scratch.data(), scratch.size()}, &endpoint);
        if (!sent.has_value())
            return core::makeError(sent.error().code(), sent.error().message());
        return {};
    }
};

ServerMesh::ServerMesh(transport::ITransport &transport) : _impl{std::make_unique<Impl>(transport)} {}

ServerMesh::~ServerMesh() = default;

core::Expected<void> ServerMesh::addNode(MeshNode node)
{
    for (const auto &n : _impl->nodes)
    {
        if (n.nodeId == node.nodeId)
        {
            return core::makeError(core::ErrorCode::AlreadyExists, "Node already registered");
        }
    }
    _impl->nodes.push_back(std::move(node));
    _impl->missedBeats.push_back(0);
    return {};
}

core::Expected<void> ServerMesh::removeNode(core::u32 nodeId)
{
    const auto idx = _impl->indexOf(nodeId);
    if (idx < 0)
        return core::makeError(core::ErrorCode::NotFound, "Node not found");
    _impl->nodes.erase(_impl->nodes.begin() + idx);
    _impl->missedBeats.erase(_impl->missedBeats.begin() + idx);
    return {};
}

core::Expected<void> ServerMesh::migrateEntity(core::u32 targetNodeId, std::span<const core::byte> entityData)
{
    // Server meshing hand-off: the entity's serialised state is shipped to the
    // node that now owns its region, over the same transport as everything else.
    const auto idx = _impl->indexOf(targetNodeId);
    if (idx < 0)
        return core::makeError(core::ErrorCode::NotFound, "Migration target is not a known node");

    auto res = _impl->sendTo(_impl->nodes[idx], protocol::PacketType::EntityMigrate, entityData);
    if (!res.has_value())
        return res;
    ++_impl->migrationsSent;
    return {};
}

void ServerMesh::heartbeat()
{
    // One beat to each node; a node that has now missed too many in a row without
    // an ack is declared dead so its regions can be reassigned.
    static constexpr core::byte kBeat[1]{core::byte{0}};
    for (core::usize i = 0; i < _impl->nodes.size(); ++i)
    {
        [[maybe_unused]] auto sent =
            _impl->sendTo(_impl->nodes[i], protocol::PacketType::NodeHeartbeat, std::span<const core::byte>{kBeat});
        ++_impl->heartbeatsSent;
        ++_impl->missedBeats[i];
        if (_impl->missedBeats[i] >= kMaxMissedHeartbeats)
            _impl->nodes[i].alive = false;
    }
}

void ServerMesh::onHeartbeatAck(core::u32 nodeId) noexcept
{
    const auto idx = _impl->indexOf(nodeId);
    if (idx < 0)
        return;
    _impl->missedBeats[idx] = 0;
    _impl->nodes[idx].alive = true;
}

core::u64 ServerMesh::heartbeatsSent() const noexcept { return _impl->heartbeatsSent; }

core::u64 ServerMesh::migrationsSent() const noexcept { return _impl->migrationsSent; }

std::span<const MeshNode> ServerMesh::nodes() const noexcept { return _impl->nodes; }

} // namespace lpl::net
