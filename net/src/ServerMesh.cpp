/**
 * @file ServerMesh.cpp
 * @brief ServerMesh implementation stub.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/ServerMesh.hpp>
#include <stdexcept>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <algorithm>

namespace lpl::net {

struct ServerMesh::Impl
{
    transport::ITransport&  transport;
    std::vector<MeshNode>   nodes;

    explicit Impl(transport::ITransport& t) : transport{t} {}
};

ServerMesh::ServerMesh(transport::ITransport& transport)
    : _impl{std::make_unique<Impl>(transport)}
{}

ServerMesh::~ServerMesh() = default;

core::Expected<void> ServerMesh::addNode(MeshNode node)
{
    for (const auto& n : _impl->nodes)
    {
        if (n.nodeId == node.nodeId)
        {
            return core::makeError(core::ErrorCode::AlreadyExists, "Node already registered");
        }
    }
    _impl->nodes.push_back(std::move(node));
    return {};
}

core::Expected<void> ServerMesh::removeNode(core::u32 nodeId)
{
    auto it = std::find_if(_impl->nodes.begin(), _impl->nodes.end(),
                           [nodeId](const MeshNode& n) { return n.nodeId == nodeId; });
    if (it == _impl->nodes.end())
    {
        return core::makeError(core::ErrorCode::NotFound, "Node not found");
    }
    _impl->nodes.erase(it);
    return {};
}

core::Expected<void> ServerMesh::migrateEntity(
    core::u32 /*targetNodeId*/,
    std::span<const core::byte> /*entityData*/)
{
    LPL_ASSERT(false && "unimplemented");
    return {};
}

void ServerMesh::heartbeat()
{
    LPL_ASSERT(false && "unimplemented");
}

std::span<const MeshNode> ServerMesh::nodes() const noexcept
{
    return _impl->nodes;
}

} // namespace lpl::net
