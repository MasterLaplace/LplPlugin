// /////////////////////////////////////////////////////////////////////////////
/// @file ServerMesh.cpp
/// @brief ServerMesh implementation stub.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/ServerMesh.hpp>
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
    : impl_{std::make_unique<Impl>(transport)}
{}

ServerMesh::~ServerMesh() = default;

core::Expected<void> ServerMesh::addNode(MeshNode node)
{
    for (const auto& n : impl_->nodes)
    {
        if (n.nodeId == node.nodeId)
        {
            return core::makeError(core::ErrorCode::AlreadyExists, "Node already registered");
        }
    }
    impl_->nodes.push_back(std::move(node));
    return {};
}

core::Expected<void> ServerMesh::removeNode(core::u32 nodeId)
{
    auto it = std::find_if(impl_->nodes.begin(), impl_->nodes.end(),
                           [nodeId](const MeshNode& n) { return n.nodeId == nodeId; });
    if (it == impl_->nodes.end())
    {
        return core::makeError(core::ErrorCode::NotFound, "Node not found");
    }
    impl_->nodes.erase(it);
    return {};
}

core::Expected<void> ServerMesh::migrateEntity(
    core::u32 /*targetNodeId*/,
    std::span<const core::byte> /*entityData*/)
{
    LPL_ASSERT(false && "ServerMesh::migrateEntity not yet implemented");
    return {};
}

void ServerMesh::heartbeat()
{
    LPL_ASSERT(false && "ServerMesh::heartbeat not yet implemented");
}

std::span<const MeshNode> ServerMesh::nodes() const noexcept
{
    return impl_->nodes;
}

} // namespace lpl::net
