/*
** LplPlugin — server-mesh migration + heartbeat test (book §6.2.8)
**
** Proves the two userspace halves of server meshing: an entity is handed off to
** the node that owns its region (a real packet, to that node's address, carrying
** the entity bytes), and node liveness is tracked by heartbeats — a peer that
** misses too many beats without an ack is declared dead, and an ack revives it.
*/

#include <lpl/net/ServerMesh.hpp>

#include <lpl/net/Endpoint.hpp>
#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/net/transport/ITransport.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

/// Records every datagram: destination, packet type, and payload bytes.
class CapturingTransport final : public net::transport::ITransport {
public:
    struct Sent {
        net::Endpoint dest;
        net::protocol::PacketType type;
        std::vector<core::byte> payload;
    };
    std::vector<Sent> sent;

    core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "CapturingTransport"; }
    core::Expected<core::u32> receive(std::span<core::byte>, net::Endpoint *) override { return core::u32{0}; }

    core::Expected<core::u32> send(std::span<const core::byte> data, const net::Endpoint *address) override
    {
        Sent s{};
        s.dest = address ? *address : net::Endpoint{};
        if (data.size() >= sizeof(net::protocol::PacketHeader))
        {
            net::protocol::PacketHeader h{};
            std::memcpy(&h, data.data(), sizeof(h));
            s.type = h.type;
            s.payload.assign(data.begin() + sizeof(h), data.end());
        }
        sent.push_back(std::move(s));
        return static_cast<core::u32>(data.size());
    }

    [[nodiscard]] int countOfType(net::protocol::PacketType t) const
    {
        int n = 0;
        for (const auto &s : sent)
            if (s.type == t)
                ++n;
        return n;
    }
};

net::MeshNode makeNode(core::u32 id, const char *addr, core::u16 port)
{
    net::MeshNode n{};
    n.nodeId = id;
    n.address = addr;
    n.port = port;
    n.alive = true;
    return n;
}

bool aliveOf(const net::ServerMesh &mesh, core::u32 id)
{
    for (const auto &n : mesh.nodes())
        if (n.nodeId == id)
            return n.alive;
    return false;
}

} // namespace

int main()
{
    std::printf("== server-mesh migration + heartbeat ==\n");

    CapturingTransport transport;
    net::ServerMesh mesh{transport};

    check(mesh.addNode(makeNode(1, "127.0.0.1", 5001)).has_value(), "node 1 joins the mesh");
    check(mesh.addNode(makeNode(2, "127.0.0.1", 5002)).has_value(), "node 2 joins the mesh");
    check(!mesh.addNode(makeNode(1, "127.0.0.1", 5003)).has_value(), "a duplicate node id is rejected");

    // ── Entity migration is a real packet to the target node ───────────────── //
    {
        const core::byte blob[] = {core::byte{0xDE}, core::byte{0xAD}, core::byte{0xBE}, core::byte{0xEF}};
        check(mesh.migrateEntity(1, blob).has_value(), "an entity migrates to node 1");
        check(mesh.migrationsSent() == 1, "the migration is counted");

        check(!transport.sent.empty(), "a datagram was actually sent");
        const auto &s = transport.sent.back();
        check(s.type == net::protocol::PacketType::EntityMigrate, "it is an EntityMigrate packet");
        net::Endpoint expected{};
        (void) net::Endpoint::parse("127.0.0.1", 5001, expected);
        check(s.dest == expected, "it went to node 1's address");
        check(s.payload.size() == 4 && s.payload[0] == core::byte{0xDE} && s.payload[3] == core::byte{0xEF},
              "it carried the entity bytes verbatim");

        check(!mesh.migrateEntity(99, blob).has_value(), "migrating to an unknown node fails");
    }

    // ── Heartbeats age liveness; misses kill, an ack revives ───────────────── //
    {
        const int heartbeatsBefore = transport.countOfType(net::protocol::PacketType::NodeHeartbeat);
        mesh.heartbeat();
        check(transport.countOfType(net::protocol::PacketType::NodeHeartbeat) == heartbeatsBefore + 2,
              "one heartbeat is sent to each of the two nodes");
        check(aliveOf(mesh, 1) && aliveOf(mesh, 2), "one missed beat is not yet fatal");

        // Beat up to the miss threshold with no ack for either node.
        for (core::u32 i = 1; i < net::ServerMesh::kMaxMissedHeartbeats; ++i)
            mesh.heartbeat();
        check(!aliveOf(mesh, 1) && !aliveOf(mesh, 2), "a node that misses kMaxMissedHeartbeats beats is declared dead");

        // Node 1 answers; node 2 stays silent.
        mesh.onHeartbeatAck(1);
        check(aliveOf(mesh, 1), "an ack revives the node that answered");
        check(!aliveOf(mesh, 2), "the node still silent stays dead");

        check(mesh.heartbeatsSent() == static_cast<core::u64>(net::ServerMesh::kMaxMissedHeartbeats) * 2,
              "the heartbeat counter totals all beats sent");
    }

    // ── Removal keeps the node list and its liveness bookkeeping aligned ───── //
    {
        check(mesh.removeNode(2).has_value(), "node 2 leaves the mesh");
        check(mesh.nodes().size() == 1 && mesh.nodes()[0].nodeId == 1, "only node 1 remains");
        check(!mesh.removeNode(2).has_value(), "removing it again fails");
        // Node 1's liveness state must still be intact after the erase.
        mesh.onHeartbeatAck(1);
        check(aliveOf(mesh, 1), "the surviving node's bookkeeping is undisturbed");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
