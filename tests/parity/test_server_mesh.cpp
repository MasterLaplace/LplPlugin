#include <lpl/net/ServerMesh.hpp>

#include <lpl/net/Endpoint.hpp>
#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/net/transport/ITransport.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

/** @brief Records every datagram: destination, packet type, and payload bytes. */
class CapturingTransport final : public lpl::net::transport::ITransport {
public:
    struct Sent {
        lpl::net::Endpoint dest;
        lpl::net::protocol::PacketType type;
        std::vector<lpl::core::byte> payload;
    };
    std::vector<Sent> sent;

    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "CapturingTransport"; }
    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }

    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> data,
                                             const lpl::net::Endpoint *address) override
    {
        Sent s{};
        s.dest = address ? *address : lpl::net::Endpoint{};
        if (data.size() >= sizeof(lpl::net::protocol::PacketHeader))
        {
            lpl::net::protocol::PacketHeader h{};
            std::memcpy(&h, data.data(), sizeof(h));
            s.type = h.type;
            s.payload.assign(data.begin() + sizeof(h), data.end());
        }
        sent.push_back(std::move(s));
        return static_cast<lpl::core::u32>(data.size());
    }

    [[nodiscard]] int countOfType(lpl::net::protocol::PacketType t) const
    {
        int n = 0;
        for (const auto &s : sent)
            if (s.type == t)
                ++n;
        return n;
    }
};

lpl::net::MeshNode makeNode(lpl::core::u32 id, const char *addr, lpl::core::u16 port)
{
    lpl::net::MeshNode n{};
    n.nodeId = id;
    n.address = addr;
    n.port = port;
    n.alive = true;
    return n;
}

bool aliveOf(const lpl::net::ServerMesh &mesh, lpl::core::u32 id)
{
    for (const auto &n : mesh.nodes())
        if (n.nodeId == id)
            return n.alive;
    return false;
}

/**
 * @brief Entity migration is a real packet to the target node.
 *
 * @param mesh      The mesh of nodes 1 and 2.
 * @param transport The transport the mesh sends through.
 */
void checkEntityMigrationIsARealPacketToTheTargetNode(lpl::net::ServerMesh &mesh, const CapturingTransport &transport)
{
    const lpl::core::byte blob[] = {lpl::core::byte{0xDE}, lpl::core::byte{0xAD}, lpl::core::byte{0xBE},
                                    lpl::core::byte{0xEF}};
    check(mesh.migrateEntity(1, blob).has_value(), "an entity migrates to node 1");
    check(mesh.migrationsSent() == 1, "the migration is counted");

    check(!transport.sent.empty(), "a datagram was actually sent");
    const auto &s = transport.sent.back();
    check(s.type == lpl::net::protocol::PacketType::EntityMigrate, "it is an EntityMigrate packet");
    lpl::net::Endpoint expected{};
    (void) lpl::net::Endpoint::parse("127.0.0.1", 5001, expected);
    check(s.dest == expected, "it went to node 1's address");
    check(s.payload.size() == 4 && s.payload[0] == lpl::core::byte{0xDE} && s.payload[3] == lpl::core::byte{0xEF},
          "it carried the entity bytes verbatim");

    check(!mesh.migrateEntity(99, blob).has_value(), "migrating to an unknown node fails");
}

/**
 * @brief Heartbeats age liveness; misses kill, an ack revives.
 *
 * @details Beats up to the miss threshold with no ack for either node, then node 1 answers while node 2 stays
 *          silent.
 *
 * @param mesh      The mesh of nodes 1 and 2.
 * @param transport The transport the mesh sends through.
 */
void checkMissedHeartbeatsKillAndAnAckRevives(lpl::net::ServerMesh &mesh, const CapturingTransport &transport)
{
    const int heartbeatsBefore = transport.countOfType(lpl::net::protocol::PacketType::NodeHeartbeat);
    mesh.heartbeat();
    check(transport.countOfType(lpl::net::protocol::PacketType::NodeHeartbeat) == heartbeatsBefore + 2,
          "one heartbeat is sent to each of the two nodes");
    check(aliveOf(mesh, 1) && aliveOf(mesh, 2), "one missed beat is not yet fatal");

    for (lpl::core::u32 i = 1; i < lpl::net::ServerMesh::kMaxMissedHeartbeats; ++i)
        mesh.heartbeat();
    check(!aliveOf(mesh, 1) && !aliveOf(mesh, 2), "a node that misses kMaxMissedHeartbeats beats is declared dead");

    mesh.onHeartbeatAck(1);
    check(aliveOf(mesh, 1), "an ack revives the node that answered");
    check(!aliveOf(mesh, 2), "the node still silent stays dead");

    check(mesh.heartbeatsSent() == static_cast<lpl::core::u64>(lpl::net::ServerMesh::kMaxMissedHeartbeats) * 2,
          "the heartbeat counter totals all beats sent");
}

/**
 * @brief Removal keeps the node list and its liveness bookkeeping aligned.
 *
 * @details Node 1's liveness state must still be intact after node 2 is erased.
 *
 * @param mesh The mesh of nodes 1 and 2.
 */
void checkRemovalKeepsNodesAndLivenessAligned(lpl::net::ServerMesh &mesh)
{
    check(mesh.removeNode(2).has_value(), "node 2 leaves the mesh");
    check(mesh.nodes().size() == 1 && mesh.nodes()[0].nodeId == 1, "only node 1 remains");
    check(!mesh.removeNode(2).has_value(), "removing it again fails");
    mesh.onHeartbeatAck(1);
    check(aliveOf(mesh, 1), "the surviving node's bookkeeping is undisturbed");
}

} // namespace

/**
 * @brief Server-mesh migration + heartbeat test (book §6.2.8).
 *
 * @details Proves the two userspace halves of server meshing: an entity is handed off to the node that owns
 *          its region (a real packet, to that node's address, carrying the entity bytes), and node liveness is
 *          tracked by heartbeats — a peer that misses too many beats without an ack is declared dead, and an
 *          ack revives it.
 */
int main()
{
    std::printf("== server-mesh migration + heartbeat ==\n");

    CapturingTransport transport;
    lpl::net::ServerMesh mesh{transport};

    check(mesh.addNode(makeNode(1, "127.0.0.1", 5001)).has_value(), "node 1 joins the mesh");
    check(mesh.addNode(makeNode(2, "127.0.0.1", 5002)).has_value(), "node 2 joins the mesh");
    check(!mesh.addNode(makeNode(1, "127.0.0.1", 5003)).has_value(), "a duplicate node id is rejected");

    checkEntityMigrationIsARealPacketToTheTargetNode(mesh, transport);
    checkMissedHeartbeatsKillAndAnAckRevives(mesh, transport);
    checkRemovalKeepsNodesAndLivenessAligned(mesh);

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
