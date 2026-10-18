#include <lpl/engine/systems/SessionSystem.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/input/InputManager.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/net/Endpoint.hpp>
#    include <lpl/net/protocol/Protocol.hpp>
#    include <lpl/net/session/SessionManager.hpp>

#    include <cstdio>
#    include <cstring>
#    include <memory>
#    include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

/**
 * @brief Captures the last packet the server sent (the welcome), so the test can read back the id the client
 *        is told to use as "me".
 */
class WelcomeCapture final : public lpl::net::transport::ITransport {
public:
    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "WelcomeCapture"; }
    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }
    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> data, const lpl::net::Endpoint *) override
    {
        last.assign(data.begin(), data.end());
        return static_cast<lpl::core::u32>(data.size());
    }
    std::vector<lpl::core::byte> last;
};

/**
 * @brief The single entity's raw id, or kNull if the registry is not holding exactly one.
 */
[[nodiscard]] lpl::core::u32 soleEntityRaw(lpl::ecs::Registry &reg)
{
    lpl::core::u32 found = lpl::ecs::EntityId::kNull;
    lpl::core::u32 seen = 0;
    for (const auto &part : reg.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const auto ids = chunk->entities();
            for (lpl::core::u32 i = 0; i < chunk->count(); ++i)
            {
                found = ids[i].raw();
                ++seen;
            }
        }
    }
    return seen == 1 ? found : lpl::ecs::EntityId::kNull;
}

/**
 * @brief The ONE id: session, bound entity and input slot all name the entity's ECS id.
 *
 * @param sessions  The server's sessions after the handshake.
 * @param inputs    The server's input state after the handshake.
 * @param entityRaw The raw ECS id of the entity the handshake created.
 */
void checkOneIdNamesSessionEntityAndInputSlot(const lpl::net::session::SessionManager &sessions,
                                              const lpl::input::InputManager &inputs, lpl::core::u32 entityRaw)
{
    auto *s = sessions.find(entityRaw);
    check(s != nullptr, "the session is keyed by the entity's ECS id");
    check(s != nullptr && s->boundEntity().raw() == entityRaw, "the session is bound to that entity");
    check(inputs.hasEntity(entityRaw), "input is registered under the entity id (so MovementSystem matches it)");
}

/**
 * @brief The welcome payload, which the client will echo back as its own id, is the entity's ECS id.
 *
 * @param transport The transport that captured the welcome.
 * @param entityRaw The raw ECS id of the entity the handshake created.
 */
void checkWelcomeHandsTheClientTheEntityId(const WelcomeCapture &transport, lpl::core::u32 entityRaw)
{
    check(transport.last.size() == sizeof(lpl::net::protocol::PacketHeader) + 4, "a welcome packet was sent");
    if (transport.last.size() >= sizeof(lpl::net::protocol::PacketHeader) + 4)
    {
        const auto &header = *reinterpret_cast<const lpl::net::protocol::PacketHeader *>(transport.last.data());
        check(header.type == lpl::net::protocol::PacketType::HandshakeAck, "it is a HandshakeAck");
        lpl::core::u32 welcomeId = 0;
        std::memcpy(&welcomeId, transport.last.data() + sizeof(lpl::net::protocol::PacketHeader), 4);
        check(welcomeId == entityRaw, "the welcome hands the client the entity's ECS id — the loop is closed");
    }
}

} // namespace

/**
 * @brief Server session identity test.
 *
 * @details Proves the network id is unified on the entity's ECS id: when a client connects, the SAME id names
 *          the session, the input slot, the welcome payload and the ECS entity — so the input a client sends
 *          back (tagged with the welcome id) reaches the very entity MovementSystem drives (keyed by the ECS
 *          id). The port had split these into a player counter and the ECS id, so player input never moved
 *          the player and a client could not find its own avatar. A retransmitted handshake from the same
 *          address must not create a second session or entity.
 */
int main()
{
    std::printf("== server session identity ==\n");

    lpl::net::session::SessionManager sessions;
    lpl::engine::EventQueues queues;
    auto transport = std::make_shared<WelcomeCapture>();
    lpl::input::InputManager inputs;
    [[maybe_unused]] auto ir = inputs.init();
    lpl::ecs::WorldPartition world{lpl::math::Fixed32::fromFloat(10.0f), 4096};
    lpl::ecs::Registry registry;

    lpl::engine::systems::SessionSystem session{sessions, queues, transport, inputs, world, registry};

    lpl::engine::ConnectEvent ev{};
    ev.source = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 40010);
    queues.connects.push(ev);

    session.execute(1.0f / 60.0f);

    const lpl::core::u32 entityRaw = soleEntityRaw(registry);
    check(entityRaw != lpl::ecs::EntityId::kNull, "the handshake created exactly one entity");
    check(sessions.activeCount() == 1, "and exactly one session");

    checkOneIdNamesSessionEntityAndInputSlot(sessions, inputs, entityRaw);
    checkWelcomeHandsTheClientTheEntityId(*transport, entityRaw);

    queues.connects.push(ev);
    session.execute(1.0f / 60.0f);
    check(sessions.activeCount() == 1, "a retransmitted handshake does not double-connect");
    check(soleEntityRaw(registry) == entityRaw, "and does not spawn a second entity");

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#else

#    include <cstdio>

int main()
{
    std::printf("session identity test skipped: built without LPL_HAS_NET\n");
    return 0;
}

#endif
