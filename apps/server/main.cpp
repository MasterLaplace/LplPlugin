// --- LAPLACE DEDICATED SERVER --- //
// File: main.cpp
// Description: Serveur autoritaire utilisant le Core engine.
//              Enregistre les systèmes serveur, spawn les NPCs, core.run().
// Auteur: MasterLaplace

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "Core.hpp"
#include "Systems.hpp"

// ─── Timing Helper ────────────────────────────────────────────

static inline uint64_t get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// ─── MAIN ─────────────────────────────────────────────────────

int main()
{
    printf("=== LplPlugin Server ===\n\n");

    // 1. Create Core (inits GPU + Network)
    Core core;

    // 2. Spawn NPCs
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> posDist(-500.f, 500.f);
        std::uniform_real_distribution<float> velDist(-20.f, 20.f);
        std::uniform_real_distribution<float> sizeDist(1.5f, 5.f);

        for (uint32_t i = 1; i <= 50; ++i)
        {
            Partition::EntitySnapshot npc{};
            npc.id = i;
            npc.position = {posDist(rng), 10.f, posDist(rng)};
            npc.rotation = {0.f, 0.f, 0.f, 1.f};
            npc.velocity = {velDist(rng), 0.f, velDist(rng)};
            npc.mass = 1.f;
            npc.size = {sizeDist(rng), sizeDist(rng) * 1.2f, sizeDist(rng)};
            npc.health = 100;
            core.world().addEntity(npc);
        }
        printf("[WORLD] 50 NPC entities spawned\n");
    }

    // 3. Register server systems

    // PreSwap: Network Receive → Session → InputProcessing → Movement → Physics
    core.registerSystem(Systems::NetworkReceiveSystem(core.network(), core.packetQueue()));
    core.registerSystem(Systems::SessionSystem(core.sessionManager(), core.packetQueue(),
                                               core.network(), core.inputManager()));
    core.registerSystem(Systems::InputProcessingSystem(core.packetQueue(), core.inputManager()));
    core.registerSystem(Systems::MovementSystem(core.inputManager()));

    // Physics system with monitoring
    core.registerSystem({
        "Physics", 0,
        [](WorldPartition &w, float dt) {
            static uint64_t totalTime = 0;
            static uint64_t count = 0;

            uint64_t t0 = get_time_ns();
            w.step(dt);
            uint64_t t1 = get_time_ns();

            totalTime += (t1 - t0);
            count++;

            if (count % 60 == 0)
            {
                double avgMs = static_cast<double>(totalTime) / 60.0 / 1000000.0;
                printf("[PERF] Physics Step Avg: %.3f ms | Entities: %d | Chunks: %d | Transit: %zu\n",
                       avgMs, w.getEntityCount(), w.getChunkCount(), w.getTransitCount());
                totalTime = 0;
            }
        },
        {
            {ComponentId::Position, AccessMode::Write},
            {ComponentId::Velocity, AccessMode::Write},
            {ComponentId::Forces,   AccessMode::Write},
            {ComponentId::Mass,     AccessMode::Read},
        },
        SchedulePhase::PreSwap
    });

    // PostSwap: Broadcast state to all clients
    core.registerSystem(Systems::BroadcastSystem(core.sessionManager(), core.network()));

    // PostSwap: Server monitoring
    core.registerSystem({
        "ServerMonitor", 10,
        [&core](WorldPartition &w, float /*dt*/) {
            static uint64_t frameCount = 0;
            frameCount++;
            if (frameCount % 300 == 0)
            {
                printf("[SERVER] Frame %lu | Clients: %zu | Entities: %d | Chunks: %d\n",
                       static_cast<unsigned long>(frameCount),
                       core.sessionManager().getClientCount(),
                       w.getEntityCount(), w.getChunkCount());
            }
        },
        {},
        SchedulePhase::PostSwap
    });

    // 4. Build schedule and print it
    core.buildSchedule();
    core.printSchedule();

    // 5. Run
    printf("\n[SERVER] Serveur démarré. Ctrl+C pour arrêter.\n\n");
    core.run(true); // threaded mode

    printf("[SERVER] Terminé.\n");
    return 0;
}
