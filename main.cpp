// --- LAPLACE DEDICATED TEST SERVER --- //
// TEMPORARY TEST SERVER — will be replaced by production architecture
// File: main.cpp
// Description: Serveur autoritaire avec broadcast UDP vers les clients visual.
//              Reçoit les inputs clients, applique la physique, broadcast l'état.
// Auteur: MasterLaplace & Copilot

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <unistd.h>

#include "PhysicsGPU.cuh"
#include "Network.hpp"
#include "WorldPartition.hpp"
#include "SystemScheduler.hpp"
#include "SpinLock.hpp"

// ─── Global State ─────────────────────────────────────────────

// ─── Global State ─────────────────────────────────────────────

static Network g_network;
static volatile bool running = true;

// ─── Signal Handler (Ctrl+C) ─────────────────────────────────

static void sigint_handler(int) { running = false; }

// ─── Timing ───────────────────────────────────────────────────

static inline uint64_t get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// ─── MAIN ─────────────────────────────────────────────────────

int main()
{
    signal(SIGINT, sigint_handler);
    printf("=== LplPlugin Test Server (TEMPORARY) ===\n\n");

    // 1. GPU init
    gpu_init();

    // 2. Kernel ring buffer
    if (!g_network.network_init())
    {
        printf("[FATAL] Echec initialisation Network (driver manquant ?)\n");
        // Fallback or exit? Assuming forced exit as per plan to rely on driver
        gpu_cleanup();
        return 1;
    }

    // 4. Créer le monde avec des NPCs
    WorldPartition world;
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
            world.addEntity(npc);
        }
        printf("[WORLD] 50 NPC entities spawned\n");
    }

    // 5. SystemScheduler ECS
    SystemScheduler scheduler;

    // Système : Network IO (Packet Consume + Logic)
    scheduler.registerSystem({
        "NetworkIO", -20,
        [&](WorldPartition &w, float /*dt*/) {
             g_network.network_consume_packets(w);
        },
        {
            {ComponentId::Position, AccessMode::Write},
            {ComponentId::Velocity, AccessMode::Write},
            {ComponentId::Health,   AccessMode::Write},
            {ComponentId::Mass,     AccessMode::Write},
            {ComponentId::Size,     AccessMode::Write},
        }
    });

    // Système : Physique
    scheduler.registerSystem({
        "Physics", 0,
        [&](WorldPartition &w, float dt) {
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
        }
    });

    scheduler.buildSchedule();
    scheduler.printSchedule();

    // 6. (Recv Thread removed, handled by ring buffer in main thread via NetworkIO system)

    // 7. Boucle principale (infinie — Ctrl+C pour arrêter)
    printf("\n[SERVER] Serveur démarré via Kernel Driver. Ctrl+C pour arrêter.\n\n");

    constexpr uint64_t FRAME_TIME_NS = 16666666; // ~60Hz
    constexpr float DT = 1.f / 60.f;
    uint64_t frameCount = 0;

    while (running)
    {
        uint64_t frame_start = get_time_ns();

        // Tick ECS : RingConsume → ClientIO → Physics
        scheduler.threaded_tick(world, DT);

        // Swap buffers : rend les résultats de la physique visibles au read buffer
        world.swapBuffers();

        // Broadcast l'état du monde à tous les clients (depuis le read buffer)
        g_network.broadcast_state(world);

        frameCount++;
        if (frameCount % 300 == 0)
        {
            printf("[SERVER] Frame %lu | Clients: %zu | Entities: %d | Chunks: %d\n",
                   static_cast<unsigned long>(frameCount), g_network.size(),
                   world.getEntityCount(), world.getChunkCount());
        }

        // Maintenir 60Hz
        uint64_t elapsed = get_time_ns() - frame_start;
        if (elapsed < FRAME_TIME_NS)
        {
            struct timespec ts{0, static_cast<long>(FRAME_TIME_NS - elapsed)};
            nanosleep(&ts, nullptr);
        }
    }

    // 8. Shutdown
    printf("\n[SERVER] Arrêt en cours...\n");
    g_network.network_cleanup();
    gpu_cleanup();
    printf("[SERVER] Terminé.\n");
    return 0;
}
