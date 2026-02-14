// --- LAPLACE DEDICATED TEST SERVER --- //
// TEMPORARY TEST SERVER — will be replaced by production architecture
// File: main.cpp
// Description: Serveur autoritaire avec broadcast UDP vers les clients visual3d.
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
#include "NetworkDispatch.hpp"
#include "WorldPartition.hpp"
#include "SystemScheduler.hpp"
#include "SpinLock.hpp"

// ─── Global State ─────────────────────────────────────────────

static NetworkRingBuffer *g_ringBuffer = nullptr;
static volatile bool running = true;
static int g_serverSock = -1;
static std::atomic<uint32_t> g_nextEntityId{100}; // IDs 1-99 reserved for NPCs

// ─── Signal Handler (Ctrl+C) ─────────────────────────────────

static void sigint_handler(int) { running = false; }

// ─── Client Management ───────────────────────────────────────

struct ClientInfo {
    sockaddr_in addr;
    uint32_t entityId;
};

static std::vector<ClientInfo> g_clients;
static SpinLock g_clientsLock;

// ─── Message Queues (recv thread → main thread) ──────────────

struct PendingConnect {
    sockaddr_in addr;
    socklen_t addrLen;
};

struct PendingInput {
    uint32_t entityId;
    Vec3 direction;
};

static std::vector<PendingConnect> g_connectQueue;
static SpinLock g_connectLock;

static std::vector<PendingInput> g_inputQueue;
static SpinLock g_inputLock;

// ─── Timing ───────────────────────────────────────────────────

static inline uint64_t get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// ─── UDP Receive Thread ──────────────────────────────────────

static void *udp_receive_thread(void * /*arg*/)
{
    uint8_t buf[MAX_PACKET_SIZE];

    while (running)
    {
        sockaddr_in clientAddr{};
        socklen_t addrLen = sizeof(clientAddr);
        ssize_t n = recvfrom(g_serverSock, buf, sizeof(buf), 0,
                             reinterpret_cast<sockaddr *>(&clientAddr), &addrLen);
        if (n <= 0)
            continue;

        switch (buf[0])
        {
        case MSG_CONNECT: {
            LocalGuard lock(g_connectLock);
            g_connectQueue.push_back({clientAddr, addrLen});
            break;
        }
        case MSG_INPUT: {
            if (n < 17) break; // 1B type + 4B id + 12B direction
            PendingInput inp;
            inp.entityId = *reinterpret_cast<uint32_t *>(buf + 1);
            inp.direction = {
                *reinterpret_cast<float *>(buf + 5),
                *reinterpret_cast<float *>(buf + 9),
                *reinterpret_cast<float *>(buf + 13)
            };
            LocalGuard lock(g_inputLock);
            g_inputQueue.push_back(inp);
            break;
        }
        default:
            break;
        }
    }

    return nullptr;
}

// ─── Broadcast State to All Clients ──────────────────────────

/**
 * @brief Sérialise l'état du monde (read buffer) et l'envoie à tous les clients.
 *
 * Format MSG_STATE : [1B MSG_STATE][2B entityCount][{4B id, 12B pos, 12B size, 4B hp}×N]
 * 32 bytes par entité → ~43 entités par paquet de 1400 bytes.
 * Si plus d'entités, plusieurs paquets sont envoyés.
 */
static void broadcast_state(WorldPartition &world)
{
    std::vector<ClientInfo> clients;
    {
        LocalGuard lock(g_clientsLock);
        clients = g_clients;
    }
    if (clients.empty())
        return;

    uint32_t readIdx = world.getReadIdx();

    constexpr size_t MAX_UDP = 1400;
    constexpr size_t HDR_SIZE = 3;        // 1B type + 2B count
    constexpr size_t ENT_SIZE = 32;       // 4+12+12+4
    uint8_t pkt[MAX_UDP];
    uint16_t count = 0;
    uint8_t *cursor = pkt + HDR_SIZE;

    auto flush = [&]() {
        if (count == 0)
            return;
        pkt[0] = MSG_STATE;
        *reinterpret_cast<uint16_t *>(pkt + 1) = count;
        size_t pktSize = static_cast<size_t>(cursor - pkt);
        for (const auto &c : clients)
            sendto(g_serverSock, pkt, pktSize, 0,
                   reinterpret_cast<const sockaddr *>(&c.addr), sizeof(c.addr));
        cursor = pkt + HDR_SIZE;
        count = 0;
    };

    world.forEachChunk([&](Partition &p) {
        for (size_t i = 0; i < p.getEntityCount(); ++i)
        {
            if (static_cast<size_t>(cursor - pkt) + ENT_SIZE > MAX_UDP)
                flush();

            auto ent = p.getEntity(i, readIdx);

            *reinterpret_cast<uint32_t *>(cursor) = p.getEntityId(i);
            cursor += 4;
            *reinterpret_cast<float *>(cursor + 0) = ent.position.x;
            *reinterpret_cast<float *>(cursor + 4) = ent.position.y;
            *reinterpret_cast<float *>(cursor + 8) = ent.position.z;
            cursor += 12;
            *reinterpret_cast<float *>(cursor + 0) = ent.size.x;
            *reinterpret_cast<float *>(cursor + 4) = ent.size.y;
            *reinterpret_cast<float *>(cursor + 8) = ent.size.z;
            cursor += 12;
            *reinterpret_cast<int32_t *>(cursor) = ent.health;
            cursor += 4;
            count++;
        }
    });

    flush();
}

// ─── MAIN ─────────────────────────────────────────────────────

int main()
{
    signal(SIGINT, sigint_handler);
    printf("=== LplPlugin Test Server (TEMPORARY) ===\n\n");

    // 1. GPU init
    gpu_init();

    // 2. Kernel ring buffer (optional — skip if driver not loaded)
    g_ringBuffer = network_init();
    if (!g_ringBuffer)
        printf("[WARN] Driver non disponible, ring buffer désactivé.\n\n");

    // 3. UDP server socket
    g_serverSock = socket(AF_INET, SOCK_DGRAM, 0);
    if (g_serverSock < 0)
    {
        perror("[FATAL] socket()");
        gpu_cleanup();
        return 1;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(7777);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(g_serverSock, reinterpret_cast<sockaddr *>(&serverAddr), sizeof(serverAddr)) < 0)
    {
        perror("[FATAL] bind()");
        close(g_serverSock);
        gpu_cleanup();
        return 1;
    }

    // Timeout pour recvfrom (permet un shutdown propre)
    struct timeval tv{0, 100000}; // 100ms
    setsockopt(g_serverSock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

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

    // Système : Kernel ring buffer (si driver disponible)
    if (g_ringBuffer)
    {
        NetworkRingBuffer *ring = g_ringBuffer;
        scheduler.registerSystem({
            "RingConsume", -20,
            [ring](WorldPartition &w, float /*dt*/) {
                network_consume_packets(ring, w);
            },
            {
                {ComponentId::Position, AccessMode::Write},
                {ComponentId::Velocity, AccessMode::Write},
                {ComponentId::Health,   AccessMode::Write},
                {ComponentId::Mass,     AccessMode::Write},
                {ComponentId::Size,     AccessMode::Write},
            }
        });
    }

    // Système : Process connexions + inputs clients
    scheduler.registerSystem({
        "ClientIO", -10,
        [&](WorldPartition &w, float /*dt*/) {
            // ── Connexions ──
            std::vector<PendingConnect> connects;
            {
                LocalGuard lock(g_connectLock);
                std::swap(connects, g_connectQueue);
            }

            for (auto &c : connects)
            {
                uint32_t id = g_nextEntityId.fetch_add(1u);

                Partition::EntitySnapshot ent{};
                ent.id = id;
                ent.position = {0.f, 10.f, 0.f};
                ent.rotation = {0.f, 0.f, 0.f, 1.f};
                ent.velocity = {0.f, 0.f, 0.f};
                ent.mass = 1.f;
                ent.force = {0.f, 0.f, 0.f};
                ent.size = {1.f, 2.f, 1.f};
                ent.health = 100;
                w.addEntity(ent);

                // Répondre MSG_WELCOME
                uint8_t pkt[5];
                pkt[0] = MSG_WELCOME;
                *reinterpret_cast<uint32_t *>(pkt + 1) = id;
                sendto(g_serverSock, pkt, 5, 0,
                       reinterpret_cast<sockaddr *>(&c.addr), c.addrLen);

                {
                    LocalGuard lock(g_clientsLock);
                    g_clients.push_back({c.addr, id});
                }

                char ip[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &c.addr.sin_addr, ip, sizeof(ip));
                printf("[SERVER] Client %s:%d connecté → Entity #%u\n",
                       ip, ntohs(c.addr.sin_port), id);
            }

            // ── Inputs joueurs ──
            std::vector<PendingInput> inputs;
            {
                LocalGuard lock(g_inputLock);
                std::swap(inputs, g_inputQueue);
            }

            uint32_t writeIdx = w.getWriteIdx();
            for (auto &inp : inputs)
            {
                int localIdx = -1;
                Partition *chunk = w.findEntity(inp.entityId, localIdx);
                if (chunk && localIdx >= 0)
                {
                    constexpr float PLAYER_SPEED = 50.0f;
                    Vec3 vel = inp.direction * PLAYER_SPEED;
                    chunk->setVelocity(static_cast<uint32_t>(localIdx), vel, writeIdx);
                    chunk->wakeEntity(static_cast<uint32_t>(localIdx));
                }
            }
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

    // 6. Lancer le thread de réception UDP
    pthread_t recvThread;
    if (pthread_create(&recvThread, nullptr, udp_receive_thread, nullptr) != 0)
    {
        perror("[FATAL] pthread_create");
        close(g_serverSock);
        if (g_ringBuffer) network_cleanup(g_ringBuffer);
        gpu_cleanup();
        return 1;
    }

    // 7. Boucle principale (infinie — Ctrl+C pour arrêter)
    printf("\n[SERVER] Écoute sur UDP 0.0.0.0:7777. Ctrl+C pour arrêter.\n\n");

    constexpr uint64_t FRAME_TIME_NS = 16666666; // ~60Hz
    constexpr float DT = 1.f / 60.f;
    uint64_t frameCount = 0;

    while (running)
    {
        uint64_t frame_start = get_time_ns();

        // Tick ECS : RingConsume → ClientIO → Physics
        scheduler.tick(world, DT);

        // Swap buffers : rend les résultats de la physique visibles au read buffer
        world.swapBuffers();

        // Broadcast l'état du monde à tous les clients (depuis le read buffer)
        broadcast_state(world);

        frameCount++;
        if (frameCount % 300 == 0)
        {
            LocalGuard lock(g_clientsLock);
            printf("[SERVER] Frame %lu | Clients: %zu | Entities: %d | Chunks: %d\n",
                   static_cast<unsigned long>(frameCount), g_clients.size(),
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
    pthread_join(recvThread, nullptr);
    close(g_serverSock);
    if (g_ringBuffer) network_cleanup(g_ringBuffer);
    gpu_cleanup();
    printf("[SERVER] Terminé.\n");
    return 0;
}
