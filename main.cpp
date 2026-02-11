// --- LAPLACE MAIN SERVER EXAMPLE --- //
// File: main.c
// Description: Exemple de moteur côté serveur utilisant le plugin pour la communication réseau et la simulation GPU
// Note: Ce code à été partiellement généré par Copilot et édité par MasterLaplace pour démontrer les capacités du plugin
// Auteur: MasterLaplace & Copilot

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>

#include "Engine.cuh"
#include "WorldPartition.hpp"

NetworkRingBuffer *ring_buffer = NULL;
volatile bool running = true;
static PerfMetrics metrics = {0};

// Obtenir un timestamp haute résolution en nanosecondes
static inline uint64_t get_time_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void setup_kernel_link()
{
    int fd = open("/dev/lpl_driver", O_RDWR);
    if (fd < 0)
    {
        perror("[ERROR] setup_kernel_link: Impossible d'ouvrir /dev/lpl_driver");
        exit(1);
    }

    ring_buffer = (NetworkRingBuffer *)mmap(NULL, sizeof(NetworkRingBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ring_buffer == MAP_FAILED)
    {
        perror("[ERROR] setup_kernel_link: Erreur de mappage mémoire");
        close(fd);
        exit(1);
    }

    printf("[SYSTEM] setup_kernel_link: Driver connecté. Ring Buffer mappé à %p\n", ring_buffer);
    printf("[SYSTEM] Ring Buffer size: %zu bytes\n", sizeof(NetworkRingBuffer));
    printf("[SYSTEM] Initial head=%u, tail=%u\n", lpl_atomic_load(&ring_buffer->head), lpl_atomic_load(&ring_buffer->tail));
}

// --- THREAD CLIENT UDP (Producteur) ---
void *udp_client_thread(void *arg)
{
    printf("[CLIENT] Thread démarré. Envoi de paquets UDP à haute fréquence...\n");

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        perror("[ERROR] socket()");
        return NULL;
    }

    // Optimisations socket
    int sendbuf_size = 256 * 1024; // 256KB send buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf_size, sizeof(sendbuf_size));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(7777);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    int frame = 0;
    uint64_t total_sent = 0;
    uint64_t batch_start = get_time_ns();

    // HAUTE FRÉQUENCE : 1000 paquets/sec (1ms intervalle)
    const int TOTAL_PACKETS = 1000;
    const uint64_t INTERVAL_NS = 1000000; // 1ms en nanosecondes

    while (running && frame < TOTAL_PACKETS)
    {
        // Construit un paquet binaire
        uint8_t payload[256];
        uint8_t *cursor = payload;

        // 1. ID d'entité (4 bytes)
        uint32_t entity_id = 42;
        *(uint32_t *)cursor = entity_id;
        cursor += sizeof(uint32_t);

        // 2. Composant TRANSFORM (Tag + 3 floats)
        *cursor = COMP_TRANSFORM;
        cursor++;
        *(float *)cursor = cosf(frame * 0.01f) * 10.0f;
        cursor += sizeof(float);
        *(float *)cursor = sinf(frame * 0.01f) * 10.0f;
        cursor += sizeof(float);
        *(float *)cursor = 0.5f;
        cursor += sizeof(float);

        // 3. Composant HEALTH (Tag + 1 int)
        *cursor = COMP_HEALTH;
        cursor++;
        *(int *)cursor = 50 + (int)(30.0f * sinf(frame * 0.02f));
        cursor += sizeof(int);

        uint16_t payload_size = cursor - payload;

        // Envoie le paquet
        ssize_t sent = sendto(sock, payload, payload_size, 0, (struct sockaddr *)&server_addr, sizeof(server_addr));
        if (sent > 0)
        {
            total_sent++;
            metrics.packets_sent++;

            // Log tous les 100 paquets seulement (pour ne pas polluer)
            if (frame % 100 == 0)
            {
                uint64_t elapsed_ms = (get_time_ns() - batch_start) / 1000000;
                float throughput = (float)frame / ((float)elapsed_ms / 1000.0f);
                printf("[CLIENT] Frame %4d: %lu paquets envoyés | Throughput: %.0f pkt/s\n",
                       frame, (unsigned long)total_sent, throughput);
            }
        }

        // Sleep précis avec nanosleep
        struct timespec sleep_time;
        sleep_time.tv_sec = 0;
        sleep_time.tv_nsec = INTERVAL_NS;
        nanosleep(&sleep_time, NULL);

        frame++;
    }

    uint64_t total_time_ms = (get_time_ns() - batch_start) / 1000000;
    printf("[CLIENT] Thread terminé: %lu paquets en %lu ms (%.2f pkt/s)\n",
           (unsigned long)total_sent, (unsigned long)total_time_ms,
           (float)total_sent / ((float)total_time_ms / 1000.0f));

    close(sock);
    return NULL;
}

// --- MAIN (Moteur Unifié WorldPartition) ---
int main()
{
    printf("=== Démarrage du Moteur LplPlugin (Unified WorldPartition) ===\n");

    // 1. Initialiser le contexte CUDA
    engine_init();

    // 2. Setup kernel link (mmap ring buffer)
    setup_kernel_link();

    // 3. Créer le monde
    WorldPartition world;

    // 4. Pré-enregistrer l'entité joueur (ID 42)
    Partition::EntitySnapshot player{};
    player.id = 42;
    player.position = {0.f, 10.f, 0.f};
    player.rotation = {0.f, 0.f, 0.f, 1.f};
    player.velocity = {0.f, 0.f, 0.f};
    player.mass = 1.0f;
    player.force = {0.f, 0.f, 0.f};
    player.size = {1.f, 2.f, 1.f};
    player.health = 100;
    uint32_t player_handle = world.addEntity(player);
    printf("[WORLD] Entity #42 registered (handle: 0x%X)\n", player_handle);

    // 5. Lancer le thread client UDP (producteur de paquets)
    pthread_t client_thread;
    if (pthread_create(&client_thread, NULL, udp_client_thread, NULL) != 0)
    {
        perror("[ERROR] pthread_create");
        return 1;
    }

    int frame = 0;

    printf("\n[MAIN] Démarrage de la boucle simulation...\n\n");

    metrics.start_time_ns = get_time_ns();
    metrics.min_latency_ns = UINT64_MAX;
    metrics.max_latency_ns = 0;

    // Durée du test : 2 secondes à 60Hz = ~120 frames
    const int MAX_FRAMES = 120;
    const uint64_t FRAME_TIME_NS = 16666666; // ~60Hz (16.666ms)

    while (frame < MAX_FRAMES && running)
    {
        uint64_t frame_start = get_time_ns();

        // 1. Consommer les paquets réseau → WorldPartition
        uint32_t tail_before = lpl_atomic_load(&ring_buffer->tail);
        uint64_t consume_start = get_time_ns();
        engine_consume_packets(ring_buffer, world);
        uint64_t consume_end = get_time_ns();
        uint32_t tail_after = lpl_atomic_load(&ring_buffer->tail);
        uint32_t packets_consumed = (tail_after - tail_before) & (RING_SIZE - 1);

        if (packets_consumed > 0)
        {
            metrics.packets_received += packets_consumed;
            uint64_t consume_time_us = (consume_end - consume_start) / 1000;

            if (frame % 20 == 0)
            {
                printf("[RING] Frame %3d: Consommé %u pkt | Temps: %lu µs | Total reçu: %lu\n",
                       frame, packets_consumed, (unsigned long)consume_time_us,
                       (unsigned long)metrics.packets_received);
            }
        }

        // 2. Physique GPU + migration inter-chunk
        uint64_t gpu_start = get_time_ns();
        engine_physics_tick(world, 0.016f);
        uint64_t gpu_end = get_time_ns();

        // 3. Lecture de l'état de l'entité pour le "rendu"
        if (frame % 20 == 0)
        {
            int localIdx = -1;
            Partition *chunk = world.findEntity(42, localIdx);
            if (chunk && localIdx >= 0)
            {
                auto ent = chunk->getEntity(static_cast<size_t>(localIdx));
                uint64_t gpu_time_us = (gpu_end - gpu_start) / 1000;
                printf("[RENDER] Frame %3d | Pos: [%5.2f, %5.2f, %5.2f] | HP: %3d | GPU: %lu µs\n",
                       frame, ent.position.x, ent.position.y, ent.position.z,
                       ent.health, (unsigned long)gpu_time_us);
            }
        }

        // 4. Timing de la frame
        uint64_t frame_end = get_time_ns();
        uint64_t frame_time = frame_end - frame_start;

        metrics.total_latency_ns += frame_time;
        if (frame_time < metrics.min_latency_ns) metrics.min_latency_ns = frame_time;
        if (frame_time > metrics.max_latency_ns) metrics.max_latency_ns = frame_time;
        metrics.frame_count++;

        // Sleep pour maintenir 60Hz
        if (frame_time < FRAME_TIME_NS)
        {
            struct timespec sleep_time;
            uint64_t sleep_ns = FRAME_TIME_NS - frame_time;
            sleep_time.tv_sec = 0;
            sleep_time.tv_nsec = sleep_ns;
            nanosleep(&sleep_time, NULL);
        }

        frame++;
    }

    printf("\n[MAIN] Attente de la fin du thread client...\n");
    running = false;
    pthread_join(client_thread, NULL);

    engine_cleanup();

    // === RAPPORT DE PERFORMANCE FINAL ===
    uint64_t total_time_ms = (get_time_ns() - metrics.start_time_ns) / 1000000;
    float avg_latency_us = (float)metrics.total_latency_ns / (float)metrics.frame_count / 1000.0f;

    printf("\n");
    printf("# RAPPORT DE PERFORMANCE FINAL\n");
    printf("\n");
    printf("### STATISTIQUES RÉSEAU\n");
    printf("---\n");
    printf("- Paquets envoyés     : %lu\n", (unsigned long)metrics.packets_sent);
    printf("- Paquets reçus       : %lu\n", (unsigned long)metrics.packets_received);
    printf("- Paquets perdus      : %lu (%.2f%%)\n",
           (unsigned long)(metrics.packets_sent - metrics.packets_received),
           100.0f * (float)(metrics.packets_sent - metrics.packets_received) / (float)metrics.packets_sent);
    printf("- Throughput réseau   : %.2f pkt/s\n\n",
           (float)metrics.packets_received / ((float)total_time_ms / 1000.0f));

    printf("### PERFORMANCE FRAME\n");
    printf("---\n");
    printf("- Frames totales      : %lu\n", (unsigned long)metrics.frame_count);
    printf("- Temps total         : %lu ms\n", (unsigned long)total_time_ms);
    printf("- Framerate réel      : %.2f fps\n",
           (float)metrics.frame_count / ((float)total_time_ms / 1000.0f));
    printf("- Frame time (avg)    : %.2f µs\n", avg_latency_us);
    printf("- Frame time (min)    : %.2f µs\n", (float)metrics.min_latency_ns / 1000.0f);
    printf("- Frame time (max)    : %.2f µs\n\n", (float)metrics.max_latency_ns / 1000.0f);

    printf("**OBJECTIF:** <16.666ms par frame (60 FPS)\n");
    if (avg_latency_us < 16666.0f)
        printf("**OBJECTIF ATTEINT:** (%.2f µs < 16666 µs)\n", avg_latency_us);
    else
        printf("**OBJECTIF NON ATTEINT:** (%.2f µs > 16666 µs)\n", avg_latency_us);

    printf("\n=== Arrêt ===\n");
    return 0;
}
