#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>

#include "plugin.h"

NetworkRingBuffer ring_buffer = {0};
volatile bool running = true;

// Fonction helper pour spawn
uint32_t spawn_entity(uint32_t public_id)
{
    uint32_t internal = create_entity(public_id);
    printf("[SERVER] Spawned Entity ID %d -> Internal Index %d\n", public_id, internal);
    return internal;
}

// --- THREAD RÉSEAU (Producteur Dynamique) ---
void *network_thread_func(void *arg)
{
    printf("[NETWORK] Thread démarré (Mode Dynamique).\n");

    float time = 0.0f;
    uint32_t target_id = 42;

    while (running)
    {
        time += 0.05f;

        // 1. Préparation du paquet directement dans le Ring Buffer (Zero Copy à l'écriture aussi !)
        // On récupère l'index où écrire
        uint32_t head = atomic_load(&ring_buffer.head);
        uint32_t tail = atomic_load(&ring_buffer.tail);
        uint32_t next_head = (head + 1) & (RING_SIZE - 1);

        if (next_head != tail)
        {
            DynamicPacket *pkt = &ring_buffer.packets[head];
            uint8_t *cursor = pkt->data;

            // A. Écriture de l'ID (4 bytes)
            *(uint32_t *)cursor = target_id;
            cursor += sizeof(uint32_t);

            // B. Écriture du composant TRANSFORM (Tag + 3 floats)
            *cursor = COMP_TRANSFORM;
            cursor++;
            *(float *)cursor = cosf(time) * 10.0f;
            cursor += sizeof(float); // X
            *(float *)cursor = sinf(time) * 10.0f;
            cursor += sizeof(float); // Y
            *(float *)cursor = 0.0f;
            cursor += sizeof(float); // Z

            // C. Écriture du composant HEALTH (Tag + 1 int)
            // On fait varier la vie pour le fun
            *cursor = COMP_HEALTH;
            cursor++;
            *(int *)cursor = (int)(50 + sinf(time * 2.0f) * 50);
            cursor += sizeof(int);

            // D. Finalisation
            pkt->size = cursor - pkt->data; // Calcul de la taille réelle utilisée

            // Validation de l'écriture
            atomic_store(&ring_buffer.head, next_head);
        }

        usleep(16000); // ~60hz d'envoi
    }
    return NULL;
}

// --- MAIN (Moteur de Rendu) ---
int main()
{
    printf("=== Démarrage du Moteur SAO (Dynamic) ===\n");

    server_init();

    uint32_t player_handle = spawn_entity(42);

    pthread_t net_thread;
    if (pthread_create(&net_thread, NULL, network_thread_func, NULL) != 0)
        return 1;

    int frame = 0;
    float *gpu_x, *gpu_y, *gpu_z;
    int *gpu_hp; // Pointeur pour lire la vie

    while (frame < 100)
    {
        consume_packets(&ring_buffer);
        swap_buffers();

        get_render_pointers(&gpu_x, &gpu_y, &gpu_z);
        get_health_pointer(&gpu_hp); // Récupération du buffer de vie

        uint16_t real_index = get_entity_id(player_handle);

        if (frame % 5 == 0 && is_entity_valid(player_handle))
        {
            // Affiche la position ET la vie, prouvant que les 2 composants passent
            printf("Frame %3d | Pos: [%5.2f, %5.2f] | HP: %3d | Gen: %d\n",
                   frame, gpu_x[real_index], gpu_y[real_index], gpu_hp[real_index], get_entity_generation(player_handle));
        }

        usleep(16000);
        frame++;
    }

    running = false;
    pthread_join(net_thread, NULL);
    server_cleanup();
    printf("=== Arrêt ===\n");
    return 0;
}
