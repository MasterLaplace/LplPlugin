#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>

#include "plugin.h"

NetworkRingBuffer *ring_buffer = NULL;
volatile bool running = true;

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
}

// Fonction helper pour spawn
uint32_t spawn_entity(uint32_t public_id)
{
    uint32_t internal = create_entity(public_id);
    printf("[SERVER] Spawned Entity ID %d -> Internal Index %d\n", public_id, internal);
    return internal;
}

// --- MAIN (Moteur de Rendu) ---
int main()
{
    printf("=== Démarrage du Moteur SAO (Dynamic) ===\n");

    server_init();
    setup_kernel_link();

    uint32_t player_handle = spawn_entity(42);

    int frame = 0;
    float *gpu_x, *gpu_y, *gpu_z;
    int *gpu_hp;

    while (frame < 100)
    {
        consume_packets(ring_buffer);
        run_physics_gpu(0.016f); // dt ~ 16ms
        swap_buffers();

        get_render_pointers(&gpu_x, &gpu_y, &gpu_z);
        get_health_pointer(&gpu_hp);

        uint16_t i = get_entity_id(player_handle);

        if (frame % 5 == 0 && is_entity_valid(player_handle))
        {
            printf("Frame %3d | Pos: [%5.2f, %5.2f, %5.2f] | HP: %3d | Gen: %d\n",
                   frame, gpu_x[i], gpu_y[i], gpu_z[i], gpu_hp[i], get_entity_generation(player_handle));
        }

        usleep(16000);
        frame++;
    }

    running = false;
    server_cleanup();
    printf("=== Arrêt ===\n");
    return 0;
}
