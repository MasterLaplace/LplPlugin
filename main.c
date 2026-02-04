#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // Pour les threads
#include <unistd.h>  // Pour usleep()
#include <math.h>    // Pour faire bouger les entités avec sin()

#include "plugin.c"

// --- VARIABLES GLOBALES POUR LA SIMULATION ---
NetworkRingBuffer ring_buffer = {0};
volatile bool running = true;

// Fonction utilitaire pour "créer" une entité (l'inverse de destroy)
// Dans un vrai moteur, ce serait une fonction 'create_entity'
uint32_t spawn_entity(uint32_t public_id)
{
    // 2. On récupère l'index interne disponible
    uint32_t internal = create_entity(public_id);

    printf("[SERVER] Spawned Entity ID %d -> Internal Index %d\n", public_id, internal);
    return internal;
}

// --- LE THREAD RÉSEAU (Producteur) ---
// Simule la réception de paquets depuis Internet
void* network_thread_func(void* arg)
{
    printf("[NETWORK] Thread démarré.\n");

    float time = 0.0f;
    uint32_t target_id = 42; // L'ID public du joueur qu'on va bouger

    while (running) {
        // 1. Simulation de données (le joueur fait des cercles)
        time += 0.01f;
        EntityPacket pkt;
        pkt.entity_id = target_id;
        pkt.pos_x = cosf(time) * 10.0f; // Oscille entre -10 et 10
        pkt.pos_y = sinf(time) * 10.0f;
        pkt.pos_z = 0.0f;

        // 2. Écriture dans le Ring Buffer (Lock-Free)
        uint32_t head = atomic_load(&ring_buffer.head);
        uint32_t tail = atomic_load(&ring_buffer.tail);
        uint32_t next_head = (head + 1) & (RING_SIZE - 1);

        // Si le buffer n'est pas plein...
        if (next_head != tail) {
            ring_buffer.packets[head] = pkt;

            // On s'assure que l'écriture du paquet est finie avant de bouger la tête
            atomic_store(&ring_buffer.head, next_head);
        } else {
            // Drop packet (le réseau va trop vite pour le CPU !)
            // printf("Packet Dropped!\n");
        }

        // Simulation de latence réseau (1ms)
        usleep(1000);
    }
    return NULL;
}

// --- LE MAIN (Consommateur / Moteur de Rendu) ---
int main()
{
    printf("=== Démarrage du Moteur SAO ===\n");

    // 1. Initialisation de la mémoire
    init_server();

    // 3. On fait apparaître notre joueur (ID 42)
    spawn_entity(42);

    // 4. Lancement du thread réseau
    pthread_t net_thread;
    if (pthread_create(&net_thread, NULL, network_thread_func, NULL) != 0) {
        perror("Erreur thread");
        return 1;
    }

    // 5. La Boucle de Jeu (Game Loop)
    int frame = 0;
    float* gpu_x = NULL;
    float* gpu_y = NULL;
    float* gpu_z = NULL;

    while (frame < 200) { // On simule 200 frames

        // --- ÉTAPE A : Lecture du Réseau ---
        // Vide le tampon et met à jour le buffer "Back" (caché)
        consume_packets(&ring_buffer);

        // --- ÉTAPE B : Synchronisation ---
        // Échange les buffers Back <-> Front
        swap_buffers();

        // --- ÉTAPE C : Préparation Rendu (Zero Copy) ---
        // Récupère les pointeurs vers les données stables
        get_render_pointers(&gpu_x, &gpu_y, &gpu_z);

        // --- ÉTAPE D : Simulation Rendu GPU ---
        // On lit les données comme le ferait la carte graphique
        // On doit retrouver l'index interne du joueur 42 pour l'afficher
        uint32_t internal_id = sparse_lookup[42];

        if (frame % 10 == 0) { // On affiche toutes les 10 frames pour pas spammer
            printf("Frame %3d | Buffer Lect: %p | Entity 42 Pos: [%.2f, %.2f]\n",
                   frame, (void*)gpu_x, gpu_x[internal_id], gpu_y[internal_id]);
        }

        // Simulation 60 FPS (16ms)
        usleep(16000);
        frame++;
    }

    running = false;
    pthread_join(net_thread, NULL);
    printf("=== Arrêt du Moteur ===\n");

    return 0;
}

// BUILD: gcc main.c -o engine -lpthread -lm
// RUN: ./engine
