#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#define MAX_ENTITIES 10000
#define MAX_ID 1000000

#define RING_SIZE 4096 // Puissance de 2 pour utiliser un masque binaire rapide

typedef struct {
    uint32_t entity_id; // L'ID unique de l'objet/joueur
    float pos_x;
    float pos_y;
    float pos_z;
    // On pourrait ajouter la vélocité plus tard
} EntityPacket;

typedef struct {
    EntityPacket packets[RING_SIZE];
    atomic_uint head;
    atomic_uint tail;
} NetworkRingBuffer;
// Pour avancer l'index sans déborder (modulo rapide)
// index = (index + 1) & (RING_SIZE - 1);

// On double la mise : [2] pour le double buffering
float ecs_pos_x[2][MAX_ENTITIES];
float ecs_pos_y[2][MAX_ENTITIES];
float ecs_pos_z[2][MAX_ENTITIES];

// Notre table d'indirection
uint32_t sparse_lookup[MAX_ID];

uint32_t dirty_stack[MAX_ENTITIES];
atomic_uint dirty_count = 0u;

// On ajoute un bitset pour savoir si une entité est déjà marquée "sale"
// atomic_bool garantit que deux threads ne l'ajouteront pas deux fois
atomic_bool is_dirty[MAX_ENTITIES];

uint32_t free_indices[MAX_ENTITIES];
atomic_uint free_count = 0u;

// Cet index atomique nous dit quel buffer est actuellement "prêt" pour l'écriture
// 0 ou 1
atomic_uint write_idx = 0u;

void init_server()
{
    for (uint32_t index = 0u; index < MAX_ID; ++index)
        sparse_lookup[index] = UINT32_MAX;

    uint32_t value = MAX_ENTITIES;
    for (uint32_t index = 0u; index < MAX_ENTITIES; ++index)
        free_indices[index] = --value;

    atomic_store(&free_count, MAX_ENTITIES);
}

uint32_t create_entity(uint32_t public_id)
{
    uint32_t pos = atomic_fetch_sub(&free_count, 1u) - 1u;
    uint32_t internal_index = free_indices[pos];

    sparse_lookup[public_id] = internal_index;
    return internal_index;
}

void destroy_entity(uint32_t entity_id)
{
    if (entity_id >= MAX_ID)
        return;

    uint32_t internal_index = sparse_lookup[entity_id];

    if (internal_index >= MAX_ENTITIES)
        return;

    sparse_lookup[entity_id] = UINT32_MAX;

    uint32_t pos = atomic_fetch_add(&free_count, 1);

    free_indices[pos] = internal_index;
}

void dispatch_packet(EntityPacket pkt)
{
    // On récupère l'index de lecture/écriture actuel
    uint32_t current_w = atomic_load(&write_idx);

    uint32_t internal_index = sparse_lookup[pkt.entity_id];

    if (internal_index >= MAX_ENTITIES)
        return;

    // On écrit dans le buffer de travail actuel
    ecs_pos_x[current_w][internal_index] = pkt.pos_x;
    ecs_pos_y[current_w][internal_index] = pkt.pos_y;
    ecs_pos_z[current_w][internal_index] = pkt.pos_z;

    // 2. On tente de marquer l'entité comme "dirty"
    // atomic_exchange renvoie l'ancienne valeur.
    // Si c'était 'false', c'est qu'on est le premier à la marquer cette frame !
    if (atomic_exchange(&is_dirty[internal_index], true) == false)
    {
        // 3. On réserve une place unique dans la liste via fetch_add
        uint32_t pos = atomic_fetch_add(&dirty_count, 1);

        if (pos < MAX_ENTITIES)
            dirty_stack[pos] = internal_index;
    }
}

void swap_buffers()
{
    // Si write_idx était 0, il devient 1. S'il était 1, il devient 0.
    uint32_t old_w  = atomic_load(&write_idx);
    uint32_t next_w = old_w ^ 1u;
    atomic_store(&write_idx, next_w);

    uint32_t count = atomic_load(&dirty_count);

    for (uint32_t stack_index = 0u; stack_index < count; ++stack_index)
    {
        uint32_t entity_index = dirty_stack[stack_index];

        ecs_pos_x[next_w][entity_index] = ecs_pos_x[old_w][entity_index];
        ecs_pos_y[next_w][entity_index] = ecs_pos_y[old_w][entity_index];
        ecs_pos_z[next_w][entity_index] = ecs_pos_z[old_w][entity_index];

        atomic_store(&is_dirty[entity_index], false);
    }

    atomic_store(&dirty_count, 0u);

    // Ici, on enverrait normalement un signal au GPU pour lui dire :
    // "Hé, tu peux lire le buffer qui vient d'être rempli !"
}

void consume_packets(NetworkRingBuffer *ring)
{
    uint32_t tail = atomic_load(&ring->tail);
    uint32_t head = atomic_load(&ring->head);

    while (tail != head)
    {
        EntityPacket pkt = ring->packets[tail];
        dispatch_packet(pkt);
        tail = (tail + 1u) & (RING_SIZE - 1u);
    }

    atomic_store(&ring->tail, tail);
}

void get_render_pointers(float **out_x, float **out_y, float **out_z)
{
    uint32_t write_i  = atomic_load(&write_idx);
    uint32_t read_i = write_i ^ 1u;

    *out_x = ecs_pos_x[read_i];
    *out_y = ecs_pos_y[read_i];
    *out_z = ecs_pos_z[read_i];
}
