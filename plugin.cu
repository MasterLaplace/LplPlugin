// --- LAPLACE PLUGIN SYSTEM --- //
// File: plugin.cu
// Description: Implémentation du plugin côté serveur (GPU)
// Auteur: MasterLaplace

#include "plugin.h"

// --- ECS DATA ---//
// Au lieu de tableaux statiques, on a des pointeurs vers la RAM épinglée.
// [0] = Buffer A, [1] = Buffer B
float *ecs_pos_x[2];
float *ecs_pos_y[2];
float *ecs_pos_z[2];
int *ecs_health[2];

// POINTEURS GPU (Device Pointers)
// Ce sont les adresses que le GPU utilisera pour lire la même mémoire que le CPU (Zero-Copy)
float *d_ecs_pos_x[2];
float *d_ecs_pos_y[2];
float *d_ecs_pos_z[2];
int *d_ecs_health[2];

// --- CORE SYSTEMS ---//
static Core core = {
    .entity_generations = {0},
    .dirty_count = 0u,
    .free_count = 0u,
    .write_idx = 0u
};
static Chronos chrono;

// --- PRIVATE SYSTEMS ---//

// __host__  __device__
static void alloc_pinned_buffer(void **host, void **device, size_t size)
{
    // 1. Allocation Pinned + Mapped
    // cudaHostAllocMapped : Permet au GPU d'accéder à cette RAM via PCIe
    // cudaHostAllocPortable : La mémoire est visible par tous les contextes CUDA
    CUDA_CHECK(cudaHostAlloc(host, size, cudaHostAllocMapped | cudaHostAllocPortable));

    // 2. Récupération du pointeur "Device"
    // C'est l'adresse virtuelle que le GPU devra utiliser pour lire cette zone
    CUDA_CHECK(cudaHostGetDevicePointer(device, *host, 0));

    // Nettoyage initial (memset sur le CPU)
    memset(*host, 0, size);
}

__global__ void kernel_physics_update(float *pos_y, int count, float delta_time)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    float y = pos_y[idx];

    y -= 9.81f * delta_time;
    pos_y[idx] = (y < 0.0f) ? 0.0f : y;
}

static void dispatch_packet(DynamicPacket *pkt)
{
    uint32_t current_w = atomic_load(&core.write_idx);
    uint8_t *cursor = pkt->data;

    uint32_t public_id = *(uint32_t *)cursor;
    cursor += sizeof(uint32_t);

    uint32_t internal_index = core.sparse_lookup[public_id];

    if (internal_index >= MAX_ENTITIES)
        return (void)printf("[ERROR] dispatch_packet: internal_index(%d) >= MAX_ENTITIES(%d)\n", internal_index, MAX_ENTITIES);

    while (cursor < pkt->data + pkt->size)
    {
        uint8_t component_id = *cursor;
        ++cursor;

        switch (component_id) {
        case COMP_TRANSFORM:
        {
            float x = *(float *)cursor;
            cursor += sizeof(float);
            float y = *(float *)cursor;
            cursor += sizeof(float);
            float z = *(float *)cursor;
            cursor += sizeof(float);

            ecs_pos_x[current_w][internal_index] = x;
            ecs_pos_y[current_w][internal_index] = y;
            ecs_pos_z[current_w][internal_index] = z;
            break;
        }
        case COMP_HEALTH:
        {
            int hp = *(int *)cursor;
            cursor += sizeof(int);
            ecs_health[current_w][internal_index] = hp;
            break;
        }
        default:
            printf("[WARNING] dispatch_packet: unknown component_id(%d)\n", (uint32_t)component_id);
            break;
        }
    }

    // On tente de marquer l'entité comme "dirty"
    // Si c'était 'false', c'est qu'on est le premier à la marquer cette frame !
    if (atomic_exchange(&core.is_dirty[internal_index], true) == false)
    {
        uint32_t pos = atomic_fetch_add(&core.dirty_count, 1);

        if (pos < MAX_ENTITIES)
            core.dirty_stack[pos] = internal_index;
        else
            printf("[WARNING] dispatch_packet: dirty_count(%d) < MAX_ENTITIES(%d)\n", pos, MAX_ENTITIES);
    }
}

// --- PUBLIC API ---//

void server_init()
{
    for (uint32_t index = 0u; index < MAX_ID; ++index)
        core.sparse_lookup[index] = UINT32_MAX;

    uint32_t value = MAX_ENTITIES;
    for (uint32_t index = 0u; index < MAX_ENTITIES; ++index)
        core.free_indices[index] = --value;

    atomic_store(&core.free_count, MAX_ENTITIES);

    size_t float_size = MAX_ENTITIES * sizeof(float);
    size_t int_size = MAX_ENTITIES * sizeof(int);

    for (uint32_t i = 0u; i < 2u; ++i)
    {
        alloc_pinned_buffer((void **)&ecs_pos_x[i], (void **)&d_ecs_pos_x[i], float_size);
        alloc_pinned_buffer((void **)&ecs_pos_y[i], (void **)&d_ecs_pos_y[i], float_size);
        alloc_pinned_buffer((void **)&ecs_pos_z[i], (void **)&d_ecs_pos_z[i], float_size);
        alloc_pinned_buffer((void **)&ecs_health[i], (void **)&d_ecs_health[i], int_size);
    }

    CUDA_CHECK(cudaEventCreate(&chrono.start_event));
    CUDA_CHECK(cudaEventCreate(&chrono.stop_event));
}

void server_cleanup()
{
    for (uint32_t i = 0u; i < 2u; ++i)
    {
        CUDA_CHECK(cudaFreeHost(ecs_pos_x[i]));
        CUDA_CHECK(cudaFreeHost(ecs_pos_y[i]));
        CUDA_CHECK(cudaFreeHost(ecs_pos_z[i]));
        CUDA_CHECK(cudaFreeHost(ecs_health[i]));
    }

    CUDA_CHECK(cudaEventDestroy(chrono.start_event));
    CUDA_CHECK(cudaEventDestroy(chrono.stop_event));
}

uint32_t create_entity(uint32_t public_id)
{
    uint32_t pos = atomic_fetch_sub(&core.free_count, 1u);
    if (pos == 0 || pos - 1u >= MAX_ENTITIES)
        return printf("[ERROR] create_entity: pos(%d) == 0 || (pos - 1u)(%d) >= MAX_ENTITIES(%d)\n", pos, (pos - 1u), MAX_ENTITIES), UINT32_MAX;

    uint32_t internal_index = core.free_indices[pos - 1u];
    uint32_t generation = core.entity_generations[internal_index];
    uint32_t smart_id = (generation << INDEX_BITS) | internal_index;

    core.sparse_lookup[public_id] = internal_index;
    return smart_id;
}

void destroy_entity(uint32_t public_id)
{
    if (public_id >= MAX_ID)
        return (void)printf("[ERROR] destroy_entity: public_id(%d) >= MAX_ID(%d)\n", public_id, MAX_ID);

    uint32_t internal_index = core.sparse_lookup[public_id];

    if (internal_index >= MAX_ENTITIES)
        return (void)printf("[ERROR] destroy_entity: internal_index(%d) >= MAX_ENTITIES(%d)\n", internal_index, MAX_ENTITIES);

    core.sparse_lookup[public_id] = UINT32_MAX;
    core.entity_generations[public_id]++;

    uint32_t pos = atomic_fetch_add(&core.free_count, 1);

    if (pos < MAX_ENTITIES)
        core.free_indices[pos] = internal_index;
    else
        printf("[ERROR] destroy_entity: pos(%d) >= MAX_ENTITIES(%d)\n", pos, MAX_ENTITIES);
}

bool is_entity_valid(uint32_t smart_id)
{
    uint16_t index = get_entity_id(smart_id);
    uint32_t gen = get_entity_generation(smart_id);

    if (index >= MAX_ENTITIES)
        return false;
    return core.entity_generations[index] == gen;
}

void swap_buffers()
{
    uint32_t old_w = atomic_load(&core.write_idx);
    uint32_t next_w = old_w ^ 1u;
    atomic_store(&core.write_idx, next_w);

    uint32_t count = atomic_load(&core.dirty_count);

    for (uint32_t stack_index = 0u; stack_index < count; ++stack_index)
    {
        uint32_t entity_index = core.dirty_stack[stack_index];

        ecs_pos_x[next_w][entity_index] = ecs_pos_x[old_w][entity_index];
        ecs_pos_y[next_w][entity_index] = ecs_pos_y[old_w][entity_index];
        ecs_pos_z[next_w][entity_index] = ecs_pos_z[old_w][entity_index];

        ecs_health[next_w][entity_index] = ecs_health[old_w][entity_index];

        atomic_store(&core.is_dirty[entity_index], false);
    }

    atomic_store(&core.dirty_count, 0u);

    // Ici, on enverrait normalement un signal au GPU pour lui dire :
    // "Hé, tu peux lire le buffer qui vient d'être rempli !"
}

void consume_packets(NetworkRingBuffer *ring)
{
    uint32_t tail = atomic_load(&ring->tail);
    uint32_t head = atomic_load(&ring->head);

    while (tail != head)
    {
        dispatch_packet(&ring->packets[tail]);
        tail = (tail + 1u) & (RING_SIZE - 1u);
        atomic_store(&ring->tail, tail);
    }
}

void get_render_pointers(float **out_x, float **out_y, float **out_z)
{
    uint32_t read_i = atomic_load(&core.write_idx) ^ 1u;

    if (!out_x || !out_y || !out_z)
        return (void)printf("[ERROR] get_render_pointers: !out_x(%p) || !out_y(%p) || !out_z(%p)\n", out_x, out_y, out_z);

    *out_x = ecs_pos_x[read_i];
    *out_y = ecs_pos_y[read_i];
    *out_z = ecs_pos_z[read_i];
}

void get_health_pointer(int **out_health)
{
    uint32_t read_i = atomic_load(&core.write_idx) ^ 1u;

    if (!out_health)
        return (void)printf("[ERROR] get_health_pointer: !out_health(%p)\n", out_health);
    *out_health = ecs_health[read_i];
}

void get_gpu_pointers(float **dev_x, float **dev_y, float **dev_z)
{
    uint32_t read_i = atomic_load(&core.write_idx) ^ 1u;

    if (!dev_x || !dev_y || !dev_z)
        return (void)printf("[ERROR] get_gpu_pointers: !dev_x(%p) || !dev_y(%p) || !dev_z(%p)\n", dev_x, dev_y, dev_z);

    *dev_x = d_ecs_pos_x[read_i];
    *dev_y = d_ecs_pos_y[read_i];
    *dev_z = d_ecs_pos_z[read_i];
}

void run_physics_gpu(float delta_time)
{
    uint32_t write_i = atomic_load(&core.write_idx);

    int threadsPerBlock = 256;

    int blocksPerGrid = (MAX_ENTITIES + threadsPerBlock - 1) / threadsPerBlock;

    kernel_physics_update<<<blocksPerGrid, threadsPerBlock>>>(d_ecs_pos_y[write_i], MAX_ENTITIES, delta_time);
    cudaDeviceSynchronize();
}
