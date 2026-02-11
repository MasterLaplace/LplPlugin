// --- LAPLACE ENGINE --- //
// File: Engine.cu
// Description: Implémentation du moteur côté serveur (GPU)
// Auteur: MasterLaplace

#include "Engine.cuh"
#include "WorldPartition.hpp"
#include "Math.hpp"

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

// --- GPU Physics Timing ---
static cudaEvent_t g_physStart, g_physStop;
static bool g_physEventsCreated = false;

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

// --- NEW: Chunk-based physics kernel (Vec3 SoA) ---

/**
 * @brief Kernel de physique par chunk.
 * Applique la gravité + intégration Euler semi-implicite sur les vecteurs SoA.
 * Identique à Partition::physicsTick mais exécuté massivement en parallèle sur le GPU.
 */
__global__ void kernel_physics_tick(
    Vec3 *positions, Vec3 *velocities, Vec3 *forces, float *masses,
    uint32_t count, float dt)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    // Gravité
    forces[idx] = Vec3{0.0f, -9.81f * masses[idx], 0.0f};

    // Euler semi-implicite : v += a*dt, puis p += v*dt
    if (masses[idx] > 0.0001f)
    {
        float invMass = 1.0f / masses[idx];
        velocities[idx] += forces[idx] * invMass * dt;
    }

    positions[idx] += velocities[idx] * dt;
}

static void dispatch_packet(DynamicPacket *pkt)
{
    uint32_t current_w = lpl_atomic_load(&core.write_idx);
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
    if (lpl_atomic_exchange(&core.is_dirty[internal_index], true) == false)
    {
        uint32_t pos = lpl_atomic_fetch_add(&core.dirty_count, 1);

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

    lpl_atomic_store(&core.free_count, MAX_ENTITIES);

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
    uint32_t pos = lpl_atomic_fetch_sub(&core.free_count, 1u);
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
    core.entity_generations[internal_index]++;

    uint32_t pos = lpl_atomic_fetch_add(&core.free_count, 1);

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
    uint32_t old_w = lpl_atomic_load(&core.write_idx);
    uint32_t next_w = old_w ^ 1u;
    lpl_atomic_store(&core.write_idx, next_w);

    uint32_t count = lpl_atomic_load(&core.dirty_count);

    for (uint32_t stack_index = 0u; stack_index < count; ++stack_index)
    {
        uint32_t entity_index = core.dirty_stack[stack_index];

        ecs_pos_x[next_w][entity_index] = ecs_pos_x[old_w][entity_index];
        ecs_pos_y[next_w][entity_index] = ecs_pos_y[old_w][entity_index];
        ecs_pos_z[next_w][entity_index] = ecs_pos_z[old_w][entity_index];

        ecs_health[next_w][entity_index] = ecs_health[old_w][entity_index];

        lpl_atomic_store(&core.is_dirty[entity_index], false);
    }

    lpl_atomic_store(&core.dirty_count, 0u);

    // Ici, on enverrait normalement un signal au GPU pour lui dire :
    // "Hé, tu peux lire le buffer qui vient d'être rempli !"
}

void consume_packets(NetworkRingBuffer *ring)
{
    uint32_t tail = lpl_atomic_load(&ring->tail);
    uint32_t head = lpl_atomic_load(&ring->head);

    while (tail != head)
    {
        dispatch_packet(&ring->packets[tail]);
        tail = (tail + 1u) & (RING_SIZE - 1u);
    }

    lpl_atomic_store(&ring->tail, tail);
}

void get_render_pointers(float **out_x, float **out_y, float **out_z)
{
    uint32_t read_i = lpl_atomic_load(&core.write_idx) ^ 1u;

    if (!out_x || !out_y || !out_z)
        return (void)printf("[ERROR] get_render_pointers: !out_x(%p) || !out_y(%p) || !out_z(%p)\n", out_x, out_y, out_z);

    *out_x = ecs_pos_x[read_i];
    *out_y = ecs_pos_y[read_i];
    *out_z = ecs_pos_z[read_i];
}

void get_health_pointer(int **out_health)
{
    uint32_t read_i = lpl_atomic_load(&core.write_idx) ^ 1u;

    if (!out_health)
        return (void)printf("[ERROR] get_health_pointer: !out_health(%p)\n", out_health);
    *out_health = ecs_health[read_i];
}

void get_gpu_pointers(float **dev_x, float **dev_y, float **dev_z)
{
    uint32_t read_i = lpl_atomic_load(&core.write_idx) ^ 1u;

    if (!dev_x || !dev_y || !dev_z)
        return (void)printf("[ERROR] get_gpu_pointers: !dev_x(%p) || !dev_y(%p) || !dev_z(%p)\n", dev_x, dev_y, dev_z);

    *dev_x = d_ecs_pos_x[read_i];
    *dev_y = d_ecs_pos_y[read_i];
    *dev_z = d_ecs_pos_z[read_i];
}

void run_physics_gpu(float delta_time)
{
    uint32_t write_i = lpl_atomic_load(&core.write_idx);

    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_ENTITIES + threadsPerBlock - 1) / threadsPerBlock;

#ifdef LPL_MONITORING
    CUDA_CHECK(cudaEventRecord(chrono.start_event, 0));
#endif

    kernel_physics_update<<<blocksPerGrid, threadsPerBlock>>>(d_ecs_pos_y[write_i], MAX_ENTITIES, delta_time);
    cudaDeviceSynchronize();

#ifdef LPL_MONITORING
    CUDA_CHECK(cudaEventRecord(chrono.stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(chrono.stop_event));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, chrono.start_event, chrono.stop_event));

    // Log GPU performance tous les 100 frames seulement
    static int log_counter = 0;
    if (log_counter++ % 100 == 0)
    {
        printf("[GPU] run_physics_gpu: %.3f ms (%.0f µs) | %d entities | %.2f M entities/sec\n",
               milliseconds,
               milliseconds * 1000.0f,
               MAX_ENTITIES,
               (float)MAX_ENTITIES / milliseconds / 1000.0f);
    }
#endif
}

// ==========================================================================
// NEW UNIFIED GPU API (WorldPartition-based)
// ==========================================================================

void engine_init()
{
    // Création d'événements pour le timing GPU
    if (!g_physEventsCreated)
    {
        CUDA_CHECK(cudaEventCreate(&g_physStart));
        CUDA_CHECK(cudaEventCreate(&g_physStop));
        g_physEventsCreated = true;
    }
}

void engine_cleanup()
{
    if (g_physEventsCreated)
    {
        CUDA_CHECK(cudaEventDestroy(g_physStart));
        CUDA_CHECK(cudaEventDestroy(g_physStop));
        g_physEventsCreated = false;
    }
}

void engine_physics_tick(WorldPartition &world, float dt)
{
    static constexpr int THREADS_PER_BLOCK = 256;

#ifdef LPL_MONITORING
    CUDA_CHECK(cudaEventRecord(g_physStart, 0));
#endif

    // Phase 1 : Lancement des kernels GPU par chunk
    world.forEachChunk([&](Partition &partition) {
        uint32_t count = static_cast<uint32_t>(partition.getEntityCount());
        if (count == 0u)
            return;

        // Obtenir les device pointers depuis la pinned memory
        Vec3  *d_pos = nullptr, *d_vel = nullptr, *d_forces = nullptr;
        float *d_masses = nullptr;

        CUDA_CHECK(cudaHostGetDevicePointer(&d_pos,    partition.getPositionsData(),  0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_vel,    partition.getVelocitiesData(), 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_forces, partition.getForcesData(),     0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_masses, partition.getMassesData(),     0));

        int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_physics_tick<<<blocks, THREADS_PER_BLOCK>>>(d_pos, d_vel, d_forces, d_masses, count, dt);
    });

    // Synchronisation GPU — toutes les entités de tous les chunks ont été mises à jour
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef LPL_MONITORING
    CUDA_CHECK(cudaEventRecord(g_physStop, 0));
    CUDA_CHECK(cudaEventSynchronize(g_physStop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_physStart, g_physStop));

    static int log_counter = 0;
    if (log_counter++ % 100 == 0)
        printf("[GPU] engine_physics_tick: %.3f ms\n", ms);
#endif

    // Phase 2 : Migration inter-chunk (CPU)
    world.migrateEntities();
}

// ==========================================================================
// NETWORK → WORLDPARTITION PIPELINE (Phase 4)
// ==========================================================================

/**
 * @brief Dispatch un paquet réseau vers le WorldPartition.
 * - Entité existante : mise à jour des composants en place.
 * - Entité inconnue : création automatique avec les composants du paquet.
 */
static void dispatch_packet_world(DynamicPacket *pkt, WorldPartition &world)
{
    uint8_t *cursor = pkt->data;
    uint32_t public_id = *(uint32_t *)cursor;
    cursor += sizeof(uint32_t);

    // Cherche l'entité dans le monde
    int localIdx = -1;
    Partition *chunk = world.findEntity(public_id, localIdx);

    if (chunk && localIdx >= 0)
    {
        // Entité existante → mise à jour des composants en place
        while (cursor < pkt->data + pkt->size)
        {
            uint8_t comp = *cursor++;
            switch (comp)
            {
            case COMP_TRANSFORM: {
                Vec3 pos{*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                chunk->setPosition(static_cast<uint32_t>(localIdx), pos);
                break;
            }
            case COMP_HEALTH: {
                int32_t hp = *(int32_t *)(cursor);
                cursor += 4;
                chunk->setHealth(static_cast<uint32_t>(localIdx), hp);
                break;
            }
            case COMP_VELOCITY: {
                Vec3 vel{*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                chunk->setVelocity(static_cast<uint32_t>(localIdx), vel);
                break;
            }
            case COMP_MASS: {
                float m = *(float *)(cursor);
                cursor += 4;
                chunk->setMass(static_cast<uint32_t>(localIdx), m);
                break;
            }
            default:
                printf("[WARNING] dispatch_packet_world: unknown component_id(%d)\n", comp);
                return;
            }
        }
    }
    else
    {
        // Nouvelle entité → parser en snapshot puis ajouter au monde
        Partition::EntitySnapshot snap{};
        snap.id = public_id;
        snap.rotation = {0.f, 0.f, 0.f, 1.f};
        snap.mass = 1.0f;
        snap.size = {1.f, 1.f, 1.f};
        snap.health = 100;

        while (cursor < pkt->data + pkt->size)
        {
            uint8_t comp = *cursor++;
            switch (comp)
            {
            case COMP_TRANSFORM:
                snap.position = {*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                break;
            case COMP_HEALTH:
                snap.health = *(int32_t *)(cursor);
                cursor += 4;
                break;
            case COMP_VELOCITY:
                snap.velocity = {*(float *)(cursor), *(float *)(cursor + 4), *(float *)(cursor + 8)};
                cursor += 12;
                break;
            case COMP_MASS:
                snap.mass = *(float *)(cursor);
                cursor += 4;
                break;
            default:
                printf("[WARNING] dispatch_packet_world: unknown component_id(%d) for new entity %u\n", comp, public_id);
                return;
            }
        }

        world.addEntity(snap);
    }
}

void engine_consume_packets(NetworkRingBuffer *ring, WorldPartition &world)
{
    uint32_t tail = lpl_atomic_load(&ring->tail);
    uint32_t head = lpl_atomic_load(&ring->head);

    while (tail != head)
    {
        dispatch_packet_world(&ring->packets[tail], world);
        tail = (tail + 1u) & (RING_SIZE - 1u);
    }

    lpl_atomic_store(&ring->tail, tail);
}
