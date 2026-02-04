#ifndef PLUGIN_H
#define PLUGIN_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#define MAX_ENTITIES 10000
#define MAX_ID 1000000
#define RING_SIZE 4096
#define MAX_PACKET_SIZE 256

typedef enum {
    COMP_TRANSFORM = 1,
    COMP_HEALTH    = 2
} ComponentID;

typedef struct {
    uint8_t data[MAX_PACKET_SIZE];
    uint16_t size;
} DynamicPacket;

typedef struct {
    DynamicPacket packets[RING_SIZE];
    atomic_uint head;
    atomic_uint tail;
} NetworkRingBuffer;

// Public API
void server_init(void);
void server_cleanup(void);
uint32_t create_entity(uint32_t public_id);
void destroy_entity(uint32_t public_id);
bool is_entity_valid(uint32_t smart_id);
void swap_buffers(void);
void consume_packets(NetworkRingBuffer *ring);
void get_render_pointers(float **out_x, float **out_y, float **out_z);
void get_health_pointer(int **out_health);
void get_gpu_pointers(float **dev_x, float **dev_y, float **dev_z);
void run_physics_gpu(float delta_time);

// Inline helpers (exposés pour main.c)
#define INDEX_BITS 14
#define INDEX_MASK 0x3FFF

static inline uint16_t get_entity_id(uint32_t entity)
{
    return entity & INDEX_MASK;
}

static inline uint32_t get_entity_generation(uint32_t entity)
{
    return entity >> INDEX_BITS;
}

#endif // PLUGIN_H
