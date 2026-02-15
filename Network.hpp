// --- LAPLACE NETWORK DISPATCH --- //
// File: Network.hpp
// Description: Routage des paquets réseau + gestion du driver kernel
// Auteur: MasterLaplace

#pragma once

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <cstring>

#include "lpl_protocol.h"
#include "WorldPartition.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

class Network {
public:

    /// <summary>
    /// Represents the event ID.
    /// </summary>
    enum class EventId : uint8_t
    {
        MOVE_LEFT = 0,
        MOVE_RIGHT = 1,
        MOVE_UP = 2,
        MOVE_DOWN = 3,
        MOVE_FRONT = 4,
        MOVE_BACK = 5,
        LOOK_LEFT = 6,
        LOOK_RIGHT = 7,
        LOOK_UP = 8,
        LOOK_DOWN = 9,
        SHOOT = 10,
        MAX_EVENT
    };

    /// <summary>
    /// Represents the state of an event.
    /// </summary>
    enum class EventState : uint8_t
    {
        PRESSED = 0,
        RELEASED = 1,
        MAX_STATE
    };

    struct ButtonEvent {
        EventId entityId;
        EventState direction;
    };

    struct JoyStickEvent {
        EventId entityId;
        float axis;
    };

    struct ClientEndpoint {
        uint32_t ip;
        uint16_t port;
        uint32_t entityId;
    };

public:
    /**
     * @brief Ouvre /dev/lpl_driver et mmap le ring buffer partagé.
     *
     * @return Pointeur vers le NetworkRingBuffer mappé, ou nullptr en cas d'erreur.
     */
    bool network_init()
    {
        _driverFd = open("/dev/" LPL_DEVICE_NAME, O_RDWR);
        if (_driverFd < 0)
        {
            perror("[ERROR] network_init: Impossible d'ouvrir /dev/" LPL_DEVICE_NAME);
            return false;
        }

        void *adrr = mmap(nullptr, sizeof(LplSharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, _driverFd, 0);
        if (adrr == MAP_FAILED)
        {
            perror("[ERROR] network_init: Erreur de mappage mémoire");
            close(_driverFd);
            _driverFd = -1;
            return false;
        }

        _shm = static_cast<LplSharedMemory *>(adrr);
        _rx = &_shm->rx;
        _tx = &_shm->tx;
        _rx->idx.tail = _rx->idx.head;

        printf("[NET] Driver connecté. Ring Buffer mappé à %p (%zu bytes)\n", _shm, sizeof(LplSharedMemory));
        return true;
    }

    /**
     * @brief Libère le mapping mémoire du ring buffer.
     *
     * @param ring Pointeur obtenu via network_init().
     */
    void network_cleanup()
    {
        if (_shm && _shm != MAP_FAILED)
            munmap(_shm, sizeof(LplSharedMemory));

        if (_driverFd >= 0)
        {
            close(_driverFd);
            _driverFd = -1;
        }
    }

    /**
     * @brief Consomme les paquets du ring buffer et les dispatch dans le WorldPartition.
     *
     * @note Prevent infinite loop if head is moving
     *
     * @param world Le WorldPartition cible.
     */
    void network_consume_packets(WorldPartition &world)
    {
        if (!_rx)
            return;

        uint32_t rx_head = smp_load_acquire(&_rx->idx.head);
        uint32_t rx_tail = _rx->idx.tail;

        int loop_safety = RING_SLOTS * 2;
        while (rx_head != rx_tail && loop_safety-- > 0)
        {
            dispatch_packet(&_rx->packets[rx_tail], world);
            rx_tail = (rx_tail + 1u) & (RING_SLOTS - 1u);
            smp_store_release(&_rx->idx.tail, rx_tail);
            rx_head = smp_load_acquire(&_rx->idx.head);
        }
        _rx->idx.tail = rx_tail;
    }

    // --- Server Side API ---

    void broadcast_state(WorldPartition &world)
    {
        if (_clients.empty()) return;

        uint32_t readIdx = world.getReadIdx();
        constexpr size_t MAX_UDP = MAX_PACKET_SIZE; // Limited by ring buffer packet size
        // Wrapper for packet data
        uint8_t pkt[MAX_UDP];

        // Protocol: [MSG_STATE (1)][Count (2)][EntityData...]
        // EntityData: [ID (4)][Pos (12)][Size (12)][HP (4)] = 32 bytes
        constexpr size_t HEADER_SIZE = 3u;
        constexpr size_t ENTITY_SIZE = 32u;
        constexpr size_t MAX_ENTITIES_PER_PACKET = (MAX_UDP - HEADER_SIZE) / ENTITY_SIZE;

        uint16_t count = 0u;
        uint8_t *cursor = pkt + HEADER_SIZE;

        auto flush_packet = [&](bool force = false) {
            if (count == 0 && !force) return;

            pkt[0] = MSG_STATE;
            *reinterpret_cast<uint16_t*>(pkt + 1) = count;
            uint16_t len = static_cast<uint16_t>(cursor - pkt);

            for (const auto& client : _clients) {
                send_packet(client.ip, client.port, len, pkt);
            }

            count = 0u;
            cursor = pkt + HEADER_SIZE;
        };

        world.forEachChunk([&](Partition &p) {
            for (size_t i = 0; i < p.getEntityCount(); ++i)
            {
                if (count >= MAX_ENTITIES_PER_PACKET)
                    flush_packet();

                auto ent = p.getEntity(i, readIdx);
                uint32_t entId = p.getEntityId(i);

                *reinterpret_cast<uint32_t*>(cursor) = entId; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.position.x; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.position.y; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.position.z; cursor += 4;

                *reinterpret_cast<float*>(cursor) = ent.size.x; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.size.y; cursor += 4;
                *reinterpret_cast<float*>(cursor) = ent.size.z; cursor += 4;

                *reinterpret_cast<int32_t*>(cursor) = ent.health; cursor += 4;

                count++;
            }
        });

        if (count > 0)
            flush_packet();
    }

    // --- Client Side API ---

    void send_connect(const char* ip_str, uint16_t port)
    {
        uint32_t ip;
        inet_pton(AF_INET, ip_str, &ip);
        // host byte order for send_packet? No, send_packet expects host order for convenience or network?
        // Let's assume send_packet takes HOST byte order for IP/Port and converts it.
        // inet_pton returns NETWORK byte order.
        // My send_packet does `dst_ip = htonl(ip)`.
        // So I should pass HOST byte order to send_packet.
        // `inet_pton` -> Network Order (Big Endian).
        // `ntohl` -> Host Order.

        send_packet(ntohl(ip), port, 1u, (uint8_t*)&(uint8_t){MSG_CONNECT});
    }

    void send_input(uint32_t entityId, const Vec3& dir)
    {
        // MSG_INPUT: [Type(1)][ID(4)][Dir(12)]
        uint8_t pkt[17];
        pkt[0] = MSG_INPUT;
        *reinterpret_cast<uint32_t*>(pkt + 1) = entityId;
        *reinterpret_cast<float*>(pkt + 5) = dir.x;
        *reinterpret_cast<float*>(pkt + 9) = dir.y;
        *reinterpret_cast<float*>(pkt + 13) = dir.z;

        // Assuming we are connected to someone, but client needs to know where to send.
        // For now, let's assume we store the server address or just broadcast?
        // Client usually sends to a specific server.
        // Let's add `_serverIp` and `_serverPort` to Network class?
        // Or just pass it in.
        if (_serverPort != 0u)
            send_packet(_serverIp, _serverPort, 17, pkt);
    }

    void set_server_info(const char* ip_str, uint16_t port)
    {
        uint32_t ip_n;
        inet_pton(AF_INET, ip_str, &ip_n);
        _serverIp = ntohl(ip_n);
        _serverPort = port;
    }

    uint32_t get_local_entity_id() const { return _localEntityId; }
    bool is_connected() const { return _connected; }

private:
    void dispatch_packet(RxPacket *pkt, WorldPartition &world)
    {
        uint8_t *cursor = pkt->data;
        if (pkt->length < 1u)
            return;
        uint8_t msg_type = *cursor;

        // Convert network endian to host for IP/Port (stored in RxPacket in network order? No compy_bits just copies bytes)
        // Wait, lpl_kmod puts `ip->saddr` (Network Byte Order) into `pkt->src_ip`.
        uint32_t src_ip_h = ntohl(pkt->src_ip);
        uint16_t src_port_h = ntohs(pkt->src_port);

        switch (msg_type)
        {
        case MSG_CONNECT: {
            handle_connect(world, src_ip_h, src_port_h);
            break;
        }
        case MSG_INPUT: {
             if (pkt->length < 17) break;
             uint32_t eid = *reinterpret_cast<uint32_t*>(cursor + 1);
             Vec3 dir{
                 *reinterpret_cast<float*>(cursor + 5),
                 *reinterpret_cast<float*>(cursor + 9),
                 *reinterpret_cast<float*>(cursor + 13)
             };
             handle_input(world, eid, dir);
             break;
        }
        case MSG_WELCOME: {
            if (pkt->length < 5) break;
            _localEntityId = *reinterpret_cast<uint32_t*>(cursor + 1);
            _connected = true;
            printf("[NET] Connected! Entity ID: %u\n", _localEntityId);
            break;
        }
        case MSG_STATE: {
            if (pkt->length < 3) break;
            uint16_t count = *reinterpret_cast<uint16_t*>(cursor + 1);
            dispatch_state_update(world, count, cursor + 3, pkt->length - 3);
            break;
        }
        default:
            // Could be component based packet?
            // dispatch_packet_world(world, pkt->data);
            break;
        }
    }

    void handle_connect(WorldPartition &world, uint32_t client_ip, uint16_t client_port)
    {
        for (auto &c : _clients)
            if (c.ip == client_ip && c.port == client_port)
                return;

        uint32_t newId = _nextEntityId++;

        // Add player entity
        Partition::EntitySnapshot ent{};
        ent.id = newId;
        ent.position = {0.f, 10.f, 0.f};
        ent.rotation = {0.f, 0.f, 0.f, 1.f};
        ent.velocity = {0.f, 0.f, 0.f};
        ent.mass = 1.f;
        ent.force = {0.f, 0.f, 0.f};
        ent.size = {1.f, 2.f, 1.f};
        ent.health = 100;
        world.addEntity(ent);

        _clients.push_back({client_ip, client_port, newId});

        // Send Welcome
        uint8_t resp[5u];
        resp[0u] = MSG_WELCOME;
        *reinterpret_cast<uint32_t*>(resp + 1u) = newId;
        send_packet(client_ip, client_port, 5u, resp);

        printf("[NET] Client connected: %u -> Entity %u\n", client_ip, newId);
    }

    void handle_input(WorldPartition &world, uint32_t entityId, const Vec3 &dir)
    {
        uint32_t writeIdx = world.getWriteIdx();
        int localIdx = -1;
        Partition *chunk = world.findEntity(entityId, localIdx);
        if (chunk && localIdx >= 0)
        {
            constexpr float PLAYER_SPEED = 50.0f;
            Vec3 vel = dir * PLAYER_SPEED;
            chunk->setVelocity(static_cast<uint32_t>(localIdx), vel, writeIdx);
            chunk->wakeEntity(static_cast<uint32_t>(localIdx));
        }
    }

    void dispatch_state_update(WorldPartition &world, uint16_t count, uint8_t *data, uint32_t len)
    {
        uint8_t *cursor = data;
        uint32_t bytesRead = 0u;
        uint32_t writeIdx = world.getWriteIdx();

        for (uint16_t i = 0u; i < count; ++i) {
            if (bytesRead + 32u > len) break;

            uint32_t id = *reinterpret_cast<uint32_t*>(cursor); cursor += 4;
            Vec3 pos{ *reinterpret_cast<float*>(cursor), *reinterpret_cast<float*>(cursor+4), *reinterpret_cast<float*>(cursor+8)}; cursor += 12;
            Vec3 size{ *reinterpret_cast<float*>(cursor), *reinterpret_cast<float*>(cursor+4), *reinterpret_cast<float*>(cursor+8)}; cursor += 12;
            int32_t hp = *reinterpret_cast<int32_t*>(cursor); cursor += 4;
            bytesRead += 32u;

            int localIdx = -1;
            Partition *chunk = world.findEntity(id, localIdx);

            /// Simple client-side reconciliation / update
            if (chunk && localIdx >= 0)
            {
                chunk->setPosition(static_cast<uint32_t>(localIdx), pos, writeIdx);
                chunk->setSize(static_cast<uint32_t>(localIdx), size);
                chunk->setHealth(static_cast<uint32_t>(localIdx), hp);
            }
            else
            {
                Partition::EntitySnapshot snap{};
                snap.id = id;
                snap.position = pos;
                snap.size = size;
                snap.health = hp;
                snap.mass = 1.0f; // default
                snap.rotation = {0,0,0,1};
                world.addEntity(snap);
            }
        }
    }

    void send_packet(uint32_t ip, uint16_t port, uint16_t length, uint8_t *data)
    {
        if (!_rx)
            return;

        uint32_t tx_tail = _tx->idx.tail;
        uint32_t tx_head = smp_load_acquire(&_tx->idx.head);
        uint32_t tx_next_tail = (tx_tail + 1u) & (RING_SLOTS - 1u);

        if (tx_next_tail != tx_head)
        {
            TxPacket *tx_pkt = &_tx->packets[tx_tail];
            tx_pkt->dst_ip = htonl(ip);
            tx_pkt->dst_port = htons(port);
            tx_pkt->length = length;
            memcpy(tx_pkt->data, data, (length > MAX_PACKET_SIZE) ? MAX_PACKET_SIZE: length);

            _tx->idx.tail = tx_next_tail;
            smp_store_release(&_tx->idx.tail, tx_next_tail);
            ioctl(_driverFd, LPL_IOC_KICK_TX, 0);
        }
        else
        {
            fprintf(stderr, "[NET] TX Ring Full\n");
        }
    }

private:
    LplSharedMemory *_shm = nullptr;
    RxRingBuffer *_rx = nullptr;
    TxRingBuffer *_tx = nullptr;
    uint32_t _nextEntityId = 100u; // Reserved 0-99
    int _driverFd = -1;

    std::vector<ClientEndpoint> _clients;

    // Client State
    uint32_t _serverIp = 0;
    uint16_t _serverPort = 0;
    uint32_t _localEntityId = 0;
    bool _connected = false;
    int _driverFd = -1;
};
