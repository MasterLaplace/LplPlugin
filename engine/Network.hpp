// File: Network.hpp
// Description: Transport réseau pur — envoi/réception de paquets + sérialisation/désérialisation.
//              Dispatch vers des queues typées (PacketQueue) pour consommation par les systèmes ECS.
//              Ne contient plus de logique métier (inputs, physique, session).
//              Fallback socket côté serveur pour WSL.
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
#include <vector>
#include <algorithm>
#include <iostream>

#ifdef LPL_USE_SOCKET
    #include <sys/socket.h>
    #include <netinet/in.h>
#endif

#include "lpl_protocol.h"
#include "PacketQueue.hpp"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "Protocol"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define PRINTER(...) LOGI(__VA_ARGS__)
#else
#define PRINTER(...) printf(__VA_ARGS__)
#endif

/**
 * @brief Transport réseau — envoi/réception/sérialisation de paquets UDP.
 *
 * Trois modes :
 *   - LPL_USE_SOCKET : socket UDP standard (client)
 *   - Sinon (serveur) : essaie le driver kernel en premier, fallback sur socket UDP si indisponible (WSL, dev)
 *
 * network_consume_packets() désérialise les paquets et les pousse dans
 * des PacketQueue typées. Les systèmes ECS consomment ensuite via drain().
 *
 * Ne contient plus de logique de session, d'inputs ou de physique.
 */
class Network {
public:
    bool network_init()
    {
#ifdef LPL_USE_SOCKET
        return init_socket_mode();
#else
        // Server: try kernel driver first, fallback to socket mode
        if (init_kernel_mode())
        {
            _useKernelDriver = true;
            return true;
        }

        printf("[NET] Kernel driver unavailable, falling back to socket mode (WSL/dev)\n");
        if (init_socket_mode())
        {
            _useKernelDriver = false;
            return true;
        }

        return false;
#endif
    }

    void network_cleanup()
    {
#ifdef LPL_USE_SOCKET
        cleanup_socket_mode();
#else
        // Cleanup whichever mode was active
        if (_useKernelDriver)
            cleanup_kernel_mode();
        else
            cleanup_socket_mode();
#endif
    }

    void network_consume_packets(PacketQueue &queue)
    {
#ifdef LPL_USE_SOCKET
        consume_packets_socket(queue);
#else
        if (_useKernelDriver)
            consume_packets_kernel(queue);
        else
            consume_packets_socket(queue);
#endif
    }

    void send_packet(uint32_t ip, uint16_t port, uint16_t length, uint8_t *data)
    {
#ifdef LPL_USE_SOCKET
        send_packet_socket(ip, port, length, data);
#else
        if (_useKernelDriver)
            send_packet_kernel(ip, port, length, data);
        else
            send_packet_socket(ip, port, length, data);
#endif
    }

    // ─── Packet Construction Helpers ─────────────────────────

    /**
     * @brief Envoie un MSG_CONNECT au serveur.
     */
    void send_connect(const char* ip_str, uint16_t port)
    {
        uint32_t ip;
        inet_pton(AF_INET, ip_str, &ip); // Network Order
        uint8_t type = MSG_CONNECT;
        send_packet(ntohl(ip), port, 1u, &type);
    }

    /**
     * @brief Envoie un MSG_WELCOME à un client.
     */
    void send_welcome(uint32_t clientIp, uint16_t clientPort, uint32_t entityId)
    {
        uint8_t resp[5u];
        resp[0u] = MSG_WELCOME;
        *reinterpret_cast<uint32_t*>(resp + 1u) = entityId;
        send_packet(clientIp, clientPort, 5u, resp);
    }

    /**
     * @brief Sérialise et envoie les inputs au serveur.
     */
    void send_inputs(uint32_t entityId, const std::vector<uint8_t> &keyData, const std::vector<uint8_t> &axisData, const std::vector<uint8_t> &neuralData)
    {
        uint8_t pkt[MAX_PACKET_SIZE];
        pkt[0] = MSG_INPUTS;
        *reinterpret_cast<uint32_t*>(pkt + 1u) = entityId;

        uint8_t *cursor = pkt + 5u;
        size_t totalLen = 5u;

        uint8_t keyCount = static_cast<uint8_t>(keyData.size() / 2u);
        *cursor = keyCount; cursor++; totalLen++;
        if (keyData.size() > 0u)
        {
            memcpy(cursor, keyData.data(), keyData.size());
            cursor += keyData.size();
            totalLen += keyData.size();
        }

        uint8_t axisCount = static_cast<uint8_t>(axisData.size() / 5u);
        *cursor = axisCount; cursor++; totalLen++;
        if (axisData.size() > 0u)
        {
            memcpy(cursor, axisData.data(), axisData.size());
            cursor += axisData.size();
            totalLen += axisData.size();
        }

        if (neuralData.size() >= 13u)
        {
            memcpy(cursor, neuralData.data(), 13u);
            cursor += 13u;
            totalLen += 13u;
        }

        if (_serverPort != 0u && totalLen <= MAX_PACKET_SIZE && totalLen >= 7u)
            send_packet(_serverIp, _serverPort, (uint16_t)totalLen, pkt);
    }

    /**
     * @brief Configure l'adresse du serveur (côté client).
     */
    void set_server_info(const char* ip_str, uint16_t port)
    {
        uint32_t ip_n;
        inet_pton(AF_INET, ip_str, &ip_n);
        _serverIp = ntohl(ip_n);
        _serverPort = port;
    }

private:
    // ─── Socket Mode Implementation ──────────────────────────

    bool init_socket_mode()
    {
        _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (_sockfd < 0) {
            perror("[ERROR] init_socket_mode: Socket creation failed");
            return false;
        }

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
#ifdef LPL_USE_SOCKET
        addr.sin_port = 0; // Ephemeral port (client)
#else
        addr.sin_port = htons(LPL_PORT); // Fixed port 7777 (server)
#endif

        if (bind(_sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("[ERROR] init_socket_mode: Bind failed");
            close(_sockfd);
            _sockfd = -1;
            return false;
        }

        // Get assigned port
        socklen_t len = sizeof(addr);
        if (getsockname(_sockfd, (struct sockaddr*)&addr, &len) == 0) {
#ifdef LPL_USE_SOCKET
            printf("[NET] Socket UDP init sur le port %d (Mode CLIENT)\n", ntohs(addr.sin_port));
#else
            printf("[NET] Socket UDP init sur le port %d (Mode SERVER fallback)\n", ntohs(addr.sin_port));
#endif
        }

        // Non-blocking mode
        int flags = fcntl(_sockfd, F_GETFL, 0);
        fcntl(_sockfd, F_SETFL, flags | O_NONBLOCK);

        return true;
    }

    void cleanup_socket_mode()
    {
        if (_sockfd >= 0) {
            close(_sockfd);
            _sockfd = -1;
        }
    }

    void consume_packets_socket(PacketQueue &queue)
    {
        if (_sockfd < 0) return;

        uint8_t buffer[MAX_PACKET_SIZE];
        struct sockaddr_in src_addr;
        socklen_t addr_len = sizeof(src_addr);

        while (true)
        {
            ssize_t len = recvfrom(_sockfd, buffer, MAX_PACKET_SIZE, 0, (struct sockaddr*)&src_addr, &addr_len);
            if (len <= 0) break;

            RxPacket pkt;
            pkt.src_ip = src_addr.sin_addr.s_addr; // Network Byte Order
            pkt.src_port = src_addr.sin_port;      // Network Byte Order
            pkt.length = (uint16_t)len;
            memcpy(pkt.data, buffer, len);

            dispatch_packet(&pkt, queue);
        }
    }

    void send_packet_socket(uint32_t ip, uint16_t port, uint16_t length, uint8_t *data)
    {
        if (_sockfd < 0) return;

        struct sockaddr_in dest_addr;
        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_addr.s_addr = htonl(ip); // htonl because send_packet expects Host Order
        dest_addr.sin_port = htons(port);      // htons because send_packet expects Host Order

        sendto(_sockfd, data, length, 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
    }

#ifndef LPL_USE_SOCKET
    // ─── Kernel Driver Mode Implementation ───────────────────

    bool init_kernel_mode()
    {
        _driverFd = open("/dev/" LPL_DEVICE_NAME, O_RDWR);
        if (_driverFd < 0)
        {
            return false;
        }

        void *adrr = mmap(nullptr, sizeof(LplSharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, _driverFd, 0);
        if (adrr == MAP_FAILED)
        {
            close(_driverFd);
            _driverFd = -1;
            return false;
        }

        _shm = static_cast<LplSharedMemory *>(adrr);
        _rx = &_shm->rx;
        _tx = &_shm->tx;

        PRINTER("[NET] Driver connecté. Ring Buffer mappé à %p (%zu bytes)\n", _shm, sizeof(LplSharedMemory));
        return true;
    }

    void cleanup_kernel_mode()
    {
        if (_shm && _shm != MAP_FAILED)
            munmap(_shm, sizeof(LplSharedMemory));

        if (_driverFd >= 0)
        {
            close(_driverFd);
            _driverFd = -1;
        }
    }

    void consume_packets_kernel(PacketQueue &queue)
    {
        if (!_rx)
            return;

        uint32_t rx_head = smp_load_acquire(&_rx->idx.head);
        uint32_t rx_tail = _rx->idx.tail;

        int loop_safety = RING_SLOTS * 2;
        while (rx_head != rx_tail && loop_safety-- > 0)
        {
            dispatch_packet(&_rx->packets[rx_tail], queue);
            rx_tail = (rx_tail + 1u) & (RING_SLOTS - 1u);
            smp_store_release(&_rx->idx.tail, rx_tail);
            rx_head = smp_load_acquire(&_rx->idx.head);
        }
        _rx->idx.tail = rx_tail;
    }

    void send_packet_kernel(uint32_t ip, uint16_t port, uint16_t length, uint8_t *data)
    {
        if (!_tx)
            return;

        uint32_t tx_tail = _tx->idx.tail;
        uint32_t tx_head = smp_load_acquire(&_tx->idx.head);
        uint32_t tx_next_tail = (tx_tail + 1u) & (RING_SLOTS - 1u);

        if (tx_next_tail != tx_head)
        {
            TxPacket *tx_pkt = &_tx->packets[tx_tail];
            tx_pkt->dst_ip = htonl(ip);
            tx_pkt->dst_port = htons(port);
            uint16_t safe_len = (length > MAX_PACKET_SIZE) ? MAX_PACKET_SIZE : length;
            tx_pkt->length = safe_len;
            memcpy(tx_pkt->data, data, safe_len);

            _tx->idx.tail = tx_next_tail;
            smp_store_release(&_tx->idx.tail, tx_next_tail);
            ioctl(_driverFd, LPL_IOC_KICK_TX, 0);
        }
        else
        {
            fprintf(stderr, "[NET] TX Ring Full\n");
        }
    }
#endif

    /**
     * @brief Désérialise un paquet brut et le pousse dans la queue appropriée.
     *
     * Pur routage — aucune logique métier.
     */
    void dispatch_packet(RxPacket *pkt, PacketQueue &queue)
    {
        uint8_t *cursor = pkt->data;
        if (pkt->length < 1u)
            return;
        uint8_t msg_type = *cursor;

        uint32_t src_ip_h = ntohl(pkt->src_ip);
        uint16_t src_port_h = ntohs(pkt->src_port);

        switch (msg_type)
        {
        case MSG_CONNECT: {
            queue.connects.push({src_ip_h, src_port_h});
            break;
        }
        case MSG_WELCOME: {
            if (pkt->length < 5) break;
            uint32_t entityId = *reinterpret_cast<uint32_t*>(cursor + 1);
            queue.welcomes.push({entityId});
            PRINTER("[NET] Received MSG_WELCOME: Entity ID %u\n", entityId);
            break;
        }
        case MSG_STATE: {
            if (pkt->length < 3) break;
            uint16_t count = *reinterpret_cast<uint16_t*>(cursor + 1);
            deserialize_state(queue, count, cursor + 3, pkt->length - 3);
            break;
        }
        case MSG_INPUTS: {
            if (pkt->length < 7) break;
            uint32_t eid = *reinterpret_cast<uint32_t*>(cursor + 1);
            deserialize_inputs(queue, eid, cursor + 5, pkt->length - 5);
            break;
        }
        default:
            break;
        }
    }

    /**
     * @brief Désérialise un MSG_STATE et pousse les entités dans la queue.
     */
    static void deserialize_state(PacketQueue &queue, const uint16_t count, uint8_t *data, const uint32_t len)
    {
        StateUpdateEvent event;
        event.entities.reserve(count);

        uint8_t *cursor = data;
        uint32_t bytesRead = 0u;

        for (uint16_t i = 0u; i < count; ++i)
        {
            if (bytesRead + 32u > len) break;

            StateEntityData ent;
            ent.id = *reinterpret_cast<uint32_t*>(cursor); cursor += 4;
            ent.pos = {
                *reinterpret_cast<float*>(cursor),
                *reinterpret_cast<float*>(cursor + 4),
                *reinterpret_cast<float*>(cursor + 8)
            }; cursor += 12;
            ent.size = {
                *reinterpret_cast<float*>(cursor),
                *reinterpret_cast<float*>(cursor + 4),
                *reinterpret_cast<float*>(cursor + 8)
            }; cursor += 12;
            ent.hp = *reinterpret_cast<int32_t*>(cursor); cursor += 4;
            bytesRead += 32u;

            event.entities.push_back(ent);
        }

        if (!event.entities.empty())
            queue.states.push(std::move(event));
    }

    /**
     * @brief Désérialise un MSG_INPUTS et pousse l'événement dans la queue.
     */
    static void deserialize_inputs(PacketQueue &queue, uint32_t entityId, uint8_t *data, uint32_t len)
    {
        InputEvent event;
        event.entityId = entityId;

        uint8_t *cursor = data;
        uint32_t remaining = len;

        // Keys
        if (remaining < 1u) { queue.inputs.push(std::move(event)); return; }
        uint8_t keyCount = *cursor; cursor++; remaining--;

        for (uint8_t i = 0u; i < keyCount; ++i)
        {
            if (remaining < 2u) break;
            uint16_t packed = *reinterpret_cast<uint16_t*>(cursor);
            uint16_t key = packed & 0x7FFF;
            bool state = (packed & 0x8000) != 0;
            event.keys.push_back({key, state});
            cursor += 2u; remaining -= 2u;
        }

        // Axes
        if (remaining < 1u) { queue.inputs.push(std::move(event)); return; }
        uint8_t axisCount = *cursor; cursor++; remaining--;

        for (uint8_t i = 0u; i < axisCount; ++i)
        {
            if (remaining < 5u) break;
            uint8_t axisId = *cursor;
            float value = *reinterpret_cast<float*>(cursor + 1);
            event.axes.push_back({axisId, value});
            cursor += 5u; remaining -= 5u;
        }

        // Neural data
        if (remaining >= 13u)
        {
            event.neural.alpha = *reinterpret_cast<float*>(cursor);
            event.neural.beta = *reinterpret_cast<float*>(cursor + 4);
            event.neural.concentration = *reinterpret_cast<float*>(cursor + 8);
            event.neural.blink = *(cursor + 12) != 0;
            event.hasNeural = true;
        }

        queue.inputs.push(std::move(event));
    }

private:
#ifdef LPL_USE_SOCKET
    int _sockfd = -1;
#else
    int _sockfd = -1;                          // Socket fallback (WSL/dev)
    bool _useKernelDriver = false;             // Flag to track which mode is active
    LplSharedMemory *_shm = nullptr;
    RxRingBuffer *_rx = nullptr;
    TxRingBuffer *_tx = nullptr;
    int _driverFd = -1;
#endif

    // Client-side server address
    uint32_t _serverIp = 0u;
    uint16_t _serverPort = 0u;
};
