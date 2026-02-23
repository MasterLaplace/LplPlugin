/*
** EPITECH PROJECT, 2026
** LplPlugin
** File description:
** Core
*/

#pragma once

#include <iostream>
#include <signal.h>
#include <stdexcept>
#include "Network.hpp"
#include "SystemScheduler.hpp"
#include "WorldPartition.hpp"

class Core {
public:
    Core()
    {
        std::cout << "=== LplPlugin Core Initialized ===\n";
        if (_instance)
        {
            std::cerr << "[ERROR] Core instance already exists\n";
            throw std::runtime_error("Core instance already exists");
        }
        signal(SIGINT, &Core::static_sigint_handler);
        _instance = this;
        gpu_init();

        if (!_network.network_init())
            throw std::runtime_error("[ERROR] Network init failed\n");
    }
    ~Core()
    {
        std::cout << "=== LplPlugin Core Shutting Down ===\n";
        _network.network_cleanup();
        gpu_cleanup();
    }

    void runServer()
    {
        if (_running)
        {
            std::cerr << "[ERROR] Server already running\n";
            return;
        }
        _running = true;
        while (_running)
        {
            uint64_t frame_start = get_time_ns();
            _scheduler.ordered_tick(_world, DELTA_TIME);
            _world.swapBuffers();
            _network.broadcast_state(_world);
            uint64_t elapsed = get_time_ns() - frame_start;
            if (elapsed < FRAME_TIME_NS)
            {
                struct timespec ts{0, static_cast<long>(FRAME_TIME_NS - elapsed)};
                nanosleep(&ts, nullptr);
            }
        }
    }

    void runClient()
    {
        if (_running)
        {
            std::cerr << "[ERROR] Client already running\n";
            return;
        }
        _running = true;
        while (_running)
        {
            _network.network_consume_packets(_world);
            _scheduler.ordered_tick(_world, DELTA_TIME);
            _world.swapBuffers();
        }
    }

    void initNetwork(const char *serverIp, uint16_t serverPort)
    {
        _network.set_server_info(serverIp, serverPort);
        _network.send_connect(serverIp, serverPort);

        std::cout << "[CLIENT] MSG_CONNECT sent to " << serverIp << ":" << serverPort << "\n";
    }

protected:
    [[nodiscard]] uint64_t get_time_ns()
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    }

    static void static_sigint_handler(int) { _instance->_running = false; }

private:
    static Core *_instance;
    static constexpr uint64_t FRAME_TIME_NS = 16666666;
    static constexpr float DELTA_TIME = 1.f / 60.f;
    volatile bool _running = false;
    WorldPartition _world;
    Network _network;
    SystemScheduler _scheduler;
};
