/*
** EPITECH PROJECT, 2026
** LplPlugin
** File description:
** Core — Point d'entrée unifié de l'engine (composition).
**
** Classe unique utilisée côté serveur et client.
** Les applications enregistrent leurs systèmes spécifiques puis appellent run().
**
** Architecture :
**   Core possède : WorldPartition, Network, SystemScheduler, InputManager,
**                  PacketQueue, SessionManager (optionnel, serveur).
**   Les systèmes ECS (enregistrés via registerSystem) font toute la logique.
**   La boucle principale : PreSwap → swapBuffers → PostSwap → sleep.
*/

#pragma once

#include <atomic>
#include <iostream>
#include <signal.h>
#include <stdexcept>
#include "Network.hpp"
#include "SystemScheduler.hpp"
#include "WorldPartition.hpp"
#include "InputManager.hpp"
#include "PacketQueue.hpp"
#include "SessionManager.hpp"

/**
 * @brief Point d'entrée unifié de l'engine LplPlugin.
 *
 * Usage serveur :
 * @code
 *   Core core;
 *   core.registerSystem(Systems::NetworkReceiveSystem(core.network(), core.packetQueue()));
 *   core.registerSystem(Systems::SessionSystem(...));
 *   // ... autres systèmes serveur ...
 *   core.buildSchedule();
 *   core.run();
 * @endcode
 *
 * Usage client :
 * @code
 *   Core core;
 *   core.registerSystem(Systems::NetworkReceiveSystem(core.network(), core.packetQueue()));
 *   core.registerSystem(Systems::WelcomeSystem(core.packetQueue(), myEntityId, connected));
 *   // ... autres systèmes client ...
 *   core.buildSchedule();
 *   core.run();
 * @endcode
 */
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
        struct sigaction sa{};
        sa.sa_handler = &Core::static_sigint_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0; // No SA_RESTART → nanosleep interrupted immediately
        sigaction(SIGINT, &sa, nullptr);
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
        _instance = nullptr;
    }

    // ─── System Registration ─────────────────────────────────

    void registerSystem(SystemDescriptor desc)
    {
        _scheduler.registerSystem(std::move(desc));
    }

    void buildSchedule()
    {
        _scheduler.buildSchedule();
    }

    void printSchedule() const
    {
        _scheduler.printSchedule();
    }

    // ─── Main Loop ───────────────────────────────────────────

    /**
     * @brief Boucle principale unifiée.
     *
     * Flux par frame :
     *   1. PreSwap systems (réseau, inputs, physique, etc.)
     *   2. swapBuffers()
     *   3. PostSwap systems (broadcast, rendu, etc.)
     *   4. Sleep pour maintenir le framerate cible
     *
     * @param useThreaded  Si true, utilise le mode parallèle intra-stage.
     * @param onPostLoop   Callback optionnel appelé à la fin de chaque frame
     *                     (ex: glfwSwapBuffers, glfwPollEvents côté client).
     */
    void run(bool useThreaded = false)
    {
        run(useThreaded, [](float){});
    }

    template <typename PostLoopFn>
    void run(bool useThreaded, PostLoopFn onPostLoop)
    {
        if (_running)
        {
            std::cerr << "[ERROR] Core already running\n";
            return;
        }
        _running = true;

        while (_running)
        {
            uint64_t frame_start = get_time_ns();

            // PreSwap
            if (useThreaded)
                _scheduler.threaded_tick_pre_swap(_world, DELTA_TIME);
            else
                _scheduler.ordered_tick_pre_swap(_world, DELTA_TIME);

            // Swap double buffer
            _world.swapBuffers();

            // PostSwap
            if (useThreaded)
                _scheduler.threaded_tick_post_swap(_world, DELTA_TIME);
            else
                _scheduler.ordered_tick_post_swap(_world, DELTA_TIME);

            // Application-specific post-loop callback
            onPostLoop(DELTA_TIME);

            // Maintain target framerate
            uint64_t elapsed = get_time_ns() - frame_start;
            if (elapsed < FRAME_TIME_NS)
            {
                struct timespec ts{0, static_cast<long>(FRAME_TIME_NS - elapsed)};
                nanosleep(&ts, nullptr);
            }
        }
    }

    /**
     * @brief Boucle principale avec variable dt (pour le client qui calcule son propre delta).
     *
     * @param computeDt  Fonction retournant le delta time (ex: depuis glfwGetTime).
     * @param onPostLoop Callback appelé après PostSwap chaque frame.
     */
    template <typename DtFn, typename PostLoopFn>
    void runVariableDt(DtFn computeDt, PostLoopFn onPostLoop)
    {
        if (_running)
        {
            std::cerr << "[ERROR] Core already running\n";
            return;
        }
        _running = true;

        while (_running)
        {
            float dt = computeDt();
            if (dt > 0.1f)
                dt = 0.1f; // Clamp to avoid huge jumps

            _scheduler.ordered_tick_pre_swap(_world, dt);
            _world.swapBuffers();
            _scheduler.ordered_tick_post_swap(_world, dt);

            onPostLoop(dt);
        }
    }

    void stop() { _running = false; }

    // ─── Accessors ───────────────────────────────────────────

    [[nodiscard]] WorldPartition &world() noexcept { return _world; }
    [[nodiscard]] const WorldPartition &world() const noexcept { return _world; }

    [[nodiscard]] Network &network() noexcept { return _network; }
    [[nodiscard]] const Network &network() const noexcept { return _network; }

    [[nodiscard]] SystemScheduler &scheduler() noexcept { return _scheduler; }
    [[nodiscard]] const SystemScheduler &scheduler() const noexcept { return _scheduler; }

    [[nodiscard]] InputManager &inputManager() noexcept { return _inputManager; }
    [[nodiscard]] const InputManager &inputManager() const noexcept { return _inputManager; }

    [[nodiscard]] PacketQueue &packetQueue() noexcept { return _packetQueue; }
    [[nodiscard]] const PacketQueue &packetQueue() const noexcept { return _packetQueue; }

    [[nodiscard]] SessionManager &sessionManager() noexcept { return _sessionManager; }
    [[nodiscard]] const SessionManager &sessionManager() const noexcept { return _sessionManager; }

    [[nodiscard]] bool isRunning() const noexcept { return _running; }

    // ─── Network Helpers ─────────────────────────────────────

    void initClientNetwork(const char *serverIp, uint16_t serverPort)
    {
        _network.set_server_info(serverIp, serverPort);
        _network.send_connect(serverIp, serverPort);
        std::cout << "[CLIENT] MSG_CONNECT sent to " << serverIp << ":" << serverPort << "\n";
    }

protected:
    [[nodiscard]] static uint64_t get_time_ns()
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    }

    static void static_sigint_handler(int) { if (_instance) _instance->_running.store(false, std::memory_order_relaxed); }

private:
    static inline Core *_instance = nullptr;
    static constexpr uint64_t FRAME_TIME_NS = 16666666; // ~60Hz
    static constexpr float DELTA_TIME = 1.f / 60.f;

    std::atomic<bool> _running{false};
    WorldPartition _world;
    Network _network;
    SystemScheduler _scheduler;
    InputManager _inputManager;
    PacketQueue _packetQueue;
    SessionManager _sessionManager;
};
