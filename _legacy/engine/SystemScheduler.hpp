/**
 * @file SystemScheduler.hpp
 * @brief ECS System Scheduler avec dépendances composants, DAG automatique et phases PreSwap/PostSwap.
 *
 * Chaque System déclare les composants qu'il lit/écrit via ComponentAccess,
 * ainsi que sa phase d'exécution (PreSwap ou PostSwap).
 *
 * Le scheduler construit un DAG d'exécution par phase :
 *   - Deux systèmes qui écrivent le même composant → séquentialisés
 *   - Un Write et un Read sur le même composant → séquentialisés
 *   - Deux Reads sur le même composant → parallélisables
 *
 * Les systèmes sans conflit sont regroupés dans le même "stage" et pourraient
 * à terme tourner en parallèle (CUDA streams, threads).
 *
 * Flux d'exécution :
 *   1. tick_pre_swap(world, dt)  — systèmes PreSwap (inputs, physique, etc.)
 *   2. world.swapBuffers()       — (appelé par l'utilisateur/Core)
 *   3. tick_post_swap(world, dt) — systèmes PostSwap (broadcast, rendu, etc.)
 *
 * @author MasterLaplace
 */

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <latch>
#include "ThreadPool.hpp"

class WorldPartition; // Forward declaration

// ─── Component identifiers ───────────────────────────────────

/**
 * @brief Identifiants de composants pour le système de dépendances.
 * Chaque valeur correspond à un vecteur SoA dans Partition.
 */
enum class ComponentId : uint8_t {
    Position = 0,
    Velocity,
    Forces,
    Mass,
    Rotation,
    Size,
    Health,
    NeuralControl,
    COUNT
};

/**
 * @brief Mode d'accès à un composant.
 */
enum class AccessMode : uint8_t {
    Read,
    Write
};

/**
 * @brief Déclaration d'accès d'un système à un composant.
 */
struct ComponentAccess {
    ComponentId component;
    AccessMode  mode;
};

// ─── Schedule Phase ──────────────────────────────────────────

/**
 * @brief Phase d'exécution d'un système dans le scheduler.
 *
 * PreSwap  — exécuté avant world.swapBuffers() (écrit dans le write buffer).
 * PostSwap — exécuté après  world.swapBuffers() (lit depuis le read buffer, ex: broadcast, rendu).
 */
enum class SchedulePhase : uint8_t {
    PreSwap = 0,
    PostSwap
};

// ─── System Descriptor ───────────────────────────────────────

/**
 * @brief Descripteur complet d'un système ECS.
 *
 * @param name     Nom lisible (debug/logging).
 * @param priority Priorité d'exécution (plus petit = plus tôt) au sein d'un même stage.
 * @param tick     Foncteur appelé chaque frame avec (world, dt).
 * @param access   Liste des composants lus/écrits par ce système.
 * @param phase    Phase d'exécution : PreSwap (défaut) ou PostSwap.
 */
struct SystemDescriptor {
    std::string name;
    int32_t priority = 0;
    std::function<void(WorldPartition &, float)> tick;
    std::vector<ComponentAccess> access;
    SchedulePhase phase = SchedulePhase::PreSwap;
};

// ─── System Scheduler ────────────────────────────────────────

/**
 * @brief Scheduler ECS avec analyse de dépendances DAG et phases PreSwap/PostSwap.
 *
 * Usage :
 * @code
 *   SystemScheduler scheduler;
 *   scheduler.registerSystem({"Physics", 0, physicsFn,
 *       {{ComponentId::Position, AccessMode::Write}, ...}, SchedulePhase::PreSwap});
 *   scheduler.registerSystem({"Broadcast", 0, broadcastFn,
 *       {{ComponentId::Position, AccessMode::Read}, ...}, SchedulePhase::PostSwap});
 *   scheduler.buildSchedule();
 *
 *   while (running) {
 *       scheduler.ordered_tick_pre_swap(world, dt);
 *       world.swapBuffers();
 *       scheduler.ordered_tick_post_swap(world, dt);
 *   }
 * @endcode
 */
class SystemScheduler {
public:
    /**
     * @brief Enregistre un système.
     * Invalide le schedule courant — appeler buildSchedule() après.
     */
    void registerSystem(SystemDescriptor desc)
    {
        _systems.push_back(std::move(desc));
        _scheduleDirty = true;
    }

    /**
     * @brief Construit le DAG d'exécution pour chaque phase.
     *
     * Algorithme (par phase) :
     * 1. Filtrer les systèmes de la phase.
     * 2. Trier par priorité croissante.
     * 3. Pour chaque système, vérifier les conflits avec ceux déjà placés dans le stage courant.
     * 4. Si conflit → nouveau stage. Sinon → même stage.
     *
     * Un conflit existe si deux systèmes accèdent au même composant
     * et qu'au moins un des accès est Write.
     */
    void buildSchedule()
    {
        if (_systems.empty())
            return;

        // Tri global par priorité croissante (stable pour les indices)
        // On trie les indices, pas les systèmes eux-mêmes, pour garder les références stables
        std::vector<uint32_t> sortedIndices(_systems.size());
        for (size_t i = 0; i < _systems.size(); ++i)
            sortedIndices[i] = static_cast<uint32_t>(i);

        std::sort(sortedIndices.begin(), sortedIndices.end(),
            [this](uint32_t a, uint32_t b) {
                return _systems[a].priority < _systems[b].priority;
            });

        // Construire les stages pour chaque phase séparément
        _preSwapStages.clear();
        _postSwapStages.clear();

        buildStagesForPhase(sortedIndices, SchedulePhase::PreSwap, _preSwapStages);
        buildStagesForPhase(sortedIndices, SchedulePhase::PostSwap, _postSwapStages);

        _scheduleDirty = false;

#ifdef LPL_MONITORING
        printSchedule();
#endif
    }

    // ─── Ordered (sequential) execution ──────────────────────

    /**
     * @brief Exécute les systèmes PreSwap en respectant le DAG (séquentiel).
     */
    void ordered_tick_pre_swap(WorldPartition &world, float dt)
    {
        if (_scheduleDirty)
            buildSchedule();

        executeStagesOrdered(_preSwapStages, world, dt);
    }

    /**
     * @brief Exécute les systèmes PostSwap en respectant le DAG (séquentiel).
     */
    void ordered_tick_post_swap(WorldPartition &world, float dt)
    {
        if (_scheduleDirty)
            buildSchedule();

        executeStagesOrdered(_postSwapStages, world, dt);
    }

    /**
     * @brief Exécute tous les systèmes PreSwap puis PostSwap (séquentiel).
     * @deprecated Préférer ordered_tick_pre_swap + swapBuffers + ordered_tick_post_swap.
     */
    void ordered_tick(WorldPartition &world, float dt)
    {
        ordered_tick_pre_swap(world, dt);
        // Note: le swapBuffers doit être appelé entre les deux par l'utilisateur
        ordered_tick_post_swap(world, dt);
    }

    // ─── Threaded (parallel) execution ───────────────────────

    /**
     * @brief Exécute les systèmes PreSwap en respectant le DAG (parallèle intra-stage).
     */
    void threaded_tick_pre_swap(WorldPartition &world, float dt)
    {
        if (_scheduleDirty)
            buildSchedule();

        executeStagesThreaded(_preSwapStages, world, dt);
    }

    /**
     * @brief Exécute les systèmes PostSwap en respectant le DAG (parallèle intra-stage).
     */
    void threaded_tick_post_swap(WorldPartition &world, float dt)
    {
        if (_scheduleDirty)
            buildSchedule();

        executeStagesThreaded(_postSwapStages, world, dt);
    }

    /**
     * @brief Exécute tous les systèmes PreSwap puis PostSwap (parallèle).
     * @deprecated Préférer threaded_tick_pre_swap + swapBuffers + threaded_tick_post_swap.
     */
    void threaded_tick(WorldPartition &world, float dt)
    {
        threaded_tick_pre_swap(world, dt);
        threaded_tick_post_swap(world, dt);
    }

    // ─── Queries ─────────────────────────────────────────────

    /** @brief Nombre de stages PreSwap. */
    [[nodiscard]] size_t getPreSwapStageCount() const noexcept { return _preSwapStages.size(); }

    /** @brief Nombre de stages PostSwap. */
    [[nodiscard]] size_t getPostSwapStageCount() const noexcept { return _postSwapStages.size(); }

    /** @brief Nombre total de stages (les deux phases combinées). */
    [[nodiscard]] size_t getStageCount() const noexcept { return _preSwapStages.size() + _postSwapStages.size(); }

    /** @brief Nombre total de systèmes enregistrés. */
    [[nodiscard]] size_t getSystemCount() const noexcept { return _systems.size(); }

    /**
     * @brief Log le schedule courant (debug).
     */
    void printSchedule() const
    {
        printf("[SCHEDULER] %zu systems (%zu PreSwap stages, %zu PostSwap stages):\n",
               _systems.size(), _preSwapStages.size(), _postSwapStages.size());

        auto printPhase = [this](const char *phaseName, const std::vector<std::vector<uint32_t>> &stages) {
            if (stages.empty()) return;
            printf("  ── %s ──\n", phaseName);
            for (size_t s = 0; s < stages.size(); ++s)
            {
                printf("    Stage %zu:", s);
                for (uint32_t idx : stages[s])
                {
                    printf(" [%s p=%d (", _systems[idx].name.c_str(), _systems[idx].priority);
                    for (size_t a = 0; a < _systems[idx].access.size(); ++a)
                    {
                        const auto &acc = _systems[idx].access[a];
                        printf("%s%s:%s",
                            a > 0 ? " " : "",
                            componentName(acc.component),
                            acc.mode == AccessMode::Read ? "R" : "W");
                    }
                    printf(")]");
                }
                printf("\n");
            }
        };

        printPhase("PreSwap", _preSwapStages);
        printPhase("PostSwap", _postSwapStages);
    }

private:
    /**
     * @brief Construit les stages pour une phase donnée.
     */
    void buildStagesForPhase(const std::vector<uint32_t> &sortedIndices,
                             SchedulePhase phase,
                             std::vector<std::vector<uint32_t>> &outStages)
    {
        outStages.clear();

        for (uint32_t idx : sortedIndices)
        {
            if (_systems[idx].phase != phase)
                continue;

            if (outStages.empty())
            {
                outStages.push_back({idx});
                continue;
            }

            bool conflictWithCurrentStage = false;
            for (uint32_t sysIdx : outStages.back())
            {
                if (hasConflict(_systems[idx], _systems[sysIdx]))
                {
                    conflictWithCurrentStage = true;
                    break;
                }
            }

            if (conflictWithCurrentStage)
                outStages.push_back({idx}); // Nouveau stage
            else
                outStages.back().push_back(idx); // Même stage
        }
    }

    /**
     * @brief Exécute des stages séquentiellement.
     */
    void executeStagesOrdered(const std::vector<std::vector<uint32_t>> &stages,
                              WorldPartition &world, float dt)
    {
        for (const auto &stage : stages)
        {
            for (uint32_t sysIdx : stage)
                _systems[sysIdx].tick(world, dt);
        }
    }

    /**
     * @brief Exécute des stages avec parallélisme intra-stage.
     */
    void executeStagesThreaded(const std::vector<std::vector<uint32_t>> &stages,
                               WorldPartition &world, float dt)
    {
        for (const auto &stage : stages)
        {
            if (stage.size() > 1u)
            {
                std::latch sync(static_cast<ptrdiff_t>(stage.size()));
                for (uint32_t sysIdx : stage)
                {
                    _pool.enqueueDetached([&world, dt, sysIdx, this, &sync]() {
                        _systems[sysIdx].tick(world, dt);
                        sync.count_down();
                    });
                }
                sync.wait();
            }
            else if (!stage.empty())
            {
                _systems[stage[0]].tick(world, dt);
            }
        }
    }

    /**
     * @brief Détecte un conflit entre deux systèmes.
     *
     * Un conflit existe si les deux systèmes accèdent au même composant
     * et qu'au moins un des accès est Write.
     *
     * Read + Read  → pas de conflit (parallélisable)
     * Read + Write → conflit (séquentialiser)
     * Write + Write → conflit (séquentialiser)
     */
    [[nodiscard]] static bool hasConflict(const SystemDescriptor &a, const SystemDescriptor &b) noexcept
    {
        for (const auto &accA : a.access)
        {
            for (const auto &accB : b.access)
            {
                if (accA.component == accB.component && (accA.mode == AccessMode::Write || accB.mode == AccessMode::Write))
                    return true;
            }
        }
        return false;
    }

    /**
     * @brief Nom lisible d'un ComponentId (pour le debug).
     */
    [[nodiscard]] static const char *componentName(ComponentId id) noexcept
    {
        switch (id)
        {
        case ComponentId::Position: return "Pos";
        case ComponentId::Velocity: return "Vel";
        case ComponentId::Forces:   return "Forces";
        case ComponentId::Mass:     return "Mass";
        case ComponentId::Rotation: return "Rot";
        case ComponentId::Size:     return "Size";
        case ComponentId::Health:   return "HP";
        case ComponentId::NeuralControl: return "Neural";
        default:                    return "?";
        }
    }

private:
    std::vector<SystemDescriptor> _systems;
    std::vector<std::vector<uint32_t>> _preSwapStages;  ///< Stages exécutés avant swapBuffers
    std::vector<std::vector<uint32_t>> _postSwapStages; ///< Stages exécutés après swapBuffers
    ThreadPool _pool;
    bool _scheduleDirty = false;
};
