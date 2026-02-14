/**
 * @file SystemScheduler.hpp
 * @brief ECS System Scheduler avec dépendances composants et DAG automatique.
 *
 * Chaque System déclare les composants qu'il lit/écrit via ComponentAccess.
 * Le scheduler construit un DAG d'exécution :
 *   - Deux systèmes qui écrivent le même composant → séquentialisés
 *   - Un Write et un Read sur le même composant → séquentialisés
 *   - Deux Reads sur le même composant → parallélisables
 *
 * Les systèmes sans conflit sont regroupés dans le même "stage" et pourraient
 * à terme tourner en parallèle (CUDA streams, threads).
 *
 * L'exécution se fait en stages ordonnés. Après tous les stages,
 * le scheduler appelle world.swapBuffers() automatiquement.
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

// ─── System Descriptor ───────────────────────────────────────

/**
 * @brief Descripteur complet d'un système ECS.
 *
 * @param name     Nom lisible (debug/logging).
 * @param priority Priorité d'exécution (plus petit = plus tôt) au sein d'un même stage.
 * @param tick     Foncteur appelé chaque frame avec (world, dt).
 * @param access   Liste des composants lus/écrits par ce système.
 */
struct SystemDescriptor {
    std::string name;
    int32_t priority = 0;
    std::function<void(WorldPartition &, float)> tick;
    std::vector<ComponentAccess> access;
};

// ─── System Scheduler ────────────────────────────────────────

/**
 * @brief Scheduler ECS avec analyse de dépendances DAG.
 *
 * Usage :
 * @code
 *   SystemScheduler scheduler;
 *   scheduler.registerSystem({"Physics", 0, physicsFn, {{ComponentId::Position, AccessMode::Write}, ...}});
 *   scheduler.registerSystem({"Network", -10, netFn, {{ComponentId::Velocity, AccessMode::Write}, ...}});
 *   scheduler.buildSchedule();
 *
 *   while (running)
 *       scheduler.tick(world, dt);
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
     * @brief Construit le DAG d'exécution.
     *
     * Algorithme :
     * 1. Trier les systèmes par priorité.
     * 2. Pour chaque système, vérifier les conflits avec ceux déjà placés dans le stage courant.
     * 3. Si conflit → nouveau stage. Sinon → même stage.
     *
     * Un conflit existe si deux systèmes accèdent au même composant
     * et qu'au moins un des accès est Write.
     */
    void buildSchedule()
    {
        if (_systems.empty())
            return;

        // Tri par priorité croissante
        std::sort(_systems.begin(), _systems.end(),
            [](const SystemDescriptor &a, const SystemDescriptor &b) {
                return a.priority < b.priority;
            });

        // Construction des stages par analyse de conflits
        _stages.clear();
        _stages.push_back({0}); // Premier système dans le premier stage

        for (size_t i = 1; i < _systems.size(); ++i)
        {
            bool conflictWithCurrentStage = false;

            // Vérifier les conflits avec tous les systèmes du stage courant
            for (uint32_t sysIdx : _stages.back())
            {
                if (hasConflict(_systems[i], _systems[sysIdx]))
                {
                    conflictWithCurrentStage = true;
                    break;
                }
            }

            if (conflictWithCurrentStage)
                _stages.push_back({static_cast<uint32_t>(i)}); // Nouveau stage
            else
                _stages.back().push_back(static_cast<uint32_t>(i)); // Même stage
        }

        _scheduleDirty = false;

#ifdef LPL_MONITORING
        printf("[SCHEDULER] Built %zu stages for %zu systems:\n", _stages.size(), _systems.size());
        for (size_t s = 0; s < _stages.size(); ++s)
        {
            printf("  Stage %zu: ", s);
            for (uint32_t idx : _stages[s])
                printf("[%s (p=%d)] ", _systems[idx].name.c_str(), _systems[idx].priority);
            printf("\n");
        }
#endif
    }

    /**
     * @brief Exécute tous les systèmes en respectant le DAG.
     *
     * Chaque stage est exécuté séquentiellement.
     * Les systèmes d'un même stage sont parallélisables.
     * Après tous les stages, appelle world.swapBuffers().
     */
    void tick(WorldPartition &world, float dt)
    {
        if (_scheduleDirty)
            buildSchedule();

        for (const auto &stage : _stages)
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
     * @brief Nombre de stages dans le schedule courant.
     */
    [[nodiscard]] size_t getStageCount() const noexcept { return _stages.size(); }

    /**
     * @brief Nombre total de systèmes enregistrés.
     */
    [[nodiscard]] size_t getSystemCount() const noexcept { return _systems.size(); }

    /**
     * @brief Log le schedule courant (debug).
     */
    void printSchedule() const
    {
        printf("[SCHEDULER] %zu systems, %zu stages:\n", _systems.size(), _stages.size());
        for (size_t s = 0; s < _stages.size(); ++s)
        {
            printf("  Stage %zu:", s);
            for (uint32_t idx : _stages[s])
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
    }

private:
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
        default:                    return "?";
        }
    }

private:
    std::vector<SystemDescriptor> _systems;
    std::vector<std::vector<uint32_t>> _stages; ///< Chaque stage = liste d'indices dans _systems
    ThreadPool _pool;
    bool _scheduleDirty = false;
};
