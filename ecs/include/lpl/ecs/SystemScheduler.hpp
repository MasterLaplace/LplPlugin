/**
 * @file SystemScheduler.hpp
 * @brief DAG-based system scheduler with automatic parallelism.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_SYSTEMSCHEDULER_HPP
    #define LPL_ECS_SYSTEMSCHEDULER_HPP

#include <lpl/ecs/System.hpp>
#include <lpl/concurrency/ThreadPool.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>
#include <lpl/core/Expected.hpp>

#include <memory>
#include <vector>

namespace lpl::ecs {

/**
 * @class SystemScheduler
 * @brief Builds a directed acyclic graph (DAG) from system descriptors and
 *        runs independent systems in parallel, respecting data dependencies.
 *
 * @par Algorithm
 * 1. For each phase, collect registered systems.
 * 2. Within a phase, build edges: if system A writes component C and system
 *    B reads or writes C, then B depends on A (or A on B if B was
 *    registered first).
 * 3. Run a topological sort to obtain execution waves.
 * 4. Each wave is dispatched to the ThreadPool in parallel; a latch waits
 *    for all systems in the wave to finish before advancing.
 */
class SystemScheduler final : public core::NonCopyable<SystemScheduler>
{
public:
    /**
     * @brief Constructs a scheduler backed by the given thread pool.
     * @param pool Thread pool used for parallel dispatch.
     */
    explicit SystemScheduler(concurrency::ThreadPool& pool);

    ~SystemScheduler();

    // --------------------------------------------------------------------- //
    //  Registration                                                          //
    // --------------------------------------------------------------------- //

    /**
     * @brief Registers a system instance.
     * @param system Owning pointer to the system.
     * @return OK on success, error if the DAG would contain a cycle.
     */
    [[nodiscard]] core::Expected<void> registerSystem(std::unique_ptr<ISystem> system);

    /**
     * @brief Rebuilds the DAG after all systems have been registered.
     * @return OK if a valid topological order exists.
     */
    [[nodiscard]] core::Expected<void> buildGraph();

    // --------------------------------------------------------------------- //
    //  Execution                                                             //
    // --------------------------------------------------------------------- //

    /**
     * @brief Runs all systems for one tick in dependency order.
     * @param dt Fixed delta-time.
     */
    void tick(core::f32 dt);

    /** @brief Returns the number of registered systems. */
    [[nodiscard]] core::u32 systemCount() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::ecs

#endif // LPL_ECS_SYSTEMSCHEDULER_HPP
