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
#    define LPL_ECS_SYSTEMSCHEDULER_HPP

#    include <lpl/concurrency/IJobSystem.hpp>
#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/std/functional.hpp>
#    include <lpl/std/memory.hpp>
#    include <lpl/std/vector.hpp>

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
class SystemScheduler final : public core::NonCopyable<SystemScheduler> {
public:
    /**
     * @brief Constructs a scheduler backed by the given job system.
     * @param jobs Job dispatcher used to run each wave (inline or parallel).
     */
    explicit SystemScheduler(concurrency::IJobSystem &jobs);

    ~SystemScheduler();

    /**
     * @name Registration
     * @{
     */
    /**
     * @brief Registers a system instance.
     * @param system Owning pointer to the system.
     * @return OK on success, error if the DAG would contain a cycle.
     */
    [[nodiscard]] core::Expected<void> registerSystem(lpl::pmr::unique_ptr<ISystem> system);

    /**
     * @brief Rebuilds the DAG after all systems have been registered.
     *
     * Each ordered pair of systems is examined exactly once and adds at most one
     * directed edge, so the adjacency lists never contain duplicates — a plain
     * vector suffices (no set needed) and keeps the in-degrees exact.
     *
     * The FIFO frontier of Kahn's algorithm is a vector consumed via a head index
     * (a std::queue would pull in <queue>, which is not in the freestanding
     * subset). Append-only + head advance preserves the exact BFS wave order.
     *
     * @return OK if a valid topological order exists.
     */
    [[nodiscard]] core::Expected<void> buildGraph();
    /** @} */

    /**
     * @name Execution
     * @{
     */
    /**
     * @brief Runs all systems for one tick in dependency order.
     *
     * All systems in a wave share the same phase (guaranteed by the DAG, which
     * creates edges across phases), so the phase of a wave is that of its first
     * system. A phase callback fires when the next wave belongs to another phase,
     * and the last phase's at the end of the tick.
     *
     * A multi-system wave goes to the job dispatcher, which runs every system and
     * only returns once the wave is complete. The inline dispatcher executes them
     * in order (deterministic); a parallel backend could fan them out — the
     * systems in a wave are data-hazard-free.
     *
     * @param dt Fixed delta-time.
     */
    void tick(core::f32 dt);

    /**
     * @brief Registers a callback to be invoked once after the last system
     *        of @p afterPhase finishes and before the first system of the
     *        next phase begins.
     *
     * Typical use: insert `registry.swapAllBuffers()` between Physics and
     * Network phases so that Broadcast reads the freshly computed data.
     *
     * @param afterPhase  The phase after which the callback fires.
     * @param callback    Invocable called with zero arguments.
     */
    void setPhaseCallback(SchedulePhase afterPhase, lpl::pmr::function<void()> callback);

    /**
     * @brief Returns the number of registered systems.
     */
    [[nodiscard]] core::u32 systemCount() const noexcept;
    /** @} */

private:
    struct Impl;
    lpl::pmr::unique_ptr<Impl> _impl;
};

} // namespace lpl::ecs

#endif // LPL_ECS_SYSTEMSCHEDULER_HPP
