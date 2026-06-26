/**
 * @file IJobSystem.hpp
 * @brief Job-dispatch seam for the ECS scheduler.
 *
 * The SystemScheduler depends on this interface rather than on a concrete
 * ThreadPool, so the same DAG-driven tick runs on every target. The kernel
 * build (and the determinism oracle, which is forced single-threaded) use the
 * header-only InlineJobSystem; a future parallel backend can implement the same
 * interface over a real scheduler without touching the ECS.
 *
 * Determinism: callers MUST NOT assume jobs run concurrently. dispatch() runs
 * every job and only returns once all have completed.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CONCURRENCY_IJOBSYSTEM_HPP
#    define LPL_CONCURRENCY_IJOBSYSTEM_HPP

#    include <lpl/std/functional.hpp>

#    include <span>

namespace lpl::concurrency {

/**
 * @class IJobSystem
 * @brief Abstract one-shot job dispatcher: run every job, block until done.
 */
class IJobSystem {
public:
    virtual ~IJobSystem() = default;

    /**
     * @brief Runs all @p jobs and returns only once every job has completed.
     * @param jobs Contiguous range of nullary callables.
     */
    virtual void dispatch(std::span<lpl::pmr::function<void()>> jobs) = 0;
};

/**
 * @class InlineJobSystem
 * @brief Single-threaded executor: runs jobs in order on the calling thread.
 *
 * Header-only and freestanding-clean (no threads, no synchronisation), so it
 * is the deterministic default for both the kernel target and the oracle.
 */
class InlineJobSystem final : public IJobSystem {
public:
    void dispatch(std::span<lpl::pmr::function<void()>> jobs) override
    {
        for (auto &job : jobs)
            job();
    }
};

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_IJOBSYSTEM_HPP
