/**
 * @file SystemScheduler.cpp
 * @brief DAG system scheduler implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/ecs/SystemScheduler.hpp>

#include <algorithm>

namespace lpl::ecs {

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct SystemScheduler::Impl {
    concurrency::IJobSystem &jobs;
    lpl::pmr::vector<lpl::pmr::unique_ptr<ISystem>> systems;
    lpl::pmr::vector<lpl::pmr::vector<core::u32>> waves;
    bool graphBuilt{false};

    // Phase boundary callbacks: indexed by SchedulePhase (fired after that phase)
    lpl::pmr::function<void()> phaseCallbacks[static_cast<core::u8>(SchedulePhase::Count)] = {};

    explicit Impl(concurrency::IJobSystem &j) : jobs{j} {}
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

SystemScheduler::SystemScheduler(concurrency::IJobSystem &jobs) : _impl{lpl::pmr::make_unique<Impl>(jobs)} {}

SystemScheduler::~SystemScheduler() = default;

core::Expected<void> SystemScheduler::registerSystem(lpl::pmr::unique_ptr<ISystem> system)
{
    if (!system)
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Null system");
    }
    _impl->systems.push_back(std::move(system));
    _impl->graphBuilt = false;
    return {};
}

core::Expected<void> SystemScheduler::buildGraph()
{
    const core::u32 n = static_cast<core::u32>(_impl->systems.size());

    // Each ordered pair (i, j) is examined exactly once below and adds at most
    // one directed edge, so the adjacency lists never contain duplicates — a
    // plain vector suffices (no set needed) and keeps inDegree exact.
    lpl::pmr::vector<lpl::pmr::vector<core::u32>> adj(n);
    lpl::pmr::vector<core::u32> inDegree(n, 0);

    for (core::u32 i = 0; i < n; ++i)
    {
        const auto &descA = _impl->systems[i]->descriptor();

        for (core::u32 j = i + 1; j < n; ++j)
        {
            const auto &descB = _impl->systems[j]->descriptor();

            if (descA.phase != descB.phase)
            {
                if (descA.phase < descB.phase)
                {
                    adj[i].push_back(j);
                    ++inDegree[j];
                }
                else
                {
                    adj[j].push_back(i);
                    ++inDegree[i];
                }
                continue;
            }

            bool conflict = false;
            for (const auto &a : descA.accesses)
            {
                for (const auto &b : descB.accesses)
                {
                    if (a.id == b.id && (a.mode == AccessMode::ReadWrite || b.mode == AccessMode::ReadWrite))
                    {
                        conflict = true;
                        break;
                    }
                }
                if (conflict)
                    break;
            }

            if (conflict)
            {
                adj[i].push_back(j);
                ++inDegree[j];
            }
        }
    }

    // FIFO frontier for Kahn's algorithm, as a vector consumed via a head index
    // (a std::queue would pull in <queue>, which is not in the freestanding
    // subset). Append-only + head advance preserves the exact BFS wave order.
    lpl::pmr::vector<core::u32> ready;
    for (core::u32 i = 0; i < n; ++i)
    {
        if (inDegree[i] == 0)
        {
            ready.push_back(i);
        }
    }

    _impl->waves.clear();
    core::u32 processed = 0;
    lpl::core::usize readyHead = 0;

    while (readyHead < ready.size())
    {
        lpl::pmr::vector<core::u32> wave;
        const auto waveSize = static_cast<core::u32>(ready.size() - readyHead);

        for (core::u32 w = 0; w < waveSize; ++w)
        {
            wave.push_back(ready[readyHead]);
            ++readyHead;
        }

        for (auto idx : wave)
        {
            for (auto dep : adj[idx])
            {
                if (--inDegree[dep] == 0)
                {
                    ready.push_back(dep);
                }
            }
        }

        _impl->waves.push_back(std::move(wave));
        processed += waveSize;
    }

    if (processed != n)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Cycle detected in system DAG");
    }

    _impl->graphBuilt = true;
    return {};
}

void SystemScheduler::tick(core::f32 dt)
{
    LPL_ASSERT(_impl->graphBuilt);

    // Track which phase was last executed to detect phase transitions
    auto lastPhase = SchedulePhase::Count;

    for (const auto &wave : _impl->waves)
    {
        // Determine the phase of the first system in this wave.
        // All systems in a wave share the same phase (guaranteed by the DAG
        // which creates edges across phases), or they are in the same phase
        // with no data conflicts.
        const auto wavePhase = _impl->systems[wave[0]]->descriptor().phase;

        // Fire phase callback on transition
        if (lastPhase != SchedulePhase::Count && wavePhase != lastPhase)
        {
            const auto idx = static_cast<core::u8>(lastPhase);
            if (_impl->phaseCallbacks[idx])
            {
                _impl->phaseCallbacks[idx]();
            }
        }
        lastPhase = wavePhase;

        if (wave.size() == 1)
        {
            _impl->systems[wave[0]]->execute(dt);
            continue;
        }

        // Multi-system wave: hand every system to the job dispatcher, which
        // runs them all and only returns once the wave is complete. The inline
        // dispatcher executes them in order (deterministic); a parallel backend
        // could fan them out — the systems in a wave are data-hazard-free.
        lpl::pmr::vector<lpl::pmr::function<void()>> jobs;
        for (auto idx : wave)
            jobs.push_back([this, idx, dt]() { _impl->systems[idx]->execute(dt); });

        _impl->jobs.dispatch(std::span<lpl::pmr::function<void()>>{jobs.data(), jobs.size()});
    }

    // Fire callback for the last phase if any
    if (lastPhase != SchedulePhase::Count)
    {
        const auto idx = static_cast<core::u8>(lastPhase);
        if (_impl->phaseCallbacks[idx])
        {
            _impl->phaseCallbacks[idx]();
        }
    }
}

void SystemScheduler::setPhaseCallback(SchedulePhase afterPhase, lpl::pmr::function<void()> callback)
{
    const auto idx = static_cast<core::u8>(afterPhase);
    _impl->phaseCallbacks[idx] = std::move(callback);
}

core::u32 SystemScheduler::systemCount() const noexcept { return static_cast<core::u32>(_impl->systems.size()); }

} // namespace lpl::ecs
