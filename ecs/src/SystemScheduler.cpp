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
#include <functional>
#include <latch>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace lpl::ecs {

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct SystemScheduler::Impl {
    concurrency::ThreadPool &pool;
    std::vector<std::unique_ptr<ISystem>> systems;
    std::vector<std::vector<core::u32>> waves;
    bool graphBuilt{false};

    // Phase boundary callbacks: indexed by SchedulePhase (fired after that phase)
    std::function<void()> phaseCallbacks[static_cast<core::u8>(SchedulePhase::Count)] = {};

    explicit Impl(concurrency::ThreadPool &p) : pool{p} {}
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

SystemScheduler::SystemScheduler(concurrency::ThreadPool &pool) : _impl{std::make_unique<Impl>(pool)} {}

SystemScheduler::~SystemScheduler() = default;

core::Expected<void> SystemScheduler::registerSystem(std::unique_ptr<ISystem> system)
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

    std::vector<std::unordered_set<core::u32>> adj(n);
    std::vector<core::u32> inDegree(n, 0);

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
                    adj[i].insert(j);
                    ++inDegree[j];
                }
                else
                {
                    adj[j].insert(i);
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
                adj[i].insert(j);
                ++inDegree[j];
            }
        }
    }

    std::queue<core::u32> ready;
    for (core::u32 i = 0; i < n; ++i)
    {
        if (inDegree[i] == 0)
        {
            ready.push(i);
        }
    }

    _impl->waves.clear();
    core::u32 processed = 0;

    while (!ready.empty())
    {
        std::vector<core::u32> wave;
        const auto waveSize = static_cast<core::u32>(ready.size());

        for (core::u32 w = 0; w < waveSize; ++w)
        {
            wave.push_back(ready.front());
            ready.pop();
        }

        for (auto idx : wave)
        {
            for (auto dep : adj[idx])
            {
                if (--inDegree[dep] == 0)
                {
                    ready.push(dep);
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

        std::latch barrier{static_cast<std::ptrdiff_t>(wave.size())};

        for (auto idx : wave)
        {
            _impl->pool.enqueueDetached([&, idx]() {
                _impl->systems[idx]->execute(dt);
                barrier.count_down();
            });
        }

        barrier.wait();
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

void SystemScheduler::setPhaseCallback(SchedulePhase afterPhase, std::function<void()> callback)
{
    const auto idx = static_cast<core::u8>(afterPhase);
    _impl->phaseCallbacks[idx] = std::move(callback);
}

core::u32 SystemScheduler::systemCount() const noexcept { return static_cast<core::u32>(_impl->systems.size()); }

} // namespace lpl::ecs
