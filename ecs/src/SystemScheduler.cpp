#include <lpl/core/Assert.hpp>
#include <lpl/ecs/SystemScheduler.hpp>

#include <algorithm>

namespace lpl::ecs {

struct SystemScheduler::Impl {
    concurrency::IJobSystem &jobs;
    lpl::pmr::vector<lpl::pmr::unique_ptr<ISystem>> systems;
    lpl::pmr::vector<lpl::pmr::vector<core::u32>> waves;
    bool graphBuilt{false};

    lpl::pmr::function<void()> phaseCallbacks[static_cast<core::u8>(SchedulePhase::Count)] = {};

    explicit Impl(concurrency::IJobSystem &j) : jobs{j} {}
};

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

            for (const auto &a : descA.resources)
            {
                if (conflict)
                    break;
                for (const auto &b : descB.resources)
                {
                    if (a.id == b.id && (a.mode == AccessMode::ReadWrite || b.mode == AccessMode::ReadWrite))
                    {
                        conflict = true;
                        break;
                    }
                }
            }

            if (conflict)
            {
                adj[i].push_back(j);
                ++inDegree[j];
            }
        }
    }

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

    auto lastPhase = SchedulePhase::Count;

    for (const auto &wave : _impl->waves)
    {
        const auto wavePhase = _impl->systems[wave[0]]->descriptor().phase;

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

        lpl::pmr::vector<lpl::pmr::function<void()>> jobs;
        for (auto idx : wave)
            jobs.push_back([this, idx, dt]() { _impl->systems[idx]->execute(dt); });

        _impl->jobs.dispatch(std::span<lpl::pmr::function<void()>>{jobs.data(), jobs.size()});
    }

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
