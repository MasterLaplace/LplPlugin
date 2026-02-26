// /////////////////////////////////////////////////////////////////////////////
/// @file SystemScheduler.cpp
/// @brief DAG system scheduler implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/ecs/SystemScheduler.hpp>
#include <lpl/core/Assert.hpp>

#include <algorithm>
#include <latch>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace lpl::ecs {

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct SystemScheduler::Impl
{
    concurrency::ThreadPool&                        pool;
    std::vector<std::unique_ptr<ISystem>>           systems;
    std::vector<std::vector<core::u32>>             waves;
    bool                                            graphBuilt{false};

    explicit Impl(concurrency::ThreadPool& p) : pool{p} {}
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

SystemScheduler::SystemScheduler(concurrency::ThreadPool& pool)
    : impl_{std::make_unique<Impl>(pool)}
{}

SystemScheduler::~SystemScheduler() = default;

core::Expected<void> SystemScheduler::registerSystem(std::unique_ptr<ISystem> system)
{
    if (!system)
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Null system");
    }
    impl_->systems.push_back(std::move(system));
    impl_->graphBuilt = false;
    return {};
}

core::Expected<void> SystemScheduler::buildGraph()
{
    const core::u32 n = static_cast<core::u32>(impl_->systems.size());

    std::vector<std::unordered_set<core::u32>> adj(n);
    std::vector<core::u32> inDegree(n, 0);

    for (core::u32 i = 0; i < n; ++i)
    {
        const auto& descA = impl_->systems[i]->descriptor();

        for (core::u32 j = i + 1; j < n; ++j)
        {
            const auto& descB = impl_->systems[j]->descriptor();

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
            for (const auto& a : descA.accesses)
            {
                for (const auto& b : descB.accesses)
                {
                    if (a.id == b.id &&
                        (a.mode == AccessMode::ReadWrite || b.mode == AccessMode::ReadWrite))
                    {
                        conflict = true;
                        break;
                    }
                }
                if (conflict) break;
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

    impl_->waves.clear();
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

        impl_->waves.push_back(std::move(wave));
        processed += waveSize;
    }

    if (processed != n)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Cycle detected in system DAG");
    }

    impl_->graphBuilt = true;
    return {};
}

void SystemScheduler::tick(core::f32 dt)
{
    LPL_ASSERT(impl_->graphBuilt);

    for (const auto& wave : impl_->waves)
    {
        if (wave.size() == 1)
        {
            impl_->systems[wave[0]]->execute(dt);
            continue;
        }

        std::latch barrier{static_cast<std::ptrdiff_t>(wave.size())};

        for (auto idx : wave)
        {
            impl_->pool.enqueueDetached([&, idx]() {
                impl_->systems[idx]->execute(dt);
                barrier.count_down();
            });
        }

        barrier.wait();
    }
}

core::u32 SystemScheduler::systemCount() const noexcept
{
    return static_cast<core::u32>(impl_->systems.size());
}

} // namespace lpl::ecs
