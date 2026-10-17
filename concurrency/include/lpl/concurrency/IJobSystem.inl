#ifndef LPL_CONCURRENCY_IJOBSYSTEM_INL
#define LPL_CONCURRENCY_IJOBSYSTEM_INL

namespace lpl::concurrency {

inline void InlineJobSystem::dispatch(std::span<lpl::pmr::function<void()>> jobs)
{
    for (auto &job : jobs)
        job();
}

} // namespace lpl::concurrency

#endif // LPL_CONCURRENCY_IJOBSYSTEM_INL
