#ifndef LPL_BENCH_HARNESS_INL
#define LPL_BENCH_HARNESS_INL

namespace lpl::bench {

template <typename T> LPL_FORCEINLINE void doNotOptimize(const T &value)
{
#if defined(LPL_COMPILER_GCC) || defined(LPL_COMPILER_CLANG)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    volatile const T &sink = value;
    (void) sink;
#endif
}

LPL_FORCEINLINE void clobberMemory()
{
#if defined(LPL_COMPILER_GCC) || defined(LPL_COMPILER_CLANG)
    asm volatile("" : : : "memory");
#endif
}

template <typename Fn> Result run(const char *label, Fn &&fn, Config cfg)
{
    using clock = std::chrono::steady_clock;

    for (core::u32 i = 0; i < cfg.warmup; ++i)
        fn();

    std::vector<core::f64> samples;
    samples.reserve(cfg.maxReps);
    core::f64 totalNs = 0.0;

    const EnergyBracket energy{energyMeter()};
    for (core::u32 i = 0; i < cfg.maxReps; ++i)
    {
        const auto t0 = clock::now();
        fn();
        clobberMemory();
        const auto t1 = clock::now();

        const core::f64 ns = std::chrono::duration<core::f64, std::nano>(t1 - t0).count();
        samples.push_back(ns);
        totalNs += ns;

        if (i + 1 >= cfg.minReps && totalNs >= cfg.targetTotalMs * 1e6)
            break;
    }

    return report(label, samples, energy.microjoulesPerRepetition(samples.size()));
}

} // namespace lpl::bench

#endif // LPL_BENCH_HARNESS_INL
