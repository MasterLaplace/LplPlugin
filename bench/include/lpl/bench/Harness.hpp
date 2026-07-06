/**
 * @file Harness.hpp
 * @brief Statistical micro-benchmark harness: adaptive repetition, robust
 *        statistics, and compiler dead-code-elimination barriers.
 *
 * A single timed run conflates the measured work with scheduler preemption,
 * cold caches, CPU frequency ramp, and — in an optimised build — the compiler
 * deleting work whose result is never observed. This harness neutralises all
 * four: it warms up before measuring, repeats each timed section until a
 * wall-time budget is met, reduces the samples to median / CV / min / p99, and
 * provides @ref doNotOptimize / @ref clobberMemory barriers that force results
 * to be materialised.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-06
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_BENCH_HARNESS_HPP
#    define LPL_BENCH_HARNESS_HPP

#    include <lpl/core/Platform.hpp>
#    include <lpl/core/Types.hpp>

#    include <chrono>
#    include <string>
#    include <vector>

namespace lpl::bench {

/**
 * @brief Forces @p value to be materialised in a register or memory, defeating
 *        the dead-code and dead-store elimination that would otherwise erase a
 *        benchmark whose result is never read.
 * @tparam T Type of the value to keep alive.
 * @param value Result the optimiser must not discard.
 */
template <typename T> LPL_FORCEINLINE void doNotOptimize(const T &value)
{
#    if defined(LPL_COMPILER_GCC) || defined(LPL_COMPILER_CLANG)
    asm volatile("" : : "r,m"(value) : "memory");
#    else
    volatile const T &sink = value;
    (void) sink;
#    endif
}

/**
 * @brief Compiler-level memory barrier preventing timed stores from being
 *        hoisted out of or sunk past the measured region.
 */
LPL_FORCEINLINE void clobberMemory()
{
#    if defined(LPL_COMPILER_GCC) || defined(LPL_COMPILER_CLANG)
    asm volatile("" : : : "memory");
#    endif
}

/**
 * @brief Reduced statistics for one benchmarked kernel. All durations are in
 *        nanoseconds.
 */
struct Result {
    core::f64 minNs = 0.0;    ///< Fastest observed run (least interference).
    core::f64 medianNs = 0.0; ///< Robust central estimate.
    core::f64 meanNs = 0.0;   ///< Arithmetic mean.
    core::f64 p99Ns = 0.0;    ///< 99th-percentile tail latency.
    core::f64 stddevNs = 0.0; ///< Sample standard deviation.
    core::u32 samples = 0;    ///< Number of timed repetitions collected.
};

/**
 * @brief Stopping policy for @ref run: how long to warm up and how many
 *        repetitions to collect.
 */
struct Config {
    core::u32 warmup = 3;                ///< Untimed priming runs.
    core::u32 minReps = 8;               ///< Floor on samples, even for slow kernels.
    core::u32 maxReps = 4000;            ///< Ceiling on samples for fast kernels.
    core::f64 targetTotalMs = 150.0;     ///< Stop once measured time reaches this.
};

/**
 * @brief Formats a nanosecond duration with an auto-scaled unit and four
 *        significant figures (e.g. "13.59 ms", "824.1 us", "4.213 ns").
 * @param ns Duration in nanoseconds.
 * @return Human-readable duration string.
 */
[[nodiscard]] std::string formatDuration(core::f64 ns);

/**
 * @brief Maps a per-frame time to a real-time / playable / too-slow verdict.
 * @param msPerFrame Frame time in milliseconds.
 * @return Static verdict string (padded for column alignment).
 */
[[nodiscard]] const char *frameRateVerdict(core::f64 msPerFrame);

/**
 * @brief Prints the column legend for the one-line results emitted by @ref run.
 */
void printLegend();

/**
 * @brief Prints a section separator of the form `--- title ---`.
 * @param title Section title.
 */
void section(const char *title);

/**
 * @brief Reduces raw nanosecond samples to a @ref Result, prints a one-line
 *        summary, and returns the statistics.
 * @param label Kernel name shown on the summary line.
 * @param samplesNs Timed samples in nanoseconds (sorted in place).
 * @return Reduced statistics. Behaviour is undefined if @p samplesNs is empty.
 */
[[nodiscard]] Result report(const char *label, std::vector<core::f64> &samplesNs);

/**
 * @brief Times @p fn repeatedly and reports its statistics.
 *
 * @p fn is run @ref Config::warmup times untimed, then repeatedly timed until
 * either @ref Config::maxReps is reached or the accumulated wall time exceeds
 * @ref Config::targetTotalMs (never below @ref Config::minReps). @p fn must be
 * idempotent across calls: benchmarks that mutate accumulating state should
 * build that state inside @p fn so each repetition starts clean.
 *
 * @tparam Fn Callable taking no arguments.
 * @param label Kernel name shown on the summary line.
 * @param fn Work to measure.
 * @param cfg Stopping policy.
 * @return Reduced statistics for the run.
 */
template <typename Fn> Result run(const char *label, Fn &&fn, Config cfg = {})
{
    using clock = std::chrono::steady_clock;

    for (core::u32 i = 0; i < cfg.warmup; ++i)
        fn();

    std::vector<core::f64> samples;
    samples.reserve(cfg.maxReps);
    core::f64 totalNs = 0.0;

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

    return report(label, samples);
}

} // namespace lpl::bench

#endif // LPL_BENCH_HARNESS_HPP
