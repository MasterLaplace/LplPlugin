/**
 * @file RingBench.hpp
 * @brief The ring section of lpl-benchmark: the SPSC ring against the ring it replaced, across two
 *        physical cores.
 *
 * @warning **Two threads on one core measure the scheduler, not the ring.** The producer and the
 * consumer are pinned to two cores that are not SMT siblings, so every index and every slot that
 * crosses from one to the other pays the coherence traffic the ring is built to avoid. On a
 * machine that gives the process fewer than two such cores, the section says so and measures
 * nothing.
 *
 * @warning **On a shared or virtual machine the noise is larger than most differences.** The two
 * rings are timed in pairs, one repetition each, the first of the pair alternating, so both meet the
 * same placement of the threads and the same load, and the speed-up is the median of the pairs'
 * ratios. A last row times the ring against itself: its spread is the noise floor, and a speed-up
 * inside it says nothing.
 *
 * @warning **A fast transfer that lost an element is not fast.** Each repetition checks that the
 * consumer received every sequence number exactly once, and the section stops on the first that
 * did not.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-10-08
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BENCH_RINGBENCH_HPP
#    define LPL_BENCH_RINGBENCH_HPP

namespace lpl::bench {

/**
 * @brief Runs every ring benchmark, the baseline and the current ring interleaved, and prints a
 *        summary of the time per element and the speed-up for each workload.
 */
void runRingBenchmarks();

} // namespace lpl::bench

#endif // LPL_BENCH_RINGBENCH_HPP
