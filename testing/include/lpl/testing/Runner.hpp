/**
 * @file Runner.hpp
 * @brief Runs every engine test and writes KTAP, the same way on the host and in ring 0.
 *
 * The caller writes the KTAP header and the totals: the host runner around the engine's suites
 * alone, the kernel around its own suites and these. Each suite is one subtest, and each test
 * reports its claims that did not hold and its records.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-10-07
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_TESTING_RUNNER_HPP
#    define LPL_TESTING_RUNNER_HPP

#    include <lpl/core/Types.hpp>

#    include <string_view>

namespace lpl::testing {

/**
 * @brief Where a run is written: standard output on a host, the serial port in ring 0.
 */
class Sink {
public:
    /**
     * @brief Writes @p text as it is: the runner supplies every newline.
     */
    virtual void write(std::string_view text) noexcept = 0;

protected:
    ~Sink() = default;
};

/**
 * @struct Totals
 * @brief What a run came to.
 */
struct Totals {
    core::u32 passed = 0u;   /**< Tests whose every claim held. */
    core::u32 failed = 0u;   /**< Tests with a claim that did not hold, or with no claim at all. */
    core::u32 skipped = 0u;  /**< Tests that did not run, with the reason printed. */
    core::u32 checks = 0u;   /**< Claims checked, over every test. */
    core::u32 selected = 0u; /**< Tests a selection named, so that one naming none can say so. */
};

/**
 * @brief Number of suites the run will report, for the KTAP plan.
 */
[[nodiscard]] core::u32 suiteCount() noexcept;

/**
 * @brief Runs every test, one KTAP subtest per suite, numbered from @p firstSuiteNumber.
 *
 * @param sink             Where the run is written.
 * @param selection        Comma-separated `suite.test` patterns, where `*` matches any run, ending at
 *                         a space or at the end of the string. nullptr runs every test.
 * @param firstSuiteNumber KTAP number of the first suite.
 * @return What the run came to.
 */
[[nodiscard]] Totals run(Sink &sink, const char *selection, core::u32 firstSuiteNumber) noexcept;

} // namespace lpl::testing

#endif // LPL_TESTING_RUNNER_HPP
