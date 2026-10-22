/**
 * @file Runner.hpp
 * @brief Runs every declared test and writes KTAP, the same way on the host and in ring 0.
 *
 * The caller writes the KTAP header and the totals: `testing/host/main.cpp` on a host, the kernel
 * around its own suites and these. Each suite is one subtest, and each test
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
    core::u32 required = 0u; /**< One bit per pattern of the required list, set when it named a test of the run. */
};

/** @brief Patterns a required list may hold: one bit each in Totals::required. */
inline constexpr core::u32 kMostRequiredPatterns = 32u;

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

/**
 * @brief Runs every test, as the other overload does, and fails every test @p required names that
 *        did not run, for whatever reason: not selected, or skipped by its body.
 *
 * A test that is not run reports a skip, which reads as a pass: a test a boot has to run, because
 * its hardware is there or its feature is the point of the image, is required so that skipping it
 * fails instead.
 *
 * @param sink             Where the run is written.
 * @param selection        As in the other overload.
 * @param required         Comma-separated `suite.test` patterns in the same form, at most
 *                         kMostRequiredPatterns. nullptr requires nothing.
 * @param firstSuiteNumber KTAP number of the first suite.
 * @return What the run came to, with in Totals::required the patterns that named a test of it: a
 *         caller that runs several sets of tests joins those bits before calling reportUnnamed.
 */
[[nodiscard]] Totals run(Sink &sink, const char *selection, const char *required, core::u32 firstSuiteNumber) noexcept;

/**
 * @brief Writes a diagnostic line for every pattern of @p required that named no test.
 *
 * A required pattern that names nothing would require nothing: a typo, or a test renamed since.
 *
 * @param sink     Where the lines are written.
 * @param required The required list a run was given.
 * @param named    Totals::required, joined over every run that was given the list.
 * @return How many patterns named no test.
 */
[[nodiscard]] core::u32 reportUnnamed(Sink &sink, const char *required, core::u32 named) noexcept;

} // namespace lpl::testing

#endif // LPL_TESTING_RUNNER_HPP
