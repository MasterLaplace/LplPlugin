/**
 * @file Test.hpp
 * @brief One declaration per engine test, run on the host and in ring 0, and reported in KTAP.
 *
 * A test file declares its suite once, then each test once, next to nothing else:
 *
 * @code
 * LPL_TEST_SUITE(relief);
 *
 * LPL_TEST(world_stands_on_measured_ground)
 * {
 *     const Walk walk = walkTheSurvey();
 *     test.check(walk.steps >= 8u, "the body walked more than a handful of steps");
 *     test.measureHexadecimal("walk_signature", walk.signature);
 * }
 * @endcode
 *
 * - Registration: the declaration puts a pointer in the `lpl_tests` section; no list names a test.
 *   Every `.cpp` of `tests/<module>/`, for a module the kernel compiles against, is built into
 *   `test-engine` and into debug kernels; `tests/<module>/kernel/` holds the tests of what only the
 *   kernel compiles.
 * - Order: file, then line, the same on both targets.
 * - Records: every `key=value` a test of both targets measures is compared between the host and
 *   ring 0, so such a test never measures a value that differs between them (a size, an address, a
 *   time).
 * - A test that checks nothing fails.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-10-07
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_TESTING_TEST_HPP
#    define LPL_TESTING_TEST_HPP

#    include <lpl/core/Types.hpp>

#    include <string_view>

namespace lpl::testing {

class Sink;
class Test;

/**
 * @struct Suite
 * @brief A group of tests: one per file, named once with LPL_TEST_SUITE().
 */
struct Suite {
    const char *name; /**< Name of the suite, as KTAP reports it. */
};

/**
 * @struct Case
 * @brief One test, as LPL_TEST() declares it.
 */
struct Case {
    const Suite *suite;       /**< Suite of the file the test is declared in. */
    const char *name;         /**< Name of the test within its suite. */
    const char *file;         /**< File the test is declared in, which orders the suites. */
    core::u32 line;           /**< Line the test is declared on, which orders the tests of a suite. */
    void (*body)(Test &test); /**< The test itself. */
};

/**
 * @brief The run of one test: what its body receives, and what its claims and measures report to.
 */
class Test final {
public:
    /**
     * @brief Starts the run of @p testCase, reported to @p sink.
     */
    Test(const Case &testCase, Sink &sink) noexcept;

    /**
     * @brief Checks one claim.
     *
     * @details A false claim fails the test and is printed as a KTAP diagnostic. The test goes on
     *          running, so every claim that does not hold is reported, not only the first.
     *
     * @param condition Whether the claim holds.
     * @param claim     What the condition proves, written as the sentence it is.
     * @return @p condition, so a test can stop where a later claim would make no sense.
     */
    bool check(bool condition, const char *claim) noexcept;

    /**
     * @brief Reports that the test cannot run here, and why. The caller returns right after.
     *
     * @details Checks made before the skip still count: one that failed fails the test.
     */
    void skip(const char *reason) noexcept;

    /**
     * @brief Prints a value the test measured, in decimal, as a record.
     *
     * @details The records of a test both targets run are compared between them.
     *
     * @param key   Name of the value, without a space or an equals sign.
     * @param value The value.
     */
    void measure(const char *key, core::u32 value) noexcept;

    /**
     * @brief Prints a signed value the test measured, in decimal, as a record.
     */
    void measure(const char *key, core::i32 value) noexcept;

    /**
     * @brief Prints a value the test measured, in hexadecimal, as a record.
     */
    void measureHexadecimal(const char *key, core::u32 value) noexcept;

    /**
     * @brief Prints a remark that judges nothing and is compared with nothing.
     *
     * @param text The remark, which must not read as `key=value`.
     */
    void note(const char *text) noexcept;

    /** @brief Claims checked so far. */
    [[nodiscard]] core::u32 checks() const noexcept { return _checks; }

    /** @brief Claims that did not hold. */
    [[nodiscard]] core::u32 failures() const noexcept { return _failures; }

    /** @brief Why the test did not run, or nullptr. */
    [[nodiscard]] const char *skipReason() const noexcept { return _skipReason; }

private:
    void writePrefix() noexcept;
    void writeRecord(const char *key, std::string_view value) noexcept;

    const Case &_case;
    Sink &_sink;
    core::u32 _checks = 0u;
    core::u32 _failures = 0u;
    const char *_skipReason = nullptr;
};

} // namespace lpl::testing

/**
 * @brief Names the suite of the current file.
 *
 * @param suiteName Identifier of the suite.
 */
#    define LPL_TEST_SUITE(suiteName)                                                                                  \
        static constexpr ::lpl::testing::Suite lplTestSuite { #suiteName }

/**
 * @brief Declares a test of the current file's suite and registers it with the runner.
 *
 * @param testName Identifier of the test, unique within its file.
 */
#    define LPL_TEST(testName)                                                                                         \
        static void lplTestBody_##testName(::lpl::testing::Test &test);                                                \
        static constexpr ::lpl::testing::Case lplTestCase_##testName{&lplTestSuite, #testName, __FILE__, __LINE__,     \
                                                                     &lplTestBody_##testName};                         \
        [[gnu::used, gnu::section("lpl_tests")]] static const ::lpl::testing::Case *const lplTestEntry_##testName =    \
            &lplTestCase_##testName;                                                                                   \
        static void lplTestBody_##testName(::lpl::testing::Test &test)

#endif // LPL_TESTING_TEST_HPP
