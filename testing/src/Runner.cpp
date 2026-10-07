#include <lpl/testing/Runner.hpp>
#include <lpl/testing/Test.hpp>

#include <cstddef>

/**
 * @brief Bounds of the `lpl_tests` section, which holds one pointer per declared test.
 *
 * @details The host linker defines them for a section whose name is an identifier; the kernel's
 *          linker script defines them by hand.
 */
extern "C" {
extern const lpl::testing::Case *const __start_lpl_tests[];
extern const lpl::testing::Case *const __stop_lpl_tests[];
}

namespace lpl::testing {

namespace {

/** Indentation of a line that belongs to a suite's subtest. */
constexpr std::string_view kSubtestIndent = "    ";

/** Longest "suite.test" a selection pattern is matched against; a longer name is cut there. */
constexpr std::size_t kQualifiedNameCapacity = 128u;

/** Characters of the longest number a record prints: a sign and ten digits, or `0x` and eight. */
constexpr std::size_t kNumberCapacity = 11u;

/** What became of one test. */
enum class Outcome {
    kPassed,
    kSkipped,
    kFailed
};

[[nodiscard]] std::string_view formatUnsigned(core::u32 value, char (&buffer)[kNumberCapacity]) noexcept
{
    std::size_t first = kNumberCapacity;

    do
    {
        buffer[--first] = static_cast<char>('0' + value % 10u);
        value /= 10u;
    } while (value != 0u);
    return {buffer + first, kNumberCapacity - first};
}

[[nodiscard]] std::string_view formatSigned(core::i32 value, char (&buffer)[kNumberCapacity]) noexcept
{
    const bool negative = (value < 0);
    const core::u32 magnitude = negative ? 0u - static_cast<core::u32>(value) : static_cast<core::u32>(value);
    const std::string_view digits = formatUnsigned(magnitude, buffer);

    if (!negative)
        return digits;

    const std::size_t first = kNumberCapacity - digits.size() - 1u;

    buffer[first] = '-';
    return {buffer + first, digits.size() + 1u};
}

[[nodiscard]] std::string_view formatHexadecimal(core::u32 value, char (&buffer)[kNumberCapacity]) noexcept
{
    buffer[0] = '0';
    buffer[1] = 'x';
    for (std::size_t digit = 0u; digit < 8u; ++digit)
    {
        const core::u32 nibble = (value >> ((7u - digit) * 4u)) & 0xFu;
        buffer[2u + digit] = static_cast<char>(nibble < 10u ? '0' + nibble : 'A' + nibble - 10u);
    }
    return {buffer, 10u};
}

[[nodiscard]] int compareStrings(const char *lhs, const char *rhs) noexcept
{
    while (*lhs != '\0' && *lhs == *rhs)
    {
        ++lhs;
        ++rhs;
    }
    return static_cast<int>(static_cast<unsigned char>(*lhs)) - static_cast<int>(static_cast<unsigned char>(*rhs));
}

/**
 * @brief The order the runner walks the tests in: file, then line.
 *
 * @details It does not depend on where the compiler and the linker placed the entries, so the host
 *          and every kernel build path run the tests in the same order. Two tests declared on one
 *          line, by a macro, are told apart by their addresses rather than one hiding the other.
 */
[[nodiscard]] bool comesBefore(const Case &lhs, const Case &rhs) noexcept
{
    const int fileOrder = compareStrings(lhs.file, rhs.file);

    if (fileOrder != 0)
        return fileOrder < 0;
    if (lhs.line != rhs.line)
        return lhs.line < rhs.line;
    return reinterpret_cast<core::usize>(&lhs) < reinterpret_cast<core::usize>(&rhs);
}

/**
 * @brief The first test after @p previous in the runner's order, or the first of all when it is nullptr.
 *
 * @details A scan rather than a sort: the section is read-only in ring 0, and the scan needs no
 *          buffer whose size would cap the number of tests.
 */
[[nodiscard]] const Case *after(const Case *previous) noexcept
{
    const Case *next = nullptr;

    for (const Case *const *entry = __start_lpl_tests; entry < __stop_lpl_tests; ++entry)
    {
        const Case *candidate = *entry;

        if (previous != nullptr && !comesBefore(*previous, *candidate))
            continue;
        if (next == nullptr || comesBefore(*candidate, *next))
            next = candidate;
    }
    return next;
}

[[nodiscard]] core::u32 countSuiteTests(const Case *first) noexcept
{
    core::u32 tests = 0u;

    for (const Case *testCase = first; testCase != nullptr && testCase->suite == first->suite;
         testCase = after(testCase))
        ++tests;
    return tests;
}

/**
 * @brief Matches @p subject against the pattern [@p pattern, @p patternEnd), where `*` matches any run.
 */
[[nodiscard]] bool patternMatches(const char *pattern, const char *patternEnd, const char *subject) noexcept
{
    const char *star = nullptr;
    const char *resume = nullptr;

    while (*subject != '\0')
    {
        if (pattern < patternEnd && *pattern == '*')
        {
            star = pattern++;
            resume = subject;
        }
        else if (pattern < patternEnd && *pattern == *subject)
        {
            ++pattern;
            ++subject;
        }
        else if (star != nullptr)
        {
            pattern = star + 1;
            subject = ++resume;
        }
        else
        {
            return false;
        }
    }

    while (pattern < patternEnd && *pattern == '*')
        ++pattern;
    return pattern == patternEnd;
}

void qualifiedName(const Case &testCase, char (&buffer)[kQualifiedNameCapacity]) noexcept
{
    std::size_t length = 0u;

    for (const char *c = testCase.suite->name; *c != '\0' && length + 1u < kQualifiedNameCapacity; ++c)
        buffer[length++] = *c;
    if (length + 1u < kQualifiedNameCapacity)
        buffer[length++] = '.';
    for (const char *c = testCase.name; *c != '\0' && length + 1u < kQualifiedNameCapacity; ++c)
        buffer[length++] = *c;
    buffer[length] = '\0';
}

/**
 * @brief Whether the comma-separated patterns of @p selection, which ends at a space or at the end
 *        of the string, name @p testCase.
 */
[[nodiscard]] bool isSelectedBy(const char *selection, const Case &testCase) noexcept
{
    char name[kQualifiedNameCapacity];

    qualifiedName(testCase, name);

    const char *pattern = selection;

    for (;;)
    {
        const char *patternEnd = pattern;

        while (*patternEnd != '\0' && *patternEnd != ',' && *patternEnd != ' ')
            ++patternEnd;
        if (patternEnd > pattern && patternMatches(pattern, patternEnd, name))
            return true;
        if (*patternEnd != ',')
            return false;
        pattern = patternEnd + 1;
    }
}

void writeResult(Sink &sink, std::string_view indent, bool passed, core::u32 number, const char *name,
                 const char *skipReason) noexcept
{
    char digits[kNumberCapacity];

    sink.write(indent);
    sink.write(passed ? "ok " : "not ok ");
    sink.write(formatUnsigned(number, digits));
    sink.write(" ");
    sink.write(name);
    if (skipReason != nullptr)
    {
        sink.write(" # SKIP ");
        sink.write(skipReason);
    }
    sink.write("\n");
}

/**
 * @brief Runs one test, or reports why it does not run, and writes its result line.
 */
[[nodiscard]] Outcome runOne(Sink &sink, const Case &testCase, core::u32 number, const char *selection,
                             Totals &totals) noexcept
{
    Test test(testCase, sink);

    if (selection != nullptr && !isSelectedBy(selection, testCase))
    {
        test.skip("not selected");
    }
    else
    {
        totals.selected += (selection != nullptr) ? 1u : 0u;
        testCase.body(test);
        totals.checks += test.checks();
        if (test.skipReason() == nullptr && test.checks() == 0u)
            test.note("checked nothing, so it could not have failed");
    }

    const bool checkedSomething = (test.skipReason() != nullptr || test.checks() > 0u);
    const bool passed = (test.failures() == 0u && checkedSomething);

    writeResult(sink, kSubtestIndent, passed, number, testCase.name, passed ? test.skipReason() : nullptr);
    if (!passed)
    {
        ++totals.failed;
        return Outcome::kFailed;
    }
    if (test.skipReason() != nullptr)
    {
        ++totals.skipped;
        return Outcome::kSkipped;
    }
    ++totals.passed;
    return Outcome::kPassed;
}

/**
 * @brief Runs the suite that starts at @p first as one KTAP subtest.
 *
 * @return The first test of the next suite, or nullptr after the last.
 */
[[nodiscard]] const Case *runSuite(Sink &sink, const Case *first, const char *selection, core::u32 suiteNumber,
                                   Totals &totals) noexcept
{
    const Suite *suite = first->suite;
    char digits[kNumberCapacity];
    core::u32 number = 0u;
    core::u32 failed = 0u;
    core::u32 skipped = 0u;
    const Case *testCase = first;

    sink.write(kSubtestIndent);
    sink.write("KTAP version 1\n");
    sink.write(kSubtestIndent);
    sink.write("# Subtest: ");
    sink.write(suite->name);
    sink.write("\n");
    sink.write(kSubtestIndent);
    sink.write("1..");
    sink.write(formatUnsigned(countSuiteTests(first), digits));
    sink.write("\n");

    for (; testCase != nullptr && testCase->suite == suite; testCase = after(testCase))
    {
        const Outcome outcome = runOne(sink, *testCase, ++number, selection, totals);

        failed += (outcome == Outcome::kFailed) ? 1u : 0u;
        skipped += (outcome == Outcome::kSkipped) ? 1u : 0u;
    }

    writeResult(sink, "", failed == 0u, suiteNumber, suite->name, (skipped == number) ? "every test skipped" : nullptr);
    return testCase;
}

} // namespace

Test::Test(const Case &testCase, Sink &sink) noexcept : _case(testCase), _sink(sink) {}

void Test::writePrefix() noexcept
{
    _sink.write(kSubtestIndent);
    _sink.write("# ");
    _sink.write(_case.suite->name);
    _sink.write(".");
    _sink.write(_case.name);
    _sink.write(": ");
}

void Test::writeRecord(const char *key, std::string_view value) noexcept
{
    writePrefix();
    _sink.write(key);
    _sink.write("=");
    _sink.write(value);
    _sink.write("\n");
}

bool Test::check(bool condition, const char *claim) noexcept
{
    ++_checks;
    if (condition)
        return true;

    ++_failures;
    writePrefix();
    _sink.write("does not hold: ");
    _sink.write(claim);
    _sink.write("\n");
    return false;
}

void Test::skip(const char *reason) noexcept { _skipReason = reason; }

void Test::measure(const char *key, core::u32 value) noexcept
{
    char digits[kNumberCapacity];

    writeRecord(key, formatUnsigned(value, digits));
}

void Test::measure(const char *key, core::i32 value) noexcept
{
    char digits[kNumberCapacity];

    writeRecord(key, formatSigned(value, digits));
}

void Test::measureHexadecimal(const char *key, core::u32 value) noexcept
{
    char digits[kNumberCapacity];

    writeRecord(key, formatHexadecimal(value, digits));
}

void Test::note(const char *text) noexcept
{
    writePrefix();
    _sink.write(text);
    _sink.write("\n");
}

core::u32 suiteCount() noexcept
{
    core::u32 suites = 0u;
    const Suite *current = nullptr;

    for (const Case *testCase = after(nullptr); testCase != nullptr; testCase = after(testCase))
    {
        if (testCase->suite != current)
        {
            current = testCase->suite;
            ++suites;
        }
    }
    return suites;
}

Totals run(Sink &sink, const char *selection, core::u32 firstSuiteNumber) noexcept
{
    Totals totals;
    core::u32 suiteNumber = firstSuiteNumber;

    for (const Case *first = after(nullptr); first != nullptr;)
        first = runSuite(sink, first, selection, suiteNumber++, totals);
    return totals;
}

} // namespace lpl::testing
