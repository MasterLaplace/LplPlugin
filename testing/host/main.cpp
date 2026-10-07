#include <lpl/testing/Runner.hpp>

#include <cstdio>

namespace {

/**
 * @brief The host's sink: standard output.
 */
class StandardOutput final : public lpl::testing::Sink {
public:
    void write(std::string_view text) noexcept override { std::fwrite(text.data(), 1u, text.size(), stdout); }
};

} // namespace

/**
 * @brief Runs the tests linked into this binary on the host and prints KTAP, the oracle a debug kernel
 *        is compared with.
 *
 * @details A repository whose tests use lpl::testing builds its host binary from this file and
 *          `testing/src/Runner.cpp`. `test-engine 'relief.*,codec.*'` runs only the tests those
 *          patterns name. The exit status is 0 when no test failed and the selection, if any, named
 *          one. Standard output is flushed line by line, so a line logged on standard error never
 *          lands inside a KTAP line.
 */
int main(int argc, char **argv)
{
    StandardOutput output;
    const char *selection = (argc > 1) ? argv[1] : nullptr;

    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("KTAP version 1\n1..%u\n", lpl::testing::suiteCount());

    const lpl::testing::Totals totals = lpl::testing::run(output, selection, 1u);

    const bool namedNothing = (selection != nullptr && totals.selected == 0u);

    if (namedNothing)
        std::printf("# the selection named no test\n");
    std::printf("# Totals: pass:%u fail:%u skip:%u checks:%u\n", totals.passed, totals.failed, totals.skipped,
                totals.checks);
    return (totals.failed == 0u && !namedNothing) ? 0 : 1;
}
