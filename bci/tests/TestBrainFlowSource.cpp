#include "lpl/bci/source/BrainFlowSource.hpp"
#include <catch2/catch_test_macros.hpp>

namespace lpl::bci {

/**
 * @brief A start that succeeds stops cleanly; one that fails, as it does when the hardware is not
 *        available, reports a BrainFlow error.
 */
TEST_CASE("BrainFlowSource start/stop behavior", "[source][brainflow]")
{
    lpl::bci::source::BrainFlowConfig cfg;
    cfg.boardId = 0;

    lpl::bci::source::BrainFlowSource src(cfg);
    auto res = src.start();
    if (res)
    {
        src.stop();
        SUCCEED();
    }
    else
    {
        auto code = res.error().code;
        REQUIRE((code == ErrorCode::kBrainFlowInitFailed || code == ErrorCode::kBrainFlowStreamFailed));
    }
}

} // namespace lpl::bci
