#include <catch2/catch_test_macros.hpp>
#include "source/BrainFlowSource.hpp"

using namespace bci::source;
using namespace bci;

TEST_CASE("BrainFlowSource start/stop behavior", "[source][brainflow]")
{
    BrainFlowConfig cfg;
    cfg.boardId = 0; // synthetic / default id

    BrainFlowSource src(cfg);
    auto res = src.start();
    if (res) {
        // if it succeeded, we should be able to stop cleanly
        src.stop();
        SUCCEED();
    } else {
        // expect initialization failure when hardware not available
        auto code = res.error().code;
        REQUIRE((code == ErrorCode::kBrainFlowInitFailed ||
                 code == ErrorCode::kBrainFlowStreamFailed));
    }
}
