#include <lpl/procgen/WorldBuilder.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(world_builder);

/**
 * @brief A builder whose playability was never judged does not report it as passed.
 *
 * @details gatePassed() read true until validate() ran, so a world nobody had judged was reported as
 * passing its gate, and WorldSnapshot carried that pass on.
 */
LPL_TEST(an_unjudged_world_has_not_passed_its_gate)
{
    lpl::procgen::WorldBuilder builder{7u};

    builder.terrain(16u, 16u);
    test.check(!builder.gatePassed(), "a builder that never validated reports no pass");
}
