#include <lpl/samples/CubePile.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(cube_pile);

namespace {

constexpr lpl::core::u32 kWidth = 192u;
constexpr lpl::core::u32 kHeight = 120u;

/** The target the pile is drawn into, static because a kernel stack cannot hold it. */
lpl::core::u32 gColour[kWidth * kHeight];
lpl::core::f32 gDepth[kWidth * kHeight];

} // namespace

/**
 * @brief The cube pile the client runs: its Fixed32 state and the image of it, after 8 and after
 *        64 ticks, are the same on both targets.
 *
 * @details Not run twice: 64 more ticks are most of the boot's test time in ring 0, and the two
 *          targets agreeing is the stronger statement of determinism.
 */
LPL_TEST(pile_settles_the_same_way)
{
    lpl::render::RenderTarget target{gColour, gDepth, kWidth, kHeight};
    const auto early = lpl::samples::runCubePileAndFold(target, 8u);
    const auto late = lpl::samples::runCubePileAndFold(target, 64u);

    test.check(early.state_signature != late.state_signature, "the state moves between tick 8 and tick 64");
    test.check(early.image_signature != late.image_signature, "and so does the image of it");

    test.measureHexadecimal("state_signature_at_8", early.state_signature);
    test.measureHexadecimal("image_signature_at_8", early.image_signature);
    test.measureHexadecimal("state_signature_at_64", late.state_signature);
    test.measureHexadecimal("image_signature_at_64", late.image_signature);
}
