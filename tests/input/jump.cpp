#include <lpl/input/InputManager.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(input);

namespace {

constexpr lpl::core::u32 kEntity = 7u;

/**
 * @brief The vertical velocity a still entity leaves one movement step with.
 */
[[nodiscard]] lpl::math::Fixed32 verticalAfterStep(lpl::input::InputManager &inputs)
{
    return inputs.computeMovementVelocity(kEntity, lpl::math::Vec3<lpl::math::Fixed32>{}).y;
}

} // namespace

/**
 * @brief A blink's rising edge makes a grounded entity jump once, at an impulse of 15.
 */
LPL_TEST(a_blink_jumps_once)
{
    lpl::input::InputManager inputs;

    inputs.setNeural(kEntity, 0.0f, 0.0f, 0.0f, true);
    inputs.updateGroundedState(kEntity, {});
    test.check(verticalAfterStep(inputs) == lpl::math::Fixed32::fromInt(15), "the edge makes a grounded entity jump");
    inputs.setNeural(kEntity, 0.0f, 0.0f, 0.0f, true);
    inputs.updateGroundedState(kEntity, {});
    test.check(verticalAfterStep(inputs) == lpl::math::Fixed32{}, "a blink held over two calls jumps once");
}

/**
 * @brief Only a blink makes an entity jump: an axis an input event sets does not.
 *
 * @details Axis 15 used to carry the blink, as a float compared with 0.5: an event that set it made
 * the entity jump, and a float decided a branch of the simulation.
 */
LPL_TEST(an_axis_does_not_jump)
{
    lpl::input::InputManager inputs;

    inputs.setAxis(kEntity, 15u, 1.0f);
    inputs.updateGroundedState(kEntity, {});
    test.check(verticalAfterStep(inputs) == lpl::math::Fixed32{},
               "axis 15 at one leaves a grounded entity on the ground");
}
