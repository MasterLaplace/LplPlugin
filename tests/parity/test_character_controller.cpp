#include <cstdio>
#include <lpl/engine/CharacterController.hpp>

static int failures = 0;

static void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++failures;
}

/**
 * @brief One sixtieth of a second, as the fixed step actually is.
 */
static const lpl::math::Fixed32 kStep = lpl::math::Fixed32::fromFloat(1.0f / 60.0f);

/**
 * @name Worlds
 * Every world below is a SURFACE, so each answers with procgen::surfaceSpan: this
 * ground, and open sky over it. That is what a heightfield always meant, and writing
 * it out is the point — the body now asks for a gap rather than for a height, and a
 * world with no roof in it has to say so rather than have it assumed.
 * @{
 */

/**
 * @brief Flat ground at y = 0.
 */
static lpl::procgen::VerticalSpan flatGround(lpl::core::i32, lpl::core::i32, lpl::math::Fixed32)
{
    return lpl::procgen::surfaceSpan(lpl::math::Fixed32{});
}

/**
 * @brief A wall: everything east of x = 5 is ten cells higher.
 */
static lpl::procgen::VerticalSpan wallGround(lpl::core::i32 x, lpl::core::i32, lpl::math::Fixed32)
{
    return lpl::procgen::surfaceSpan(x >= 5 ? lpl::math::Fixed32::fromFloat(10.0f) : lpl::math::Fixed32{});
}

/**
 * @brief A kerb: one cell of rise, low enough to step onto.
 */
static lpl::procgen::VerticalSpan kerbGround(lpl::core::i32 x, lpl::core::i32, lpl::math::Fixed32)
{
    return lpl::procgen::surfaceSpan(x >= 5 ? lpl::math::Fixed32::fromFloat(0.4f) : lpl::math::Fixed32{});
}

/**
 * @brief A plateau that ends at x = 3: beyond it, a drop.
 */
static lpl::procgen::VerticalSpan ledgeGround(lpl::core::i32 x, lpl::core::i32, lpl::math::Fixed32)
{
    return lpl::procgen::surfaceSpan(x <= 3 ? lpl::math::Fixed32{} : lpl::math::Fixed32::fromFloat(-40.0f));
}

/** @} */

/**
 * @brief Gravity, and landing. A body dropped from a height must fall, must stop at the ground, and must not
 *        oscillate around it.
 *
 * @details The body is shoved upward and left to come back; standing still afterwards must not drift.
 */
static void checkGravityAndLanding(const lpl::engine::CharacterParams &params)
{
    lpl::engine::CharacterController body;
    body.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, flatGround);
    check(body.isGrounded(), "a placed body starts standing on the ground");

    lpl::engine::CharacterIntent jump{};
    jump.jump = true;
    body.step(params, jump, kStep, flatGround);
    check(!body.isGrounded(), "a jump leaves the ground");
    check(body.verticalSpeed() > lpl::math::Fixed32{}, "and starts moving upward");

    lpl::math::Fixed32 apex = body.y();
    lpl::core::u32 ticks = 0u;
    const lpl::engine::CharacterIntent idle{};
    while (!body.isGrounded() && ticks < 600u)
    {
        body.step(params, idle, kStep, flatGround);
        if (body.y() > apex)
            apex = body.y();
        ++ticks;
    }
    std::printf("  jump: apex %.3f after %u ticks\n", apex.toFloat(), ticks);
    check(body.isGrounded(), "and it comes back down");
    check(apex > lpl::math::Fixed32::one(), "the jump clears more than a metre");
    check(apex < lpl::math::Fixed32::fromFloat(3.0f), "and less than three (it is a person, not a flea)");
    check(body.y() == lpl::math::Fixed32{}, "it settles exactly on the ground, not near it");

    const lpl::math::Fixed32 settled = body.y();
    for (lpl::core::u32 i = 0u; i < 120u; ++i)
        body.step(params, idle, kStep, flatGround);
    check(body.y() == settled, "and stays there — no jitter against the floor");
    check(body.jumpCount() == 1u, "exactly one jump was taken");
}

/**
 * @brief Walking. Forward must move forward, and the speed must be the stated one.
 *
 * @details Releasing the keys must stop it, and stop it soon.
 */
static void checkWalkingMovesForwardAtTheStatedSpeed(const lpl::engine::CharacterParams &params)
{
    lpl::engine::CharacterController body;
    body.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, flatGround);
    lpl::engine::CharacterIntent walk{};
    walk.forward = lpl::math::Fixed32::one();

    for (lpl::core::u32 i = 0u; i < 120u; ++i)
        body.step(params, walk, kStep, flatGround);

    const lpl::core::f32 speed = body.groundSpeed().toFloat();
    std::printf("  walk: two seconds -> (%.2f, %.2f), speed %.2f\n", body.x().toFloat(), body.z().toFloat(), speed);
    check(speed > 5.5f && speed < 6.5f, "it settles at the stated walk speed");
    check(body.isGrounded(), "walking on the flat never leaves the ground");

    const lpl::engine::CharacterIntent idle{};
    for (lpl::core::u32 i = 0u; i < 60u; ++i)
        body.step(params, idle, kStep, flatGround);
    check(body.groundSpeed() < lpl::math::Fixed32::fromFloat(0.05f), "and stops when the keys are released");
}

/**
 * @brief Diagonal input must not be faster than straight input. The classic bug.
 */
static void checkDiagonalInputIsNotFasterThanStraightInput(const lpl::engine::CharacterParams &params)
{
    lpl::engine::CharacterController straight;
    lpl::engine::CharacterController diagonal;
    straight.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, flatGround);
    diagonal.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, flatGround);

    lpl::engine::CharacterIntent one{};
    one.forward = lpl::math::Fixed32::one();
    lpl::engine::CharacterIntent two{};
    two.forward = lpl::math::Fixed32::one();
    two.strafe = lpl::math::Fixed32::one();

    for (lpl::core::u32 i = 0u; i < 120u; ++i)
    {
        straight.step(params, one, kStep, flatGround);
        diagonal.step(params, two, kStep, flatGround);
    }
    const lpl::core::f32 a = straight.groundSpeed().toFloat();
    const lpl::core::f32 b = diagonal.groundSpeed().toFloat();
    std::printf("  straight %.3f vs diagonal %.3f\n", a, b);
    check(b <= a + 0.01f, "walking diagonally is not faster than walking forward");
}

/**
 * @brief A wall stops it; a kerb does not.
 *
 * @details The body strafes east: strafe is +X at yaw 0.
 */
static void checkAWallStopsTheBodyAndAKerbDoesNot(const lpl::engine::CharacterParams &params)
{
    lpl::engine::CharacterController body;
    body.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, wallGround);
    lpl::engine::CharacterIntent east{};
    east.strafe = lpl::math::Fixed32::one();

    for (lpl::core::u32 i = 0u; i < 240u; ++i)
        body.step(params, east, kStep, wallGround);
    std::printf("  wall: stopped at x = %.2f (blocked %u times)\n", body.x().toFloat(), body.blockedCount());
    check(body.x() < lpl::math::Fixed32::fromFloat(5.0f), "a ten-metre face is a wall and stops the body");
    check(body.blockedCount() > 0u, "and the body says so rather than silently sticking");

    lpl::engine::CharacterController stepper;
    stepper.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, kerbGround);
    for (lpl::core::u32 i = 0u; i < 240u; ++i)
        stepper.step(params, east, kStep, kerbGround);
    std::printf("  kerb: walked to x = %.2f, y = %.2f\n", stepper.x().toFloat(), stepper.y().toFloat());
    check(stepper.x() > lpl::math::Fixed32::fromFloat(6.0f), "a low kerb is scenery and is walked over");
    check(stepper.y() > lpl::math::Fixed32::fromFloat(0.3f), "and the body ends up standing ON it");
}

/**
 * @brief Coyote time: a jump pressed just after walking off a ledge still fires.
 *
 * @details The press comes one tick late — with no coyote time it would be lost.
 */
static void checkCoyoteTimeLetsALateJumpFire(const lpl::engine::CharacterParams &params)
{
    lpl::engine::CharacterController body;
    body.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, ledgeGround);
    lpl::engine::CharacterIntent east{};
    east.strafe = lpl::math::Fixed32::one();

    lpl::core::u32 ticks = 0u;
    while (body.isGrounded() && ticks < 600u)
    {
        body.step(params, east, kStep, ledgeGround);
        ++ticks;
    }
    check(!body.isGrounded(), "walking off the plateau leaves the ground");

    lpl::engine::CharacterIntent lateJump{};
    lateJump.strafe = lpl::math::Fixed32::one();
    lateJump.jump = true;
    const lpl::math::Fixed32 before = body.verticalSpeed();
    body.step(params, lateJump, kStep, ledgeGround);
    std::printf("  coyote: vertical speed %.2f -> %.2f\n", before.toFloat(), body.verticalSpeed().toFloat());
    check(body.jumpCount() == 1u, "a jump pressed just after the ledge still fires");
}

/**
 * @brief Jump buffering: pressed in mid-air, it fires shortly after landing.
 *
 * @details The first version of this test was wrong and worth describing, because the mistake is the same one the
 *          feature exists to forgive: it stopped stepping the instant the body touched down, so the buffered press
 *          had no tick left to fire in. The jump check runs before the landing resolution — deliberately, so a jump
 *          gets its whole upward speed — which means a buffered jump lands on the tick AFTER touchdown. That is
 *          exactly what the buffer's six ticks are for.
 *
 * The body jumps from the ground, falls back down without touching anything, and jumps again to time its arc. Then
 * it re-runs the arc and presses once while still in the air, three ticks before it would touch down — the mistimed
 * press a player actually makes.
 */
static void checkJumpBufferingFiresAPressMadeInMidAir(const lpl::engine::CharacterParams &params)
{
    lpl::engine::CharacterController body;
    body.placeAt(lpl::math::Fixed32{}, lpl::math::Fixed32{}, lpl::math::Fixed32{}, flatGround);
    const lpl::engine::CharacterIntent idle{};
    lpl::engine::CharacterIntent jump{};
    jump.jump = true;

    body.step(params, jump, kStep, flatGround);
    check(body.jumpCount() == 1u, "the first jump fires from the ground");

    while (!body.isGrounded())
        body.step(params, idle, kStep, flatGround);

    body.step(params, jump, kStep, flatGround);
    check(body.jumpCount() == 2u, "and a second one after landing");

    lpl::core::u32 airborne = 0u;
    while (!body.isGrounded())
    {
        ++airborne;
        body.step(params, idle, kStep, flatGround);
    }
    body.step(params, jump, kStep, flatGround);
    const lpl::core::u32 before = body.jumpCount();
    for (lpl::core::u32 i = 0u; i + 3u < airborne; ++i)
        body.step(params, idle, kStep, flatGround);
    body.step(params, jump, kStep, flatGround);
    for (lpl::core::u32 i = 0u; i < 8u; ++i)
        body.step(params, idle, kStep, flatGround);

    std::printf("  buffer: %u jumps before, %u after a press made in mid-air\n", before, body.jumpCount());
    check(body.jumpCount() == before + 1u, "a jump pressed just before landing fires, and only once");
}

/**
 * @brief Determinism. The whole reason the heading is Fixed32.
 */
static void checkTheSameInputsFoldToTheSameState(const lpl::engine::CharacterParams &params)
{
    const auto run = [&params]() {
        lpl::engine::CharacterController body;
        body.placeAt(lpl::math::Fixed32::fromFloat(3.0f), lpl::math::Fixed32::fromFloat(-2.0f), lpl::math::Fixed32{},
                     kerbGround);
        for (lpl::core::u32 i = 0u; i < 300u; ++i)
        {
            lpl::engine::CharacterIntent intent{};
            intent.forward = lpl::math::Fixed32::one();
            intent.strafe = (i % 7u < 3u) ? lpl::math::Fixed32::one() : -lpl::math::Fixed32::one();
            intent.turn = lpl::math::Fixed32::fromFloat(0.01f);
            intent.jump = (i % 41u == 0u);
            body.step(params, intent, kStep, kerbGround);
        }
        return body.fold();
    };
    const lpl::core::u32 first = run();
    const lpl::core::u32 second = run();
    std::printf("  body_fold = 0x%08X\n", first);
    check(first == second, "the same inputs fold to the same authoritative state");
    check(first != 0x811C9DC5u, "and the fold actually consumed something");
}

/**
 * @brief A body that walks has properties, and they are cheap to state and easy to break.
 *
 * A character controller is the kind of code that always looks right and is usually
 * wrong in one specific way: it works on flat ground, and the first slope, the first
 * ledge, the first jump taken half a tick early finds the case nobody wrote down. So
 * this asserts the behaviours rather than a signature — a fold would pin the current
 * arithmetic in place and say nothing about whether the body can be walked into a
 * cliff.
 *
 * The determinism check is the exception, and it is the one that matters most for
 * this project: the body is AUTHORITATIVE state, so two runs of the same inputs must
 * fold to the same word. That is what would break if the heading ever went back to
 * being a float, and it is exactly the kind of break that shows up as nothing at all
 * on one machine.
 */
int main()
{
    std::printf("== character controller ==\n");
    const lpl::engine::CharacterParams params{};

    checkGravityAndLanding(params);

    checkWalkingMovesForwardAtTheStatedSpeed(params);

    checkDiagonalInputIsNotFasterThanStraightInput(params);

    checkAWallStopsTheBodyAndAKerbDoesNot(params);

    checkCoyoteTimeLetsALateJumpFire(params);

    checkJumpBufferingFiresAPressMadeInMidAir(params);

    checkTheSameInputsFoldToTheSameState(params);

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
