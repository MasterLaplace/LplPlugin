#include <lpl/render/CommandBuffer.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(command_buffer);

/**
 * @brief Gate P6 command buffer: a finalized buffer refuses a fifth command, and submitting it
 *        twice latches the poses given at submission, not at recording.
 */
LPL_TEST(late_latching_reads_the_poses_at_submission)
{
    lpl::render::CommandBuffer buffer;
    lpl::render::Pose poses[4];

    for (lpl::core::u32 index = 0u; index < 4u; ++index)
        buffer.record(
            lpl::render::DrawCommand{0x1000u + index * 0x100u, 0x9000u + index * 0x40u, 36u, 1u, index, index & 1u});
    buffer.finalize();
    buffer.record(lpl::render::DrawCommand{});
    test.check(buffer.count() == 4u, "a finalized buffer refuses another command");

    for (lpl::core::u32 index = 0u; index < 4u; ++index)
        poses[index].x = lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(index));

    const auto first = lpl::render::submitLateLatched(buffer, poses, 4u);

    for (lpl::core::u32 index = 0u; index < 4u; ++index)
        poses[index].x = poses[index].x + lpl::math::Fixed32::fromFloat(0.5f);

    const auto second = lpl::render::submitLateLatched(buffer, poses, 4u);

    test.check(first.draws == 4u && second.draws == 4u, "both submissions draw the four commands");
    test.check(first.latched_signature != second.latched_signature, "and each latches its own poses");

    test.measureHexadecimal("recording_signature", buffer.recordingSignature());
    test.measureHexadecimal("first_latched_signature", first.latched_signature);
    test.measureHexadecimal("second_latched_signature", second.latched_signature);
}
