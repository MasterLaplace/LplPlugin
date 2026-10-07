#include <lpl/platform/kernel/KernelPlatform.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(kernel_platform);

/**
 * @brief Gate P2 HAL, display: the surface is cleared to a colour and gives it back.
 */
LPL_TEST(display_gives_back_what_it_cleared)
{
    constexpr lpl::core::u32 kClearColour = 0x00112233u;
    lpl::platform::kernel::KernelPlatform platform;
    lpl::platform::IDisplayBackend &display = platform.display();
    lpl::platform::SurfaceDescriptor surface;

    if (!display.querySurface(surface))
    {
        test.skip("no display surface on this boot");
        return;
    }
    test.check(surface.width > 0u && surface.height > 0u && surface.bitsPerPixel == 32u,
               "the surface has a size and 32 bits per pixel");

    display.clear(kClearColour);
    display.present();
    test.check(display.readPixel(0u, 0u) == kClearColour, "a cleared pixel reads back as the clear colour");

    test.measure("width", surface.width);
    test.measure("height", surface.height);
}

/**
 * @brief Gate P2 HAL, clock: the tick has a rate, and the timestamp counter moves forward.
 */
LPL_TEST(clock_ticks_and_its_counter_advances)
{
    lpl::platform::kernel::KernelPlatform platform;
    lpl::platform::IClockBackend &clock = platform.clock();
    const lpl::core::u64 before = clock.timestampCounter();
    const lpl::core::u64 after = clock.timestampCounter();

    test.check(clock.tickHertz() > 0u, "the tick has a rate");
    test.check(after != 0u && after >= before, "and the timestamp counter moves forward");

    test.measure("tick_hertz", clock.tickHertz());
}

/**
 * @brief Gate P2 HAL, input: the ring hands out no more characters than it said it held, and holds
 *        none afterwards.
 *
 * @note The kernel counts scan codes, an upper bound on the characters: a key release decodes to
 *       nothing (MasterLaplace/LplKernel#481).
 */
LPL_TEST(input_ring_drains)
{
    lpl::platform::kernel::KernelPlatform platform;
    lpl::platform::IInputBackend &input = platform.input();
    const lpl::core::u32 pending = input.pendingCount();
    lpl::core::u32 popped = 0u;
    char character = '\0';

    while (input.tryPopCharacter(character))
        ++popped;
    test.check(popped <= pending, "the ring hands out no more characters than it said it held");
    test.check(input.pendingCount() == 0u, "and holds none afterwards");

    test.measure("pending", pending);
    test.measure("popped", popped);
}

/**
 * @brief Gate P2 HAL, graphics memory: a pinned page is handed out with a physical address, and freed.
 */
LPL_TEST(graphics_memory_is_pinned_and_translated)
{
    lpl::platform::kernel::KernelPlatform platform;
    lpl::platform::IGpuMemoryBackend &memory = platform.gpuMemory();
    auto allocation = memory.allocate(4096u, lpl::platform::GpuMemoryFlags::kPersistentlyMapped |
                                                 lpl::platform::GpuMemoryFlags::kHostCoherent);

    if (!test.check(allocation.has_value(), "a page of pinned, coherent memory is allocated"))
        return;
    test.check(allocation->physicalAddress != 0u, "with a physical address the device can use");
    memory.free(*allocation);
}
