/**
 * @file KernelPlatform.cpp
 * @brief Kernel platform backends over the C HAL (libengine-only TU).
 *
 * Compiled only into libengine.a (LPL_TARGET_KERNEL=1); empty on the hosted
 * build. Each backend is a thin adaptor from the lpl::platform interfaces to
 * the kernel's <kernel/hal/hal.h> C contract — no logic of its own.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */
#include <lpl/core/Platform.hpp>

#if LPL_TARGET_KERNEL

#    include <lpl/platform/kernel/KernelPlatform.hpp>

#    include <kernel/hal/hal.h>

namespace lpl::platform::kernel {

// ---- Logger --------------------------------------------------------------

void KernelLogger::write(core::LogLevel level, std::string_view tag, std::string_view message)
{
    // string_view is not NUL-terminated and the kernel console takes C strings,
    // so copy into a bounded stack buffer (no heap: the logger must work before
    // and after the heap, and during failure paths).
    constexpr core::usize kMaxLine = 256;
    char line[kMaxLine];
    core::usize pos = 0;

    const auto append = [&line, &pos](std::string_view text) {
        for (char character : text)
        {
            if (pos + 1 >= kMaxLine)
                return;
            line[pos++] = character;
        }
    };

    static constexpr const char *kLevelNames[] = {"DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"};

    append("[");
    append(kLevelNames[static_cast<unsigned>(level)]);
    append("][");
    append(tag);
    append("] ");
    append(message);
    append("\n");

    line[pos] = '\0';
    hardware_abstraction_layer_console_write_string(line);
}

// ---- Clock ---------------------------------------------------------------

core::u32 KernelClockBackend::tickCount() const noexcept { return hardware_abstraction_layer_clock_tick_count(); }
core::u32 KernelClockBackend::tickHertz() const noexcept { return hardware_abstraction_layer_clock_tick_hertz(); }
core::u64 KernelClockBackend::timestampCounter() const noexcept
{
    return hardware_abstraction_layer_clock_timestamp_counter();
}

// ---- Display -------------------------------------------------------------

bool KernelDisplayBackend::querySurface(SurfaceDescriptor &outDescriptor) const noexcept
{
    hardware_abstraction_layer_surface_descriptor_t descriptor;
    if (!hardware_abstraction_layer_display_query_surface(&descriptor))
        return false;

    outDescriptor.buffer = descriptor.buffer;
    outDescriptor.physicalAddress = descriptor.physical_address;
    outDescriptor.width = descriptor.width;
    outDescriptor.height = descriptor.height;
    outDescriptor.pitch = descriptor.pitch;
    outDescriptor.bitsPerPixel = descriptor.bits_per_pixel;
    return true;
}

void KernelDisplayBackend::clear(core::u32 colorRgb) { hardware_abstraction_layer_display_clear(colorRgb); }

core::u32 KernelDisplayBackend::readPixel(core::u32 x, core::u32 y) const noexcept
{
    return hardware_abstraction_layer_display_read_pixel(x, y);
}

void KernelDisplayBackend::present() { hardware_abstraction_layer_display_present(); }

// ---- Input ---------------------------------------------------------------

bool KernelInputBackend::tryPopCharacter(char &outCharacter)
{
    return hardware_abstraction_layer_input_try_pop_character(&outCharacter);
}

core::u32 KernelInputBackend::pendingCount() const noexcept { return hardware_abstraction_layer_input_pending_count(); }

// ---- Graphics memory -----------------------------------------------------

core::Expected<GpuAllocation> KernelGpuMemoryBackend::allocate(core::u32 sizeBytes, GpuMemoryFlags flags)
{
    void *pointer = hardware_abstraction_layer_graphics_memory_allocate(sizeBytes);
    if (pointer == nullptr)
        return core::makeError(core::ErrorCode::kGpuOutOfMemory, "pinned graphics-memory allocation failed");

    GpuAllocation allocation;
    allocation.virtualAddress = pointer;
    allocation.sizeBytes = sizeBytes;
    allocation.flags = flags;

    core::u32 physical = 0u;
    if (hardware_abstraction_layer_graphics_memory_physical_address(pointer, &physical))
        allocation.physicalAddress = physical;

    return allocation;
}

void KernelGpuMemoryBackend::free(const GpuAllocation &allocation)
{
    hardware_abstraction_layer_graphics_memory_free(allocation.virtualAddress, allocation.sizeBytes);
}

} // namespace lpl::platform::kernel

#endif // LPL_TARGET_KERNEL
