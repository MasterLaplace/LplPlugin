/**
 * @file LinuxPlatform.cpp
 * @brief Hosted (Linux) platform backends — the determinism oracle's seam.
 *
 * Compiled only into the hosted build (LPL_TARGET_KERNEL=0); empty on the
 * kernel build. The display backend is a host-memory software linear
 * framebuffer that mirrors the kernel's software-LFB path bit-for-bit, so the
 * same engine code and the same smoke produce identical pixels on both targets.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */
#include <lpl/core/Platform.hpp>

#if !LPL_TARGET_KERNEL

#    include <lpl/platform/linux/LinuxPlatform.hpp>

#    include <chrono>

namespace lpl::platform::linux_host {

namespace {

/** @brief Pack a color_t-style 0x00RRGGBB value into the framebuffer word. */
[[nodiscard]] constexpr core::u32 normalizeRgb(core::u32 colorRgb) noexcept { return colorRgb & 0x00FFFFFFu; }

[[nodiscard]] core::u64 steadyNanoseconds() noexcept
{
    using namespace ::std::chrono;
    return static_cast<core::u64>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

} // namespace

// ---- Clock ---------------------------------------------------------------
//
// The tick contract mirrors the kernel: a millisecond tick base (1000 Hz) with
// a 64-bit rdtsc for sub-tick timing. These values are OBSERVABILITY only and
// never feed the deterministic Fixed32 authority.

LinuxClockBackend::LinuxClockBackend() noexcept : _epochNanoseconds(steadyNanoseconds()) {}

core::u32 LinuxClockBackend::tickCount() const noexcept
{
    return static_cast<core::u32>((steadyNanoseconds() - _epochNanoseconds) / 1'000'000u);
}

core::u32 LinuxClockBackend::tickHertz() const noexcept { return 1000u; }

core::u64 LinuxClockBackend::timestampCounter() const noexcept
{
#    if defined(LPL_ARCH_X64) || defined(LPL_ARCH_X86)
    return __builtin_ia32_rdtsc();
#    else
    return steadyNanoseconds();
#    endif
}

// ---- Display -------------------------------------------------------------

LinuxDisplayBackend::LinuxDisplayBackend(core::u32 width, core::u32 height)
    : _width(width), _height(height), _buffer(static_cast<core::usize>(width) * height, 0u)
{
}

bool LinuxDisplayBackend::querySurface(SurfaceDescriptor &outDescriptor) const noexcept
{
    if (_buffer.empty())
        return false;

    outDescriptor.buffer = const_cast<core::u32 *>(_buffer.data());
    outDescriptor.physicalAddress = 0u; // no physical concept on the host
    outDescriptor.width = _width;
    outDescriptor.height = _height;
    outDescriptor.pitch = _width * sizeof(core::u32);
    outDescriptor.bitsPerPixel = 32u;
    return true;
}

void LinuxDisplayBackend::clear(core::u32 colorRgb)
{
    const core::u32 value = normalizeRgb(colorRgb);
    for (core::u32 &pixel : _buffer)
        pixel = value;
}

core::u32 LinuxDisplayBackend::readPixel(core::u32 x, core::u32 y) const noexcept
{
    if (x >= _width || y >= _height)
        return 0u;
    return normalizeRgb(_buffer[static_cast<core::usize>(y) * _width + x]);
}

void LinuxDisplayBackend::present()
{
    // Host software-LFB renders straight into the backing buffer; present is a
    // no-op. The GLFW/Vulkan-swapchain backend will blit/flip here.
}

// ---- Input ---------------------------------------------------------------

bool LinuxInputBackend::tryPopCharacter(char &outCharacter)
{
    if (_head >= _ring.size())
        return false;
    outCharacter = _ring[_head++];
    if (_head >= _ring.size())
    {
        _ring.clear();
        _head = 0u;
    }
    return true;
}

core::u32 LinuxInputBackend::pendingCount() const noexcept { return static_cast<core::u32>(_ring.size() - _head); }

void LinuxInputBackend::pushCharacter(char character) { _ring.push_back(character); }

// ---- Graphics memory -----------------------------------------------------

core::Expected<GpuAllocation> LinuxGpuMemoryBackend::allocate(core::u32 sizeBytes, GpuMemoryFlags flags)
{
    if (sizeBytes == 0u)
        return core::makeError(core::ErrorCode::kInvalidArgument, "zero-size graphics-memory allocation");

    // Match the kernel's 64-byte cache-line alignment; round the size up so
    // aligned_alloc's size-is-a-multiple-of-alignment precondition holds.
    constexpr core::u32 kAlignment = 64u;
    const core::u32 rounded = (sizeBytes + (kAlignment - 1u)) & ~(kAlignment - 1u);
    void *pointer = ::operator new(rounded, ::std::align_val_t{kAlignment}, ::std::nothrow);
    if (pointer == nullptr)
        return core::makeError(core::ErrorCode::kGpuOutOfMemory, "host graphics-memory allocation failed");

    GpuAllocation allocation;
    allocation.virtualAddress = pointer;
    allocation.sizeBytes = rounded;
    allocation.flags = flags;
    // No physical addressing on the host; expose the low 32 bits of the virtual
    // pointer purely so the "physical is non-zero" observability check matches.
    allocation.physicalAddress = static_cast<core::u32>(reinterpret_cast<core::usize>(pointer));
    return allocation;
}

void LinuxGpuMemoryBackend::free(const GpuAllocation &allocation)
{
    ::operator delete(allocation.virtualAddress, ::std::align_val_t{64u}, ::std::nothrow);
}

// ---- Aggregate -----------------------------------------------------------

LinuxPlatform::LinuxPlatform(core::u32 width, core::u32 height) : _display(width, height) {}

} // namespace lpl::platform::linux_host

#endif // !LPL_TARGET_KERNEL
