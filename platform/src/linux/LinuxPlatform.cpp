#include <lpl/core/Platform.hpp>

#if !LPL_TARGET_KERNEL

#    include <lpl/platform/linux/LinuxPlatform.hpp>

#    include <chrono>
#    include <lpl/std/cstdlib.hpp>

namespace lpl::platform::linux_host {

namespace {

/**
 * @brief Pack a color_t-style 0x00RRGGBB value into the framebuffer word.
 */
[[nodiscard]] constexpr core::u32 normalizeRgb(core::u32 colorRgb) noexcept { return colorRgb & 0x00FFFFFFu; }

[[nodiscard]] core::u64 steadyNanoseconds() noexcept
{
    return static_cast<core::u64>(
        ::std::chrono::duration_cast<::std::chrono::nanoseconds>(::std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

} // namespace

void *LinuxMemoryBackend::reserve(core::usize sizeBytes, core::usize alignment)
{
    if (sizeBytes == 0)
        return nullptr;

    const core::usize rounded = ((sizeBytes + alignment - 1) / alignment) * alignment;
    return lpl::pmr::aligned_alloc(alignment, rounded);
}

void LinuxMemoryBackend::release(void *block, core::usize /*sizeBytes*/) { lpl::pmr::free(block); }

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

LinuxDisplayBackend::LinuxDisplayBackend(core::u32 width, core::u32 height)
    : _width(width), _height(height), _buffer(static_cast<core::usize>(width) * height, 0u)
{
}

bool LinuxDisplayBackend::querySurface(SurfaceDescriptor &outDescriptor) const noexcept
{
    if (_buffer.empty())
        return false;

    outDescriptor.buffer = const_cast<core::u32 *>(_buffer.data());
    outDescriptor.physicalAddress = 0u;
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

void LinuxDisplayBackend::present() {}

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

core::Expected<GpuAllocation> LinuxGpuMemoryBackend::allocate(core::u32 sizeBytes, GpuMemoryFlags flags)
{
    if (sizeBytes == 0u)
        return core::makeError(core::ErrorCode::kInvalidArgument, "zero-size graphics-memory allocation");

    constexpr core::u32 kAlignment = 64u;
    const core::u32 rounded = (sizeBytes + (kAlignment - 1u)) & ~(kAlignment - 1u);
    void *pointer = ::operator new(rounded, ::std::align_val_t{kAlignment}, ::std::nothrow);
    if (pointer == nullptr)
        return core::makeError(core::ErrorCode::kGpuOutOfMemory, "host graphics-memory allocation failed");

    GpuAllocation allocation;
    allocation.virtualAddress = pointer;
    allocation.sizeBytes = rounded;
    allocation.flags = flags;
    allocation.physicalAddress = static_cast<core::u32>(reinterpret_cast<core::usize>(pointer));
    return allocation;
}

void LinuxGpuMemoryBackend::free(const GpuAllocation &allocation)
{
    ::operator delete(allocation.virtualAddress, ::std::align_val_t{64u}, ::std::nothrow);
}

LinuxPlatform::LinuxPlatform(core::u32 width, core::u32 height) : _display(width, height) {}

} // namespace lpl::platform::linux_host

#endif // !LPL_TARGET_KERNEL
