/**
 * @file IGpuMemoryBackend.hpp
 * @brief Abstract graphics-memory backend interface.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_IGPUMEMORYBACKEND_HPP
#    define LPL_PLATFORM_IGPUMEMORYBACKEND_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>

namespace lpl::platform {

/**
 * @enum GpuMemoryFlags
 * @brief Allocation hints for graphics memory (designed in from day one).
 *
 * kPersistentlyMapped keeps a stable CPU mapping for the buffer's lifetime
 * (Late-Latching of poses without re-mapping); kHostCoherent promises writes
 * are visible to the device without explicit flushes. The kernel pinned-memory
 * backend honours both implicitly (never-relocated, identity-coherent today);
 * a VirtIO-GPU backend will map them onto host-visible coherent heaps.
 */
enum class GpuMemoryFlags : core::u32 {
    kNone = 0u,
    kPersistentlyMapped = 1u << 0,
    kHostCoherent = 1u << 1,
};

/** @brief Bitwise OR of allocation flags. */
[[nodiscard]] constexpr GpuMemoryFlags operator|(GpuMemoryFlags lhs, GpuMemoryFlags rhs) noexcept
{
    return static_cast<GpuMemoryFlags>(static_cast<core::u32>(lhs) | static_cast<core::u32>(rhs));
}

/** @brief Test whether @p flags contains @p query. */
[[nodiscard]] constexpr bool hasFlag(GpuMemoryFlags flags, GpuMemoryFlags query) noexcept
{
    return (static_cast<core::u32>(flags) & static_cast<core::u32>(query)) != 0u;
}

/**
 * @struct GpuAllocation
 * @brief Handle to a graphics-memory allocation.
 *
 * physicalAddress is the base of the FIRST page; pinned memory is only
 * "contiguous-ish", so a GPU uploader must walk the per-page scatter-gather
 * list rather than assume one contiguous range.
 */
struct GpuAllocation {
    void *virtualAddress = nullptr;               ///< CPU-visible base pointer.
    core::u32 physicalAddress = 0u;               ///< Physical base of the first page.
    core::u32 sizeBytes = 0u;                     ///< Allocation size in bytes.
    GpuMemoryFlags flags = GpuMemoryFlags::kNone; ///< Flags it was created with.
};

/**
 * @class IGpuMemoryBackend
 * @brief Strategy interface for allocating GPU-attachable graphics memory.
 *
 * Concrete implementations: a Vulkan device-memory backend on Linux, a pinned
 * (never-relocated) backend over the kernel allocator in-kernel.
 */
class IGpuMemoryBackend {
public:
    virtual ~IGpuMemoryBackend() = default;

    /** @brief Allocate @p sizeBytes of graphics memory with @p flags. */
    [[nodiscard]] virtual core::Expected<GpuAllocation> allocate(core::u32 sizeBytes, GpuMemoryFlags flags) = 0;

    /** @brief Release an allocation obtained from allocate(). */
    virtual void free(const GpuAllocation &allocation) = 0;

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::platform

#endif // LPL_PLATFORM_IGPUMEMORYBACKEND_HPP
