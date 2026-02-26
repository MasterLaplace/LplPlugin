// /////////////////////////////////////////////////////////////////////////////
/// @file VulkanComputeBackend.cpp
/// @brief Vulkan Compute backend stub implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/gpu/VulkanComputeBackend.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::gpu {

struct VulkanComputeBackend::Impl {};

VulkanComputeBackend::VulkanComputeBackend()
    : impl_{std::make_unique<Impl>()}
{}

VulkanComputeBackend::~VulkanComputeBackend() = default;

core::Expected<void> VulkanComputeBackend::init()
{
    core::Log::info("VulkanComputeBackend::init (stub)");
    return {};
}

void VulkanComputeBackend::shutdown()
{
    core::Log::info("VulkanComputeBackend::shutdown (stub)");
}

core::Expected<void*> VulkanComputeBackend::allocate(core::usize /*bytes*/)
{
    return core::makeError(core::ErrorCode::NotSupported, "Vulkan compute not yet implemented");
}

void VulkanComputeBackend::free(void* /*ptr*/) {}

core::Expected<void> VulkanComputeBackend::uploadSync(void*, const void*, core::usize)
{
    return core::makeError(core::ErrorCode::NotSupported, "Vulkan compute not yet implemented");
}

core::Expected<void> VulkanComputeBackend::downloadSync(void*, const void*, core::usize)
{
    return core::makeError(core::ErrorCode::NotSupported, "Vulkan compute not yet implemented");
}

core::Expected<void> VulkanComputeBackend::dispatch(const char*, core::u32, core::u32, std::span<const core::byte>)
{
    return core::makeError(core::ErrorCode::NotSupported, "Vulkan compute not yet implemented");
}

core::Expected<void> VulkanComputeBackend::synchronize()
{
    return {};
}

const char* VulkanComputeBackend::name() const noexcept
{
    return "VulkanComputeBackend";
}

} // namespace lpl::gpu
