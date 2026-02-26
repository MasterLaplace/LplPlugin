// /////////////////////////////////////////////////////////////////////////////
/// @file CudaBackend.cpp
/// @brief Stub CudaBackend — real implementation requires CUDA toolchain.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/gpu/CudaBackend.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::gpu {

struct CudaBackend::Impl {};

CudaBackend::CudaBackend()
    : impl_{std::make_unique<Impl>()}
{}

CudaBackend::~CudaBackend() = default;

core::Expected<void> CudaBackend::init()
{
    core::Log::info("CudaBackend::init (stub)");
    return {};
}

void CudaBackend::shutdown()
{
    core::Log::info("CudaBackend::shutdown (stub)");
}

core::Expected<void*> CudaBackend::allocate(core::usize /*bytes*/)
{
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
}

void CudaBackend::free(void* /*ptr*/) {}

core::Expected<void> CudaBackend::uploadSync(void*, const void*, core::usize)
{
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
}

core::Expected<void> CudaBackend::downloadSync(void*, const void*, core::usize)
{
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
}

core::Expected<void> CudaBackend::dispatch(const char*, core::u32, core::u32, std::span<const core::byte>)
{
    return core::makeError(core::ErrorCode::NotSupported, "CUDA not available");
}

core::Expected<void> CudaBackend::synchronize()
{
    return {};
}

const char* CudaBackend::name() const noexcept
{
    return "CudaBackend";
}

} // namespace lpl::gpu
