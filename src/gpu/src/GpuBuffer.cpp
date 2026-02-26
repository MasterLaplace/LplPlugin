// /////////////////////////////////////////////////////////////////////////////
/// @file GpuBuffer.cpp
/// @brief GpuBuffer RAII implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/gpu/GpuBuffer.hpp>
#include <lpl/core/Assert.hpp>

#include <utility>

namespace lpl::gpu {

GpuBuffer::GpuBuffer(IComputeBackend& backend, core::usize bytes)
    : backend_{&backend}
    , size_{bytes}
{
    auto result = backend_->allocate(bytes);
    LPL_VERIFY(result.has_value());
    ptr_ = result.value();
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : backend_{std::exchange(other.backend_, nullptr)}
    , ptr_{std::exchange(other.ptr_, nullptr)}
    , size_{std::exchange(other.size_, 0)}
{}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept
{
    if (this != &other)
    {
        if (ptr_ && backend_)
        {
            backend_->free(ptr_);
        }
        backend_ = std::exchange(other.backend_, nullptr);
        ptr_     = std::exchange(other.ptr_, nullptr);
        size_    = std::exchange(other.size_, 0);
    }
    return *this;
}

GpuBuffer::~GpuBuffer()
{
    if (ptr_ && backend_)
    {
        backend_->free(ptr_);
    }
}

void*       GpuBuffer::devicePtr() noexcept       { return ptr_; }
const void* GpuBuffer::devicePtr() const noexcept  { return ptr_; }
core::usize GpuBuffer::size() const noexcept       { return size_; }

core::Expected<void> GpuBuffer::upload(const void* hostSrc, core::usize bytes)
{
    LPL_ASSERT(backend_ && ptr_);
    LPL_ASSERT(bytes <= size_);
    return backend_->uploadSync(ptr_, hostSrc, bytes);
}

core::Expected<void> GpuBuffer::download(void* hostDst, core::usize bytes) const
{
    LPL_ASSERT(backend_ && ptr_);
    LPL_ASSERT(bytes <= size_);
    return backend_->downloadSync(hostDst, ptr_, bytes);
}

} // namespace lpl::gpu
