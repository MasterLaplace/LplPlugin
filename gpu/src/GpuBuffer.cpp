/**
 * @file GpuBuffer.cpp
 * @brief GpuBuffer RAII implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/gpu/GpuBuffer.hpp>
#include <lpl/core/Assert.hpp>

#include <utility>

namespace lpl::gpu {

GpuBuffer::GpuBuffer(IComputeBackend& backend, core::usize bytes)
    : _backend{&backend}
    , _size{bytes}
{
    auto result = _backend->allocate(bytes);
    LPL_VERIFY(result.has_value());
    _ptr = result.value();
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : _backend{std::exchange(other._backend, nullptr)}
    , _ptr{std::exchange(other._ptr, nullptr)}
    , _size{std::exchange(other._size, 0)}
{}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept
{
    if (this != &other)
    {
        if (_ptr && _backend)
        {
            _backend->free(_ptr);
        }
        _backend = std::exchange(other._backend, nullptr);
        _ptr     = std::exchange(other._ptr, nullptr);
        _size    = std::exchange(other._size, 0);
    }
    return *this;
}

GpuBuffer::~GpuBuffer()
{
    if (_ptr && _backend)
    {
        _backend->free(_ptr);
    }
}

void*       GpuBuffer::devicePtr() noexcept       { return _ptr; }
const void* GpuBuffer::devicePtr() const noexcept  { return _ptr; }
core::usize GpuBuffer::size() const noexcept       { return _size; }

core::Expected<void> GpuBuffer::upload(const void* hostSrc, core::usize bytes)
{
    LPL_ASSERT(_backend && _ptr);
    LPL_ASSERT(bytes <= _size);
    return _backend->uploadSync(_ptr, hostSrc, bytes);
}

core::Expected<void> GpuBuffer::download(void* hostDst, core::usize bytes) const
{
    LPL_ASSERT(_backend && _ptr);
    LPL_ASSERT(bytes <= _size);
    return _backend->downloadSync(hostDst, _ptr, bytes);
}

} // namespace lpl::gpu
