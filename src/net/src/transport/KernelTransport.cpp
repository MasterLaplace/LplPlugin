// /////////////////////////////////////////////////////////////////////////////
/// @file KernelTransport.cpp
/// @brief Kernel module transport implementation stub.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/transport/KernelTransport.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace lpl::net::transport {

struct KernelTransport::Impl
{
    const char* devicePath;
    int         fd{-1};

    explicit Impl(const char* path) : devicePath{path} {}
};

KernelTransport::KernelTransport(const char* devicePath)
    : impl_{std::make_unique<Impl>(devicePath)}
{}

KernelTransport::~KernelTransport()
{
    close();
}

core::Expected<void> KernelTransport::open()
{
    impl_->fd = ::open(impl_->devicePath, O_RDWR | O_NONBLOCK);
    if (impl_->fd < 0)
    {
        return core::makeError(core::ErrorCode::IoError,
                               "Failed to open kernel device");
    }

    core::Log::info("KernelTransport: opened device");
    return {};
}

void KernelTransport::close()
{
    if (impl_->fd >= 0)
    {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

core::Expected<core::u32> KernelTransport::send(
    std::span<const core::byte> data,
    const void* /*address*/)
{
    if (impl_->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    const auto written = ::write(impl_->fd, data.data(), data.size());
    if (written < 0)
    {
        return core::makeError(core::ErrorCode::IoError, "write() failed");
    }

    return static_cast<core::u32>(written);
}

core::Expected<core::u32> KernelTransport::receive(
    std::span<core::byte> buffer,
    void* /*fromAddress*/)
{
    if (impl_->fd < 0)
    {
        return core::makeError(core::ErrorCode::InvalidState, "Device not open");
    }

    const auto bytesRead = ::read(impl_->fd, buffer.data(), buffer.size());
    if (bytesRead < 0)
    {
        if (errno == EWOULDBLOCK || errno == EAGAIN)
        {
            return core::u32{0};
        }
        return core::makeError(core::ErrorCode::IoError, "read() failed");
    }

    return static_cast<core::u32>(bytesRead);
}

const char* KernelTransport::name() const noexcept
{
    return "KernelTransport";
}

} // namespace lpl::net::transport
