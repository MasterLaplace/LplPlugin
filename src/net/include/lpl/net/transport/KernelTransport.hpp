// /////////////////////////////////////////////////////////////////////////////
/// @file KernelTransport.hpp
/// @brief Zero-copy transport via the LPL kernel module (/dev/lpl0).
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/net/transport/ITransport.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::net::transport {

// /////////////////////////////////////////////////////////////////////////////
/// @class KernelTransport
/// @brief Accesses the LPL kernel module character device for zero-copy
///        packet I/O via mmap'd ring buffers.
///
/// Falls back to @c SocketTransport if the device node is unavailable.
// /////////////////////////////////////////////////////////////////////////////
class KernelTransport final : public ITransport,
                              public core::NonCopyable<KernelTransport>
{
public:
    /// @brief Constructs targeting the given device path.
    /// @param devicePath Path to char device (default: /dev/lpl0).
    explicit KernelTransport(const char* devicePath = "/dev/lpl0");
    ~KernelTransport() override;

    [[nodiscard]] core::Expected<void> open() override;
    void close() override;

    [[nodiscard]] core::Expected<core::u32> send(
        std::span<const core::byte> data,
        const void* address) override;

    [[nodiscard]] core::Expected<core::u32> receive(
        std::span<core::byte> buffer,
        void* fromAddress) override;

    [[nodiscard]] const char* name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::net::transport
