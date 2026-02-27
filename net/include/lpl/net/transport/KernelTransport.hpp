/**
 * @file KernelTransport.hpp
 * @brief Zero-copy transport via the LPL kernel module (/dev/lpl0).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_TRANSPORT_KERNELTRANSPORT_HPP
    #define LPL_NET_TRANSPORT_KERNELTRANSPORT_HPP

#include <lpl/net/transport/ITransport.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::net::transport {

/**
 * @class KernelTransport
 * @brief Accesses the LPL kernel module character device for zero-copy
 *        packet I/O via mmap'd ring buffers.
 *
 * Falls back to @c SocketTransport if the device node is unavailable.
 */
class KernelTransport final : public ITransport,
                              public core::NonCopyable<KernelTransport>
{
public:
    /**
     * @brief Constructs targeting the given device path.
     * @param devicePath Path to char device (default: /dev/lpl0).
     */
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
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::net::transport

#endif // LPL_NET_TRANSPORT_KERNELTRANSPORT_HPP
