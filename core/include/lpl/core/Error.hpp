/**
 * @file Error.hpp
 * @brief Structured error type with source location tracking.
 *
 * Defines a comprehensive set of engine error codes and a lightweight
 * Error value type carrying the code, a human-readable message, and the
 * source location where the error was raised.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_ERROR_HPP
    #define LPL_CORE_ERROR_HPP

    #include "Types.hpp"

    #include <source_location>
    #include <string>
    #include <expected>

namespace lpl::core {

/**
 * @brief Engine-wide error code enumeration.
 */
enum class ErrorCode : u16 {
    kNone = 0,

    kOutOfMemory,
    kBufferOverflow,
    kBufferUnderflow,
    kInvalidArgument,
    kInvalidState,
    kNotFound,
    kAlreadyExists,
    kTimeout,
    kPermissionDenied,
    kOutOfRange,
    kIoError,
    kNotSupported,
    kCorruptedData,

    kNetworkBindFailed,
    kNetworkSendFailed,
    kNetworkReceiveFailed,
    kNetworkDisconnected,
    kNetworkDesync,
    kProtocolViolation,

    kGpuInitFailed,
    kGpuKernelFailed,
    kGpuOutOfMemory,
    kGpuSyncFailed,

    kDeviceNotFound,
    kDeviceOpenFailed,
    kDeviceReadFailed,
    kDeviceWriteFailed,
    kDeviceClosed,

    kSerializationFailed,
    kDeserializationFailed,
    kChecksumMismatch,

    kCalibrationIncomplete,
    kCalibrationFailed,

    kNotImplemented,
    kInternalError,

    // Aliases (without k prefix) for convenience
    OutOfMemory         = kOutOfMemory,
    BufferOverflow      = kBufferOverflow,
    BufferUnderflow     = kBufferUnderflow,
    InvalidArgument     = kInvalidArgument,
    InvalidState        = kInvalidState,
    NotFound            = kNotFound,
    AlreadyExists       = kAlreadyExists,
    Timeout             = kTimeout,
    OutOfRange          = kOutOfRange,
    IoError             = kIoError,
    NotSupported        = kNotSupported,
    CorruptedData       = kCorruptedData,
    NotImplemented      = kNotImplemented,
};

/**
 * @brief Structured error value carrying a code, message, and origin.
 *
 * Error is a lightweight value type (no heap allocation for the message
 * if it fits in SSO). It is intended to be stored inside Expected<T>.
 */
class Error final {
public:
    /**
     * @brief Construct an error from a code and message.
     * @param code    Enumerated error code.
     * @param message Human-readable description.
     * @param loc     Source location (auto-filled by the compiler).
     */
    explicit Error(
        ErrorCode code,
        std::string message,
        std::source_location loc = std::source_location::current()
    ) : _code(code), _message(std::move(message)), _location(loc) {}

    [[nodiscard]] ErrorCode           code()     const { return _code; }
    [[nodiscard]] const std::string & message()  const { return _message; }
    [[nodiscard]] std::source_location location() const { return _location; }

private:
    ErrorCode            _code;
    std::string          _message;
    std::source_location _location;
};

/// @brief Convenience alias for std::unexpected<Error>.
using Unexpected = std::unexpected<Error>;

/// @brief Factory function to create an unexpected error.
/// @param code Error code.
/// @param message Human-readable description.
/// @param loc Source location (auto-filled).
/// @return std::unexpected<Error>.
[[nodiscard]] inline auto makeError(
    ErrorCode code,
    std::string message,
    std::source_location loc = std::source_location::current())
{
    return std::unexpected<Error>(Error{code, std::move(message), loc});
}

} // namespace lpl::core

#endif // LPL_CORE_ERROR_HPP
