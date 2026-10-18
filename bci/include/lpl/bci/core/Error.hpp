/**
 * @file Error.hpp
 * @brief Structured error handling via std::expected for the BCI pipeline.
 * @author MasterLaplace
 *
 * Provides a zero-overhead, composable error model based on C++23
 * std::expected<T, Error>. Every fallible operation in the plugin
 * returns an Expected<T> instead of throwing exceptions or returning
 * boolean success flags.
 *
 * @see https://en.cppreference.com/w/cpp/utility/expected
 */

#pragma once

#include <cstdint>
#include <expected>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace lpl::bci {

/**
 * @brief Exhaustive catalog of error conditions in the BCI pipeline.
 */
enum class ErrorCode : std::uint8_t {
    kSerialPortNotFound,
    kSerialPortConfigFailed,
    kSerialReadFailed,
    kSerialWriteFailed,
    kInvalidPacket,
    kLslStreamNotFound,
    kLslConnectionFailed,
    kBrainFlowInitFailed,
    kBrainFlowStreamFailed,
    kFileNotFound,
    kFileParseError,
    kCalibrationIncomplete,
    kCalibrationInsufficientSamples,
    kFftSizeMismatch,
    kChannelCountMismatch,
    kEmptyInput,
    kInvalidArgument,
    kInvalidState,
    kPipelineEmpty,
    kMatrixNotSpd,
    kSingularMatrix,
    kMathError,
    kNotInitialized,
    kAlreadyRunning,
    kUnknown
};

/**
 * @brief Returns a short human-readable label for the given error code.
 */
[[nodiscard]] constexpr std::string_view errorCodeName(ErrorCode code) noexcept;

/**
 * @brief Structured error with code, message, and source location.
 *
 * Designed to be lightweight (no heap allocation for the code/location)
 * and informative (carries a human-readable message and the exact
 * source location where the error was created).
 */
struct Error {
    ErrorCode code;
    std::string message;
    std::source_location location;

    /**
     * @brief Factory method for constructing an Error at the call site.
     *
     * @param code    The error code identifying the failure category
     * @param message A descriptive message (may include runtime context)
     * @param loc     Automatically captured source location
     * @return A fully constructed Error value
     *
     * @code
     *   return std::unexpected(Error::make(ErrorCode::kFileNotFound, "data.csv"));
     * @endcode
     */
    [[nodiscard]] static Error make(ErrorCode code, std::string message,
                                    std::source_location loc = std::source_location::current())
    {
        return Error{code, std::move(message), loc};
    }

    /**
     * @brief Formats the error as "[Code] message (file:line)".
     */
    [[nodiscard]] std::string format() const;
};

/**
 * @brief Alias for std::expected<T, Error> used throughout the BCI plugin.
 *
 * @tparam T The success value type
 */
template <typename T> using Expected = std::expected<T, Error>;

/**
 * @brief Convenience alias for operations that produce no value on success.
 */
using ExpectedVoid = Expected<void>;

} // namespace lpl::bci

#include "Error.inl"
