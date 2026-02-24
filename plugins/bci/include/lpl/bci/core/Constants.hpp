/**
 * @file Constants.hpp
 * @brief Compile-time constants for the BCI signal processing pipeline.
 * @author MasterLaplace
 *
 * Centralizes all physical and algorithmic parameters that were previously
 * duplicated across multiple source files in the V1 plugin. Changing a
 * constant here propagates consistently through the entire codebase.
 */

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

namespace lpl::bci {

inline constexpr std::size_t kDefaultChannelCount = 8;
inline constexpr float       kDefaultSampleRate   = 250.0f;
inline constexpr std::size_t kDefaultFftSize       = 256;
inline constexpr std::size_t kFftUpdateInterval    = 32;

static_assert(std::has_single_bit(kDefaultFftSize), "FFT size must be a power of two");

inline constexpr float kFrequencyResolution =
    kDefaultSampleRate / static_cast<float>(kDefaultFftSize);

inline constexpr float kSmoothingFactor     = 0.1f;
inline constexpr float kConcentrationSmooth = 0.1f;

inline constexpr float kBlinkThresholdUv    = 150.0f;
inline constexpr float kMuscleAlertThreshold = 0.5f;
inline constexpr float kMinSignalPower       = 0.01f;

inline constexpr std::size_t kMaxSamplesPerUpdate = 512;

/**
 * @brief OpenBCI Cyton 24-bit ADC scale factor (µV per LSB).
 *
 * Derivation: Vref / Gain / (2^23 - 1) * 1e6
 *   = 4.5 V / 24 / 8388607 * 1e6 ≈ 0.02235 µV/LSB
 */
inline constexpr float kCytonScaleFactor =
    4.5f / 24.0f / 8388607.0f * 1000000.0f;

inline constexpr std::uint32_t kCytonPacketSize  = 33;
inline constexpr std::uint32_t kCytonRingSlots   = 1024;
inline constexpr std::uint8_t  kCytonHeaderByte  = 0xA0;
inline constexpr std::uint8_t  kCytonFooterByte  = 0xC0;
inline constexpr std::uint32_t kCytonBaudRate    = 115200;

} // namespace lpl::bci
