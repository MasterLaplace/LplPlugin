/**
 * @file Types.hpp
 * @brief Fundamental data types for the BCI signal acquisition and processing pipeline.
 * @author MasterLaplace
 *
 * Defines the vocabulary types shared across the entire bci namespace:
 * frequency bands, multi-channel samples, signal blocks, neural state, and
 * source metadata. All types are value-semantic and trivially movable.
 *
 * @see Constants.hpp for default parameter values
 * @see Error.hpp for the error handling strategy
 */

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace bci {

// ─── Frequency Band ──────────────────────────────────────────────────────────

/**
 * @brief Defines a contiguous frequency range in Hz.
 *
 * Used to specify spectral regions of interest (e.g. alpha 8–12 Hz)
 * for power extraction from a PSD (Power Spectral Density) array.
 */
struct FrequencyBand {
    float low;
    float high;
};

/**
 * @brief Standard EEG frequency bands (IEEE / clinical convention).
 */
namespace band {

inline constexpr FrequencyBand kDelta{0.5f, 4.0f};
inline constexpr FrequencyBand kTheta{4.0f, 8.0f};
inline constexpr FrequencyBand kAlpha{8.0f, 12.0f};
inline constexpr FrequencyBand kBeta{13.0f, 30.0f};
inline constexpr FrequencyBand kGamma{30.0f, 100.0f};
inline constexpr FrequencyBand kEmg{40.0f, 70.0f};

} // namespace band

// ─── Sample ──────────────────────────────────────────────────────────────────

/**
 * @brief A single multi-channel EEG sample (one time point, all channels).
 *
 * The number of channels is determined at runtime by the acquisition source.
 * Values are in microvolts (µV).
 */
struct Sample {
    std::vector<float> channels;
    double timestamp = 0.0;

    [[nodiscard]] std::size_t channelCount() const noexcept { return channels.size(); }
};

// ─── Signal Block ────────────────────────────────────────────────────────────

/**
 * @brief A contiguous block of multi-channel samples with associated metadata.
 *
 * This is the primary data unit flowing through the DSP pipeline.
 * Each stage receives a SignalBlock, transforms it, and produces a new one.
 *
 * Memory layout: data[sampleIndex][channelIndex].
 */
struct SignalBlock {
    std::vector<std::vector<float>> data;
    float sampleRate = 0.0f;
    std::size_t channelCount = 0;
    std::chrono::steady_clock::time_point timestamp{};

    /**
     * @brief Returns the number of samples (time points) in this block.
     */
    [[nodiscard]] std::size_t sampleCount() const noexcept { return data.size(); }

    /**
     * @brief Returns true if the block contains no samples.
     */
    [[nodiscard]] bool empty() const noexcept { return data.empty(); }
};

// ─── Band Power ──────────────────────────────────────────────────────────────

/**
 * @brief Spectral power per frequency band, per channel.
 *
 * Produced by the BandExtractor DSP stage. Each inner vector corresponds
 * to one channel; the outer index maps to the band ordering provided
 * at construction time.
 */
struct BandPower {
    std::vector<float> perChannel;
    float mean = 0.0f;
};

// ─── Neural State ────────────────────────────────────────────────────────────

/**
 * @brief High-level neural state derived from spectral analysis.
 *
 * Aggregates alpha/beta power per channel, the Schumacher R(t) muscle
 * tension indicator, concentration ratio, and blink detection flag.
 *
 * @see Schumacher et al., 2015 — "Closed-loop BCI for muscle fatigue"
 */
struct NeuralState {
    std::vector<float> channelAlpha;
    std::vector<float> channelBeta;

    float alphaPower = 0.0f;
    float betaPower = 0.0f;
    float concentration = 0.0f;
    float schumacherR = 0.0f;
    bool blinkDetected = false;

    /**
     * @brief Constructs a zero-initialized NeuralState for the given channel count.
     */
    explicit NeuralState(std::size_t channelCount = 0)
        : channelAlpha(channelCount, 0.0f), channelBeta(channelCount, 0.0f) {}
};

// ─── Source Info ─────────────────────────────────────────────────────────────

/**
 * @brief Metadata describing an acquisition source.
 */
struct SourceInfo {
    std::string name;
    std::size_t channelCount = 0;
    float sampleRate = 0.0f;
};

// ─── Baseline ────────────────────────────────────────────────────────────────

/**
 * @brief Statistical baseline (mean ± standard deviation) computed during calibration.
 */
struct Baseline {
    float mean = 0.0f;
    float stdDev = 0.0f;
};

// ─── Acquisition Mode ────────────────────────────────────────────────────────

/**
 * @brief Enumerates supported acquisition backends.
 */
enum class AcquisitionMode : std::uint8_t {
    kSerial,
    kSynthetic,
    kLsl,
    kCsvReplay,
    kBrainFlow
};

/**
 * @brief Returns a human-readable name for the given acquisition mode.
 */
[[nodiscard]] constexpr std::string_view acquisitionModeName(AcquisitionMode mode) noexcept
{
    switch (mode) {
        case AcquisitionMode::kSerial:    return "Serial (OpenBCI)";
        case AcquisitionMode::kSynthetic: return "Synthetic";
        case AcquisitionMode::kLsl:       return "LSL Inlet";
        case AcquisitionMode::kCsvReplay: return "CSV Replay";
        case AcquisitionMode::kBrainFlow: return "BrainFlow";
    }
    return "Unknown";
}

} // namespace bci
