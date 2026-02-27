/**
 * @file SyntheticGenerator.hpp
 * @brief Deterministic multi-band EEG signal generator for testing and simulation.
 * @author MasterLaplace
 *
 * Generates synthetic EEG signals with configurable oscillators (alpha,
 * beta, gamma/EMG bands), Gaussian noise, and stochastic blink artifacts.
 * Fully deterministic when seeded, enabling reproducible test scenarios.
 *
 * @see SyntheticSource
 * @see OpenBCI Cyton specification: 8 channels, 250 Hz, 24-bit resolution
 */

#pragma once

#include "core/Constants.hpp"
#include "core/Types.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace bci::source {

/**
 * @brief Configuration for a single sinusoidal oscillator within a band.
 */
struct BandOscillator {
    float freqHz;
    float amplitudeUv;
    float phaseOffset;
};

/**
 * @brief Generation profile controlling the synthetic EEG characteristics.
 */
struct SyntheticProfile {
    std::vector<BandOscillator> oscillators = {
        {10.0f, 15.0f, 0.0f},
        {20.0f,  8.0f, 0.3f},
        {50.0f,  3.0f, 0.7f},
    };

    float noiseAmplitudeUv    = 2.0f;
    float blinkProbability    = 0.005f;
    float blinkAmplitudeUv    = 200.0f;
    float blinkDurationSec    = 0.15f;
    float channelPhaseSpread  = 0.2f;
};

/**
 * @brief Generates deterministic multi-channel EEG samples.
 *
 * @code
 *   SyntheticGenerator gen(42);
 *   auto samples = gen.generate(256);
 *   // samples[t] is a Sample with 8 channels in µV
 * @endcode
 */
class SyntheticGenerator {
public:
    /**
     * @brief Constructs a generator with the given PRNG seed.
     *
     * @param seed Random seed (0 = time-based non-deterministic)
     * @param channelCount Number of EEG channels to generate
     */
    explicit SyntheticGenerator(
        std::uint64_t seed = 0,
        std::size_t channelCount = kDefaultChannelCount);

    /**
     * @brief Replaces the current generation profile.
     *
     * @param profile New oscillator/noise/blink configuration
     */
    void setProfile(const SyntheticProfile &profile);

    /**
     * @brief Returns the current generation profile.
     */
    [[nodiscard]] const SyntheticProfile &profile() const noexcept;

    /**
     * @brief Generates a batch of multi-channel samples.
     *
     * @param count Number of time-domain samples to produce
     * @return Vector of Sample objects with channelCount channels each
     */
    [[nodiscard]] std::vector<Sample> generate(std::size_t count);

    /**
     * @brief Resets the generator state with a new seed.
     *
     * @param seed New PRNG seed (0 = time-based)
     */
    void reset(std::uint64_t seed = 0);

    /**
     * @brief Returns the total number of samples generated since construction or reset.
     */
    [[nodiscard]] std::uint64_t sampleIndex() const noexcept;

private:
    SyntheticProfile _profile;
    std::mt19937_64 _rng;
    std::normal_distribution<float> _noiseDist;
    std::uniform_real_distribution<float> _blinkDist;
    std::uint64_t _sampleIndex;
    int _blinkRemaining;
    std::size_t _channelCount;
};

} // namespace bci::source
