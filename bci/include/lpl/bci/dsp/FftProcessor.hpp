/**
 * @file FftProcessor.hpp
 * @brief Cooley-Tukey radix-2 FFT as a DSP pipeline stage.
 * @author MasterLaplace
 *
 * Transforms a time-domain SignalBlock into a frequency-domain PSD
 * (Power Spectral Density) block. This is the single, canonical FFT
 * implementation for the entire BCI plugin — replacing the five
 * duplicated copies in the V1 codebase.
 *
 * @see IStage, Pipeline
 */

#pragma once

#include "IStage.hpp"

#include <complex>
#include <cstddef>
#include <vector>

namespace lpl::bci::dsp {

/**
 * @brief Computes the Power Spectral Density of each channel via FFT.
 *
 * Input:  SignalBlock with time-domain samples [sampleCount][channelCount].
 * Output: SignalBlock with PSD magnitudes [fftSize/2][channelCount].
 *
 * The FFT uses the Cooley-Tukey radix-2 decimation-in-time algorithm
 * with bit-reversal permutation. The output is normalized by 2/N.
 *
 * @code
 *   FftProcessor fft(256);
 *   auto psd = fft.process(windowedBlock);
 * @endcode
 */
class FftProcessor final : public IStage {
public:
    /**
     * @brief Constructs an FFT processor for the given transform size.
     *
     * @param fftSize Number of time-domain samples (must be a power of 2)
     */
    explicit FftProcessor(std::size_t fftSize);

    [[nodiscard]] Expected<SignalBlock> process(const SignalBlock &input) override;
    [[nodiscard]] std::string_view name() const noexcept override;

private:
    using Complex = std::complex<float>;

    static void bitReversalPermutation(std::vector<Complex> &x);
    static void butterflyPass(std::vector<Complex> &x);

    std::size_t _fftSize;
    std::size_t _halfSize;
    float _normFactor;
};

} // namespace lpl::bci::dsp
