/**
 * @file BandExtractor.hpp
 * @brief Extracts spectral power per frequency band from a PSD signal block.
 * @author MasterLaplace
 *
 * Operates on the output of FftProcessor (PSD magnitudes) and sums power
 * within user-defined frequency bands. This replaces the manual
 * integrale() / hz_to_bin() calls scattered throughout V1 sources.
 *
 * @see FftProcessor, FrequencyBand
 */

#pragma once

#include "IStage.hpp"
#include "core/Constants.hpp"

#include <vector>

namespace bci::dsp {

/**
 * @brief Sums PSD magnitudes within each configured frequency band.
 *
 * Input:  PSD block [fftSize/2][channelCount] from FftProcessor.
 * Output: SignalBlock [bandCount][channelCount] — one row per band.
 *
 * @code
 *   BandExtractor extractor({band::kAlpha, band::kBeta, band::kEmg}, 250.0f, 256);
 *   auto bandPowers = extractor.process(psdBlock);
 * @endcode
 */
class BandExtractor final : public IStage {
public:
    /**
     * @brief Constructs a band extractor with the specified frequency bands.
     *
     * @param bands      Frequency bands to extract
     * @param sampleRate Sampling rate in Hz (determines frequency resolution)
     * @param fftSize    FFT size used upstream (determines bin count)
     */
    BandExtractor(std::vector<FrequencyBand> bands, float sampleRate, std::size_t fftSize);

    [[nodiscard]] Expected<SignalBlock> process(const SignalBlock &input) override;
    [[nodiscard]] std::string_view name() const noexcept override;

private:
    struct BinRange {
        std::size_t low;
        std::size_t high;
    };

    std::vector<FrequencyBand> _bands;
    std::vector<BinRange> _binRanges;
    float _sampleRate;
    std::size_t _fftSize;
    std::size_t _halfSize;
};

} // namespace bci::dsp
