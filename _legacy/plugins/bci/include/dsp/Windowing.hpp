/**
 * @file Windowing.hpp
 * @brief Window functions applied to signal blocks before spectral analysis.
 * @author MasterLaplace
 *
 * Implements the Hann window as a DSP pipeline stage. The window is
 * pre-computed at construction time and applied sample-by-sample to
 * every channel in the input SignalBlock.
 *
 * @see IStage
 */

#pragma once

#include "IStage.hpp"

#include <vector>

namespace bci::dsp {

/**
 * @brief Applies a Hann (raised cosine) window to each channel.
 *
 * The Hann window w[n] = 0.5 * (1 - cos(2π n / (N-1))) reduces spectral
 * leakage in FFT-based analysis. The window coefficients are pre-computed
 * for the target block size and reused across calls.
 *
 * @code
 *   HannWindow stage(256);
 *   auto result = stage.process(block);
 * @endcode
 */
class HannWindow final : public IStage {
public:
    /**
     * @brief Constructs a Hann window for the given block size.
     *
     * @param windowSize Number of samples per block (typically the FFT size)
     */
    explicit HannWindow(std::size_t windowSize);

    [[nodiscard]] Expected<SignalBlock> process(const SignalBlock &input) override;
    [[nodiscard]] std::string_view name() const noexcept override;

private:
    std::size_t _windowSize;
    std::vector<float> _coefficients;
};

} // namespace bci::dsp
