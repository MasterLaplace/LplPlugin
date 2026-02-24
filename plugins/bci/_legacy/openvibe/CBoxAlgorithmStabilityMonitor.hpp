// File: CBoxAlgorithmStabilityMonitor.hpp
// Description: OpenViBE Box Algorithm skeleton — EEG Signal Stability Monitor.
//
// Implements Sollfrank et al. (2016) approach: monitors the temporal stability
// of EEG signal characteristics by computing the Riemannian distance between
// consecutive covariance matrices. High distance = unstable mental state.
//
// This box receives a streamed signal (EEG) input, computes covariance matrices
// per window, calculates the affine-invariant Riemannian distance to a reference,
// and outputs a scalar "stability" stream for enriched feedback display.
//
// Integration with OpenViBE:
//   - Input 0 : Signal (EEG channels)
//   - Output 0 : Streamed Matrix (1×1 — stability scalar [0, 1])
//   - Setting 0 : Window size (samples, default: 256)
//   - Setting 1 : Smoothing factor (default: 0.1)
//   - Setting 2 : Distance method ("riemannian" or "mahalanobis")
//
// Références :
//   - Sollfrank et al. (2016) — "The effect of multimodal and enriched feedback
//     on SMR-BCI performance"
//   - Barachant et al. (2012) — "Multiclass BCI Classification by Riemannian Geometry"
//   - OpenViBE SDK : http://openvibe.inria.fr/sdk/
//
// Auteur: MasterLaplace

#pragma once

#include "../include/RiemannianGeometry.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <numeric>
#include <vector>

// Try Eigen-backed implementation if available
#ifdef LPL_USE_EIGEN
#    include "../include/RiemannianGeometryEigen.hpp"
#endif

namespace LplOpenViBE {

/// Configuration for the Stability Monitor box
struct StabilityMonitorConfig {
    size_t windowSize = 256;        ///< Covariance computation window (samples)
    float smoothingFactor = 0.1f;   ///< EMA smoothing factor for stability output
    float maxExpectedDist = 5.0f;   ///< Maximum expected Riemannian distance (normalizer)
    size_t historySize = 10;        ///< Number of past covariance matrices to keep
    bool useMahalanobis = false;    ///< Use Mahalanobis instead of Riemannian distance
};

/// Standalone EEG stability processor (usable both inside OpenViBE and standalone).
///
/// Maintains a sliding window of covariance matrices and computes the
/// Riemannian distance between consecutive matrices as a stability indicator.
///
/// Low distance → stable mental "focus" → good BCI performance.
/// High distance → unstable signal → poor BCI performance.
///
/// @code
///   StabilityMonitorProcessor proc({.windowSize = 256, .historySize = 10});
///   proc.configure(8); // 8 EEG channels
///
///   // Per window of data:
///   float stability = proc.update(channelData);
///   // stability ∈ [0.0, 1.0] where 1.0 = perfectly stable
/// @endcode
class StabilityMonitorProcessor {
public:
    explicit StabilityMonitorProcessor(const StabilityMonitorConfig &cfg = {})
        : _cfg(cfg), _smoothedStability(0.5f)
    {
    }

    /// Configure for N EEG channels.
    void configure(size_t channelCount)
    {
        _channelCount = channelCount;
        _covHistory.clear();
        _smoothedStability = 0.5f;
        printf("[Stability] Configured: %zu ch, window=%zu, history=%zu, %s\n", _channelCount,
               _cfg.windowSize, _cfg.historySize,
               _cfg.useMahalanobis ? "Mahalanobis" : "Riemannian");
    }

    /// Update with a new window of multi-channel EEG data.
    /// @param channels  Per-channel data vectors (size ≥ windowSize)
    /// @return Smoothed stability value ∈ [0.0, 1.0]
    [[nodiscard]] float update(const std::vector<std::vector<float>> &channels)
    {
        if (channels.size() < _channelCount)
            return _smoothedStability;

        // Compute covariance matrix for this window
#ifdef LPL_USE_EIGEN
        auto cov = RiemannianEigen::compute_covariance(channels);
#else
        auto cov = RiemannianGeometry::compute_covariance(channels);
#endif

        float rawDistance = 0.0f;

        if (!_covHistory.empty())
        {
            const auto &prevCov = _covHistory.back();

#ifdef LPL_USE_EIGEN
            if (_cfg.useMahalanobis)
                rawDistance = RiemannianEigen::mahalanobis_distance(cov, prevCov);
            else
                rawDistance = RiemannianEigen::riemannian_distance(cov, prevCov);
#else
            if (_cfg.useMahalanobis)
                rawDistance = RiemannianGeometry::mahalanobis_distance(cov, prevCov);
            else
                rawDistance = RiemannianGeometry::riemannian_distance(cov, prevCov);
#endif
        }

        // Update history (sliding window)
        _covHistory.push_back(cov);
        if (_covHistory.size() > _cfg.historySize)
            _covHistory.pop_front();

        // Normalize distance to [0, 1] stability
        float normalizedDist = std::min(rawDistance / _cfg.maxExpectedDist, 1.0f);
        float instantStability = 1.0f - normalizedDist;

        // Apply exponential moving average
        _smoothedStability = _smoothedStability * (1.0f - _cfg.smoothingFactor) +
                             instantStability * _cfg.smoothingFactor;

        return _smoothedStability;
    }

    /// Get the current smoothed stability value without updating.
    [[nodiscard]] float stability() const noexcept { return _smoothedStability; }

    /// Returns true if the signal appears stable (stability > 0.6).
    [[nodiscard]] bool isStable() const noexcept { return _smoothedStability > 0.6f; }

    /// Returns the variance of recent distances (measure of consistency).
    [[nodiscard]] float distanceVariance() const noexcept
    {
        if (_distanceHistory.size() < 2)
            return 0.0f;

        float mean = 0.0f;
        for (float d : _distanceHistory)
            mean += d;
        mean /= static_cast<float>(_distanceHistory.size());

        float var = 0.0f;
        for (float d : _distanceHistory)
            var += (d - mean) * (d - mean);
        return var / static_cast<float>(_distanceHistory.size() - 1);
    }

    [[nodiscard]] const StabilityMonitorConfig &config() const noexcept { return _cfg; }

private:
    StabilityMonitorConfig _cfg;
    size_t _channelCount = 0;
    float _smoothedStability;

#ifdef LPL_USE_EIGEN
    std::deque<RiemannianEigen::MatrixXf> _covHistory;
#else
    std::deque<RiemannianGeometry::Matrix> _covHistory;
#endif

    std::deque<float> _distanceHistory;
};

// ─── OpenViBE Box Algorithm Template ─────────────────────────────────────────
// Uncomment and adapt when linking against the OpenViBE SDK:
//
// class CBoxAlgorithmStabilityMonitor : public OpenViBE::Toolkit::TBoxAlgorithm<OpenViBE::Plugins::IBoxAlgorithm>
// {
// public:
//     void release() override { delete this; }
//
//     bool initialize() override
//     {
//         size_t windowSize = FSettingValueAutoCast(*this->getBoxAlgorithmContext(), 0);
//         float smoothing = FSettingValueAutoCast(*this->getBoxAlgorithmContext(), 1);
//         std::string method = FSettingValueAutoCast(*this->getBoxAlgorithmContext(), 2);
//
//         _processor = StabilityMonitorProcessor({
//             .windowSize = windowSize,
//             .smoothingFactor = smoothing,
//             .useMahalanobis = (method == "mahalanobis")
//         });
//
//         m_decoder.initialize(*this, 0);
//         m_encoder.initialize(*this, 0);
//         return true;
//     }
//
//     bool process() override
//     {
//         // Decode windowed EEG signal
//         // float stability = _processor.update(channelData);
//         // Encode stability as output streamed matrix
//         return true;
//     }
//
//     bool uninitialize() override
//     {
//         m_decoder.uninitialize();
//         m_encoder.uninitialize();
//         return true;
//     }
//
// private:
//     StabilityMonitorProcessor _processor;
// };

} // namespace LplOpenViBE
