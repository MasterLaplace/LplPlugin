// File: CBoxAlgorithmMuscleRelaxation.hpp
// Description: OpenViBE Box Algorithm skeleton — Muscular Relaxation Monitor.
//
// Implements Schumacher et al. (2015) metric: average power in the 40-70 Hz
// (Gamma) band across EEG channels as an indicator of muscular artifact
// activity. High R(t) indicates the user is tensing muscles instead of
// performing pure motor imagery.
//
// This box receives a streamed signal (EEG) input, computes R(t) per window,
// and outputs a scalar "muscle_relaxation" stream suitable for feedback display.
//
// Integration with OpenViBE:
//   - Input 0 : Signal (EEG channels)
//   - Output 0 : Streamed Matrix (1×1 — R(t) scalar)
//   - Setting 0 : Lower frequency bound (Hz, default: 40)
//   - Setting 1 : Upper frequency bound (Hz, default: 70)
//   - Setting 2 : Alert threshold (default: 2.0)
//
// Références :
//   - Schumacher et al. (2015) — "Muscular relaxation state in MI-BCI"
//   - OpenViBE SDK : http://openvibe.inria.fr/sdk/
//
// Auteur: MasterLaplace

#pragma once

#include "../include/SignalMetrics.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
//  OpenViBE Box Algorithm : Muscle Relaxation Monitor
//
//  NOTE: This is a SKELETON that demonstrates the algorithm structure.
//  Full compilation requires the OpenViBE SDK headers and link libraries.
//  The core computation uses our proven SignalMetrics::schumacher().
// ═══════════════════════════════════════════════════════════════════════════════

namespace LplOpenViBE {

/// Configuration for the Muscle Relaxation box
struct MuscleRelaxationConfig {
    float lowerFreqHz = 40.0f;   ///< Lower bound of the Gamma band (Hz)
    float upperFreqHz = 70.0f;   ///< Upper bound of the Gamma band (Hz)
    float alertThreshold = 2.0f; ///< R(t) threshold above which muscle_alert is triggered
    float sampleRate = 250.0f;   ///< Expected sampling rate (Hz)
    uint16_t fftSize = 256;      ///< FFT window size (samples)
};

/// Standalone muscle relaxation processor (usable both inside OpenViBE and standalone).
///
/// This class encapsulates the Schumacher R(t) computation in a reusable
/// component that can be wrapped by an OpenViBE CBoxAlgorithm or used directly.
///
/// @code
///   MuscleRelaxationProcessor proc({.sampleRate = 250, .fftSize = 256});
///   proc.configure(8); // 8 EEG channels
///
///   // Per-window:
///   float Rt = proc.compute(psdChannels);
///   bool alert = proc.isAlert(Rt);
/// @endcode
class MuscleRelaxationProcessor {
public:
    explicit MuscleRelaxationProcessor(const MuscleRelaxationConfig &cfg = {}) : _cfg(cfg)
    {
        _lowerBin = static_cast<uint16_t>(_cfg.lowerFreqHz * _cfg.fftSize / _cfg.sampleRate);
        _upperBin = static_cast<uint16_t>(_cfg.upperFreqHz * _cfg.fftSize / _cfg.sampleRate);
    }

    /// Set up for N channels.
    void configure(size_t channelCount)
    {
        _channelCount = channelCount;
        printf("[MuscleRelaxation] Configured: %zu ch, bins [%u, %u], threshold=%.1f\n",
               _channelCount, _lowerBin, _upperBin, _cfg.alertThreshold);
    }

    /// Compute R(t) from a multi-channel PSD matrix.
    /// @param psdChannels  Per-channel PSD vectors (one per channel, half-spectrum)
    /// @return Schumacher R(t) — average inter-channel power in the Gamma band
    [[nodiscard]] float compute(const std::vector<std::vector<float>> &psdChannels) const noexcept
    {
        return SignalMetrics::schumacher(psdChannels, _lowerBin, _upperBin);
    }

    /// Returns true if R(t) exceeds the configured alert threshold.
    [[nodiscard]] bool isAlert(float Rt) const noexcept { return Rt > _cfg.alertThreshold; }

    /// Returns the current config.
    [[nodiscard]] const MuscleRelaxationConfig &config() const noexcept { return _cfg; }

private:
    MuscleRelaxationConfig _cfg;
    size_t _channelCount = 0;
    uint16_t _lowerBin = 0;
    uint16_t _upperBin = 0;
};

// ─── OpenViBE Box Algorithm Template ─────────────────────────────────────────
// Uncomment and adapt when linking against the OpenViBE SDK:
//
// class CBoxAlgorithmMuscleRelaxation : public OpenViBE::Toolkit::TBoxAlgorithm<OpenViBE::Plugins::IBoxAlgorithm>
// {
// public:
//     void release() override { delete this; }
//
//     bool initialize() override
//     {
//         // Read settings from OpenViBE UI
//         float lower = FSettingValueAutoCast(*this->getBoxAlgorithmContext(), 0);
//         float upper = FSettingValueAutoCast(*this->getBoxAlgorithmContext(), 1);
//         float threshold = FSettingValueAutoCast(*this->getBoxAlgorithmContext(), 2);
//
//         _processor = MuscleRelaxationProcessor({.lowerFreqHz = lower, .upperFreqHz = upper, .alertThreshold = threshold});
//
//         // Decode signal input
//         m_decoder.initialize(*this, 0);
//         m_encoder.initialize(*this, 0);
//
//         return true;
//     }
//
//     bool process() override
//     {
//         // Decode input signal chunks
//         // Compute PSD per channel
//         // float Rt = _processor.compute(psdChannels);
//         // Encode Rt as output streamed matrix
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
//     MuscleRelaxationProcessor _processor;
//     // OpenViBE::Toolkit::TSignalDecoder<CBoxAlgorithmMuscleRelaxation> m_decoder;
//     // OpenViBE::Toolkit::TStreamedMatrixEncoder<CBoxAlgorithmMuscleRelaxation> m_encoder;
// };

} // namespace LplOpenViBE
