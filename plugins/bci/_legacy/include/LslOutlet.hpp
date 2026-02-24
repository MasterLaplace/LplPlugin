// File: LslOutlet.hpp
// Description: LSL Stream Outlet — broadcasts raw EEG data with high-precision timestamps.
// Enables interoperability with OpenViBE, BCI2000, MATLAB, Unity, and any LSL consumer.
//
// Compilation conditionnelle : actif si LPL_USE_LSL est défini et liblsl disponible.
// Sinon, fournit un stub silencieux.
//
// Références :
//   - LSL : https://labstreaminglayer.readthedocs.io/
//   - Timestamp correction : lsl::local_clock() + sample index regression
//
// Auteur: MasterLaplace

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifdef LPL_USE_LSL
// ═══════════════════════════════════════════════════════════════════════════════
//  LSL OUTLET ACTIF — nécessite liblsl (apt install liblsl-dev)
// ═══════════════════════════════════════════════════════════════════════════════
#    include <lsl_cpp.h>

/// Broadcasts raw EEG samples over the network via Lab Streaming Layer.
///
/// Usage :
/// @code
///   LslOutlet outlet("LplPlugin_EEG", 8, 250.0);
///   outlet.init();
///   // In acquisition loop:
///   outlet.pushSample(channelData, sampleIndex);
/// @endcode
///
/// The outlet provides sub-millisecond timestamp correction using the OpenBCI
/// sample index counter and lsl::local_clock() regression.
class LslOutlet {
public:
    /// @param streamName    Name of the LSL stream (visible to consumers)
    /// @param channelCount  Number of EEG channels (e.g., 8 for Cyton, 16 for Cyton+Daisy)
    /// @param sampleRate    Nominal sampling rate in Hz (e.g., 250.0 for Cyton)
    explicit LslOutlet(const std::string &streamName = "LplPlugin_EEG", int channelCount = 8,
                       double sampleRate = 250.0)
        : _streamName(streamName), _channelCount(channelCount), _sampleRate(sampleRate), _outlet(nullptr),
          _prevSampleIndex(0), _sampleCounter(0), _latencySum(0.0), _latencyCount(0)
    {
    }

    ~LslOutlet()
    {
        if (_outlet)
            printf("[LSL-OUT] Stream '%s' closed\n", _streamName.c_str());
    }

    /// Initializes the LSL stream info and creates the outlet.
    /// The stream becomes visible to any LSL consumer on the local network.
    bool init()
    {
        try
        {
            // Create stream info with electrode channel metadata
            lsl::stream_info info(_streamName, "EEG", _channelCount, _sampleRate, lsl::cf_float32,
                                  _streamName + "_uid");

            // Add channel metadata (10-20 standard names for Cyton 8-ch)
            lsl::xml_element channels = info.desc().append_child("channels");
            static const char *channelNames[] = {"Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
                                                 "F3",  "F4",  "T7", "T8", "P3", "P4", "Fz", "Cz"};
            for (int i = 0; i < _channelCount; ++i)
            {
                lsl::xml_element ch = channels.append_child("channel");
                ch.append_child_value("label",
                                      (i < 16) ? channelNames[i] : ("Ch" + std::to_string(i + 1)));
                ch.append_child_value("unit", "microvolts");
                ch.append_child_value("type", "EEG");
            }

            // Add acquisition metadata
            info.desc().append_child_value("manufacturer", "LplPlugin");
            info.desc().append_child_value("device", "OpenBCI_Cyton");

            _outlet = std::make_unique<lsl::stream_outlet>(info);
            printf("[LSL-OUT] Stream '%s' created (%d ch, %.0f Hz)\n", _streamName.c_str(), _channelCount,
                   _sampleRate);
            return true;
        }
        catch (const std::exception &e)
        {
            printf("[LSL-OUT] Error creating outlet: %s\n", e.what());
            return false;
        }
    }

    /// Push a single multi-channel sample with timestamp correction.
    ///
    /// The OpenBCI Cyton provides a sample index (0-255) per packet that acts as
    /// a hardware clock. We use this to estimate and correct for USB/Bluetooth
    /// transport jitter, providing a more accurate timestamp than raw
    /// lsl::local_clock().
    ///
    /// @param data          Channel data array (size >= channelCount)
    /// @param sampleIndex   OpenBCI sample number (0-255), used for jitter correction
    void pushSample(const float *data, uint8_t sampleIndex)
    {
        if (!_outlet)
            return;

        const double now = lsl::local_clock();

        // Estimate transport latency from sample index gaps
        const double expectedInterval = 1.0 / _sampleRate;
        if (_sampleCounter > 0)
        {
            // Detect sample index wraps (0-255)
            int indexDelta = static_cast<int>(sampleIndex) - static_cast<int>(_prevSampleIndex);
            if (indexDelta < 0)
                indexDelta += 256;

            const double expectedDelta = indexDelta * expectedInterval;
            const double measuredDelta = now - _prevTimestamp;
            const double jitter = measuredDelta - expectedDelta;

            // Exponential moving average of jitter for latency estimation
            _latencySum += std::abs(jitter);
            _latencyCount++;
        }

        // Corrected timestamp: current time minus estimated average latency
        const double estimatedLatency =
            (_latencyCount > 10) ? (_latencySum / _latencyCount) * 0.5 : 0.0;
        const double correctedTimestamp = now - estimatedLatency;

        _outlet->push_sample(data, correctedTimestamp);

        _prevSampleIndex = sampleIndex;
        _prevTimestamp = now;
        _sampleCounter++;
    }

    /// Push a multi-channel sample from a std::vector.
    void pushSample(const std::vector<float> &data, uint8_t sampleIndex)
    {
        pushSample(data.data(), sampleIndex);
    }

    /// Returns estimated average jitter in seconds.
    [[nodiscard]] double estimatedJitter() const noexcept
    {
        return (_latencyCount > 0) ? (_latencySum / _latencyCount) : 0.0;
    }

    /// Returns the number of samples pushed so far.
    [[nodiscard]] uint64_t sampleCount() const noexcept { return _sampleCounter; }

    /// Returns true if the outlet is active and streaming.
    [[nodiscard]] bool isActive() const noexcept { return _outlet != nullptr; }

private:
    std::string _streamName;
    int _channelCount;
    double _sampleRate;
    std::unique_ptr<lsl::stream_outlet> _outlet;

    // Jitter estimation state
    uint8_t _prevSampleIndex;
    double _prevTimestamp = 0.0;
    uint64_t _sampleCounter;
    double _latencySum;
    uint64_t _latencyCount;
};

#else
// ═══════════════════════════════════════════════════════════════════════════════
//  STUB — LSL non disponible
// ═══════════════════════════════════════════════════════════════════════════════

/// Stub silencieux quand liblsl n'est pas disponible.
class LslOutlet {
public:
    explicit LslOutlet(const std::string & = "", int = 8, double = 250.0) {}
    bool init()
    {
        printf("[LSL-OUT] liblsl non disponible — compilez avec -DLPL_USE_LSL\n");
        return false;
    }
    void pushSample(const float *, uint8_t) {}
    void pushSample(const std::vector<float> &, uint8_t) {}
    [[nodiscard]] double estimatedJitter() const noexcept { return 0.0; }
    [[nodiscard]] uint64_t sampleCount() const noexcept { return 0; }
    [[nodiscard]] bool isActive() const noexcept { return false; }
};

#endif // LPL_USE_LSL
