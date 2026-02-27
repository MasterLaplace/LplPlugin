// File: BrainFlowSource.hpp
// Description: BciSource implementation using the BrainFlow library.
// BrainFlow provides a hardware-agnostic API supporting 20+ EEG boards
// (OpenBCI, Muse, Neurosity, etc.) with built-in signal processing.
//
// Compilation conditionnelle :
//   - LPL_USE_BRAINFLOW : utilise BrainFlow SDK
//   - Par défaut        : stub qui échoue à init()
//
// Références :
//   - BrainFlow : https://brainflow.readthedocs.io/
//   - Boards : https://brainflow.readthedocs.io/en/stable/SupportedBoards.html
//
// Auteur: MasterLaplace

#pragma once

#include "BciSource.hpp"
#include "SignalMetrics.hpp"
#include <cstdio>

#ifdef LPL_USE_BRAINFLOW
// ═══════════════════════════════════════════════════════════════════════════════
//  BRAINFLOW ACTIF
// ═══════════════════════════════════════════════════════════════════════════════
#    include <board_shim.h>
#    include <data_filter.h>

/// BCI source using BrainFlow for hardware-agnostic EEG acquisition.
///
/// Supports all boards registered in BrainFlow (OpenBCI Cyton, Ganglion,
/// Muse S, Muse 2, Crown, Neurosity, synthetic, etc.)
///
/// Usage :
/// @code
///   BrainFlowSource src(BoardIds::SYNTHETIC_BOARD);
///   if (src.init()) {
///       NeuralState state;
///       src.update(state);
///   }
/// @endcode
class BrainFlowSource final : public BciSource {
public:
    /// @param boardId       BrainFlow board ID (e.g., CYTON_BOARD, SYNTHETIC_BOARD)
    /// @param serialPort    Serial port for hardware boards (e.g., "/dev/ttyUSB0")
    /// @param serialNumber  Serial number of the device (optional)
    explicit BrainFlowSource(int boardId = static_cast<int>(BoardIds::SYNTHETIC_BOARD),
                              const std::string &serialPort = "",
                              const std::string &serialNumber = "")
        : _boardId(boardId), _running(false), _samplesSinceLastFFT(0), _sampleIndex(0)
    {
        BrainFlowInputParams params;
        if (!serialPort.empty())
            params.serial_port = serialPort;
        if (!serialNumber.empty())
            params.serial_number = serialNumber;

        _board = std::make_unique<BoardShim>(boardId, params);

        // Pre-allocate FFT buffers
        for (size_t ch = 0; ch < BCI_CHANNELS; ++ch)
        {
            _timeDomainBuffers[ch].resize(FFT_SIZE, 0.0f);
            _fftInputs[ch].resize(FFT_SIZE);
        }
    }

    ~BrainFlowSource() override { stop(); }

    [[nodiscard]] bool init() override
    {
        try
        {
            // Enable BrainFlow logging
            BoardShim::enable_dev_board_logger();

            _board->prepare_session();

            _eegChannels = BoardShim::get_eeg_channels(_boardId);
            _sampleRate = BoardShim::get_sampling_rate(_boardId);

            printf("[BRAINFLOW] Board %d prepared (%zu EEG channels, %d Hz)\n", _boardId,
                   _eegChannels.size(), _sampleRate);

            _board->start_stream();
            _running = true;

            printf("[BRAINFLOW] Streaming started\n");
            return true;
        }
        catch (const BrainFlowException &e)
        {
            printf("[BRAINFLOW] Error: %s (code: %d)\n", e.what(), e.exit_code);
            return false;
        }
    }

    void update(NeuralState &state) override
    {
        if (!_running || !_board)
            return;

        try
        {
            // Get all available data (non-blocking)
            BrainFlowArray<double, 2> data = _board->get_board_data();
            int numSamples = data.get_size(1);

            if (numSamples == 0)
                return;

            const size_t chCount = std::min(_eegChannels.size(), static_cast<size_t>(BCI_CHANNELS));

            for (int s = 0; s < numSamples; ++s)
            {
                for (size_t ch = 0; ch < chCount; ++ch)
                {
                    _timeDomainBuffers[ch][_sampleIndex] =
                        static_cast<float>(data.at(_eegChannels[ch], s));
                }

                // Blink detection on first channel
                if (chCount > 0)
                    state.blinkDetected = (std::abs(data.at(_eegChannels[0], s)) > 150.0);

                _sampleIndex = (_sampleIndex + 1) % FFT_SIZE;
                _samplesSinceLastFFT++;

                if (_samplesSinceLastFFT >= UPDATE_INTERVAL)
                {
                    processFFT(state, chCount);
                    _samplesSinceLastFFT = 0;
                }
            }
        }
        catch (const BrainFlowException &e)
        {
            printf("[BRAINFLOW] Update error: %s\n", e.what());
        }
    }

    void stop() override
    {
        if (_running && _board)
        {
            try
            {
                _board->stop_stream();
                _board->release_session();
            }
            catch (...)
            {
            }
            _running = false;
            printf("[BRAINFLOW] Session released\n");
        }
    }

    [[nodiscard]] const char *name() const noexcept override { return "BrainFlowSource"; }
    [[nodiscard]] BciMode mode() const noexcept override { return BciMode::BrainFlow; }

private:
    void processFFT(NeuralState &state, size_t chCount)
    {
        using Complex = FastFourierTransform::Complex;
        const float normFactor = 2.0f / FFT_SIZE;
        const float freqRes = static_cast<float>(_sampleRate) / FFT_SIZE;

        const uint16_t BIN_40HZ = static_cast<uint16_t>(40.0f * FFT_SIZE / _sampleRate);
        const uint16_t BIN_70HZ = static_cast<uint16_t>(70.0f * FFT_SIZE / _sampleRate);

        std::vector<std::vector<float>> psdChannels(chCount, std::vector<float>(FFT_SIZE / 2, 0.0f));

        float alphaSum = 0.0f, betaSum = 0.0f;

        for (size_t ch = 0; ch < chCount; ++ch)
        {
            for (size_t i = 0; i < FFT_SIZE; ++i)
            {
                size_t idx = (_sampleIndex + i) % FFT_SIZE;
                _fftInputs[ch][i] = Complex(_timeDomainBuffers[ch][idx], 0.0f);
            }

            FastFourierTransform::apply_window(_fftInputs[ch]);
            FastFourierTransform::compute(_fftInputs[ch]);

            float chAlpha = 0.0f, chBeta = 0.0f;
            for (size_t i = 1; i < FFT_SIZE / 2; ++i)
            {
                float freq = static_cast<float>(i) * freqRes;
                float magnitude = std::abs(_fftInputs[ch][i]) * normFactor;
                psdChannels[ch][i] = magnitude;

                if (freq >= 8.0f && freq <= 12.0f)
                    chAlpha += magnitude;
                else if (freq >= 13.0f && freq <= 30.0f)
                    chBeta += magnitude;
            }

            state.channelAlpha[ch] = chAlpha;
            state.channelBeta[ch] = chBeta;
            alphaSum += chAlpha;
            betaSum += chBeta;
        }

        state.schumacherR = SignalMetrics::schumacher(psdChannels, BIN_40HZ, BIN_70HZ);

        const float sf = 0.1f;
        state.alphaPower = state.alphaPower * (1.0f - sf) + (alphaSum / chCount) * sf;
        state.betaPower = state.betaPower * (1.0f - sf) + (betaSum / chCount) * sf;

        const float totalPower = state.alphaPower + state.betaPower + 0.0001f;
        state.concentration = state.concentration * 0.9f + (state.betaPower / totalPower) * 0.1f;
    }

    static constexpr size_t FFT_SIZE = 256u;
    static constexpr size_t UPDATE_INTERVAL = 32u;

    int _boardId;
    std::unique_ptr<BoardShim> _board;
    std::vector<int> _eegChannels;
    int _sampleRate = 250;
    bool _running;

    std::array<std::vector<float>, BCI_CHANNELS> _timeDomainBuffers;
    std::array<std::vector<FastFourierTransform::Complex>, BCI_CHANNELS> _fftInputs;
    size_t _sampleIndex;
    size_t _samplesSinceLastFFT;
};

#else
// ═══════════════════════════════════════════════════════════════════════════════
//  STUB — BrainFlow non disponible
// ═══════════════════════════════════════════════════════════════════════════════

class BrainFlowSource final : public BciSource {
public:
    explicit BrainFlowSource(int = 0, const std::string & = "", const std::string & = "") {}

    [[nodiscard]] bool init() override
    {
        printf("[BRAINFLOW] BrainFlow non disponible — compilez avec -DLPL_USE_BRAINFLOW\n");
        return false;
    }

    void update(NeuralState &) override {}
    void stop() override {}

    [[nodiscard]] const char *name() const noexcept override
    {
        return "BrainFlowSource (STUB — BrainFlow absent)";
    }
    [[nodiscard]] BciMode mode() const noexcept override { return BciMode::BrainFlow; }
};

#endif // LPL_USE_BRAINFLOW
