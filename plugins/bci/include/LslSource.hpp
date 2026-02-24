// File: LslSource.hpp
// Description: Source BCI via Lab Streaming Layer (LSL) — plugin subsidiaire.
// Compilation conditionnelle : actif si LPL_USE_LSL est défini et liblsl disponible.
// Sinon, fournit un stub qui échoue gracieusement à l'initialisation.
//
// LSL est le standard de facto pour le streaming EEG en recherche BCI.
// Permet l'interopérabilité avec OpenViBE, BCI2000, BrainFlow, etc.
//
// Références :
//   - LSL : https://labstreaminglayer.readthedocs.io/
//   - API C++ : stream_inlet, stream_outlet, resolve_stream
//
// Auteur: MasterLaplace

#pragma once

#include "BciSource.hpp"
#include "SignalMetrics.hpp"
#include <cstdio>

#ifdef LPL_USE_LSL
// ═══════════════════════════════════════════════════════════════════════════════
//  LSL ACTIF — nécessite liblsl (apt install liblsl-dev ou build from source)
// ═══════════════════════════════════════════════════════════════════════════════
#    include <lsl_cpp.h>

class LslSource final : public BciSource {
public:
    /// @param stream_name  Nom du stream LSL à résoudre (ex: "OpenBCI_EEG", "obci_eeg1")
    /// @param timeout_s    Timeout de résolution du stream (secondes)
    explicit LslSource(const std::string &stream_name = "OpenBCI_EEG", double timeout_s = 5.0)
        : _streamName(stream_name), _timeout(timeout_s), _inlet(nullptr), _running(false), _sampleIndex(0),
          _samplesSinceLastFFT(0)
    {
        for (size_t ch = 0; ch < BCI_CHANNELS; ++ch)
        {
            _timeDomainBuffers[ch].resize(FFT_SIZE, 0.0f);
            _fftInputs[ch].resize(FFT_SIZE);
        }
    }

    ~LslSource() override { stop(); }

    [[nodiscard]] bool init() override
    {
        try
        {
            printf("[LSL] Resolving stream '%s' (timeout=%.1fs)...\n", _streamName.c_str(), _timeout);
            auto results = lsl::resolve_stream("name", _streamName, 1, _timeout);

            if (results.empty())
            {
                printf("[LSL] No stream '%s' found on network\n", _streamName.c_str());
                return false;
            }

            _inlet = std::make_unique<lsl::stream_inlet>(results[0]);
            _channelCount = static_cast<size_t>(_inlet->info().channel_count());
            _sampleRate = _inlet->info().nominal_srate();

            printf("[LSL] Connected to '%s' (%zu channels, %.0f Hz)\n", _streamName.c_str(), _channelCount,
                   _sampleRate);

            _running = true;
            return true;
        }
        catch (const std::exception &e)
        {
            printf("[LSL] Error: %s\n", e.what());
            return false;
        }
    }

    void update(NeuralState &state) override
    {
        if (!_running || !_inlet)
            return;

        std::vector<float> sample(_channelCount);
        double timestamp;

        // Drain tous les samples disponibles sans blocage
        while ((timestamp = _inlet->pull_sample(sample.data(), static_cast<int>(_channelCount), 0.0)) != 0.0)
        {
            const size_t chCount = std::min(_channelCount, static_cast<size_t>(BCI_CHANNELS));
            for (size_t ch = 0; ch < chCount; ++ch)
            {
                _timeDomainBuffers[ch][_sampleIndex] = sample[ch];
                if (ch == 0)
                    state.blinkDetected = (std::abs(sample[0]) > 150.0f);
            }

            _sampleIndex = (_sampleIndex + 1) % FFT_SIZE;
            _samplesSinceLastFFT++;

            if (_samplesSinceLastFFT >= UPDATE_INTERVAL)
            {
                processFFT(state);
                _samplesSinceLastFFT = 0;
            }
        }
    }

    void stop() override
    {
        _running = false;
        _inlet.reset();
        printf("[LSL] LslSource stopped\n");
    }

    [[nodiscard]] const char *name() const noexcept override { return "LslSource (Lab Streaming Layer)"; }
    [[nodiscard]] BciMode mode() const noexcept override { return BciMode::Lsl; }

private:
    void processFFT(NeuralState &state)
    {
        using Complex = FastFourierTransform::Complex;
        const float normFactor = 2.0f / FFT_SIZE;
        const float freqRes = static_cast<float>(_sampleRate) / FFT_SIZE;

        const uint16_t BIN_40HZ = static_cast<uint16_t>(40.0f * FFT_SIZE / _sampleRate);
        const uint16_t BIN_70HZ = static_cast<uint16_t>(70.0f * FFT_SIZE / _sampleRate);

        const size_t chCount = std::min(_channelCount, static_cast<size_t>(BCI_CHANNELS));
        std::vector<std::vector<float>> psdChannels(chCount, std::vector<float>(FFT_SIZE / 2, 0.0f));

        float alphaSum = 0.0f;
        float betaSum = 0.0f;

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

    std::string _streamName;
    double _timeout;
    std::unique_ptr<lsl::stream_inlet> _inlet;
    size_t _channelCount = 0;
    double _sampleRate = 250.0;
    bool _running;

    std::array<std::vector<float>, BCI_CHANNELS> _timeDomainBuffers;
    std::array<std::vector<FastFourierTransform::Complex>, BCI_CHANNELS> _fftInputs;
    size_t _sampleIndex;
    size_t _samplesSinceLastFFT;
};

#else
// ═══════════════════════════════════════════════════════════════════════════════
//  STUB — LSL non disponible, échoue proprement à init()
// ═══════════════════════════════════════════════════════════════════════════════

class LslSource final : public BciSource {
public:
    explicit LslSource(const std::string & /*stream_name*/ = "", double /*timeout_s*/ = 5.0) {}

    [[nodiscard]] bool init() override
    {
        printf("[LSL] liblsl non disponible — compilez avec -DLPL_USE_LSL et liez -llsl\n");
        return false;
    }

    void update(NeuralState &) override {}
    void stop() override {}

    [[nodiscard]] const char *name() const noexcept override { return "LslSource (STUB — liblsl absent)"; }
    [[nodiscard]] BciMode mode() const noexcept override { return BciMode::Lsl; }
};

#endif // LPL_USE_LSL
