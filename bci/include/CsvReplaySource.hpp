// File: CsvReplaySource.hpp
// Description: Source BCI basée sur le replay d'un fichier CSV d'enregistrement EEG.
// Format CSV attendu : une ligne par sample, colonnes = canaux (float µV).
// Supporte le mode loopback (redémarrage en fin de fichier) et le respect
// du timing réel (250 Hz) ou le burst pour les tests.
//
// Compatible avec les exports OpenBCI GUI, BrainFlow, et tout outil
// générant du CSV multi-canal.
//
// Format exemple (8 canaux, pas d'en-tête) :
//   12.34,5.67,8.90,1.23,4.56,7.89,0.12,3.45
//   ...
//
// Auteur: MasterLaplace

#pragma once

#include "BciSource.hpp"
#include "SignalMetrics.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class CsvReplaySource final : public BciSource {
public:
    /// @param csvPath   Chemin vers le fichier CSV
    /// @param loopback  Redémarrer au début en fin de fichier
    /// @param realtime  Respecter le timing 250 Hz (false = burst mode)
    explicit CsvReplaySource(const std::string &csvPath, bool loopback = true, bool realtime = true)
        : _csvPath(csvPath), _loopback(loopback), _realtime(realtime), _cursor(0), _sampleIndex(0),
          _samplesSinceLastFFT(0), _running(false)
    {
        for (size_t ch = 0; ch < BCI_CHANNELS; ++ch)
        {
            _timeDomainBuffers[ch].resize(FFT_SIZE, 0.0f);
            _fftInputs[ch].resize(FFT_SIZE);
        }
    }

    [[nodiscard]] bool init() override
    {
        if (!loadCsv())
            return false;

        _running = true;
        _lastUpdate = std::chrono::steady_clock::now();
        printf("[CSV] Loaded %zu samples × %zu channels from '%s' (loopback=%s)\n", _data.size(),
               _data.empty() ? 0 : _data[0].size(), _csvPath.c_str(), _loopback ? "on" : "off");
        return true;
    }

    void update(NeuralState &state) override
    {
        if (!_running || _data.empty())
            return;

        size_t samplesToProcess;
        if (_realtime)
        {
            auto now = std::chrono::steady_clock::now();
            const float elapsed_s = std::chrono::duration<float>(now - _lastUpdate).count();
            _lastUpdate = now;
            samplesToProcess = static_cast<size_t>(elapsed_s * SAMPLE_RATE);
            if (samplesToProcess == 0)
                samplesToProcess = 1;
            if (samplesToProcess > MAX_SAMPLES_PER_UPDATE)
                samplesToProcess = MAX_SAMPLES_PER_UPDATE;
        }
        else
        {
            samplesToProcess = UPDATE_INTERVAL;
        }

        for (size_t t = 0; t < samplesToProcess; ++t)
        {
            if (_cursor >= _data.size())
            {
                if (_loopback)
                    _cursor = 0;
                else
                {
                    _running = false;
                    printf("[CSV] End of file reached — replay stopped\n");
                    return;
                }
            }

            const auto &row = _data[_cursor++];
            const size_t chCount = std::min(row.size(), static_cast<size_t>(BCI_CHANNELS));

            for (size_t ch = 0; ch < chCount; ++ch)
            {
                _timeDomainBuffers[ch][_sampleIndex] = row[ch];
                if (ch == 0)
                    state.blinkDetected = (std::abs(row[0]) > 150.0f);
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
        printf("[CSV] CsvReplaySource stopped (cursor=%zu/%zu)\n", _cursor, _data.size());
    }

    [[nodiscard]] const char *name() const noexcept override { return "CsvReplaySource"; }
    [[nodiscard]] BciMode mode() const noexcept override { return BciMode::CsvReplay; }

    /// Nombre total de samples dans le fichier.
    [[nodiscard]] size_t totalSamples() const noexcept { return _data.size(); }

    /// Position actuelle du curseur de lecture.
    [[nodiscard]] size_t cursor() const noexcept { return _cursor; }

private:
    bool loadCsv()
    {
        std::ifstream file(_csvPath);
        if (!file.is_open())
        {
            printf("[CSV] Cannot open '%s'\n", _csvPath.c_str());
            return false;
        }

        _data.clear();
        std::string line;
        while (std::getline(file, line))
        {
            // Ignorer les lignes vides et les commentaires
            if (line.empty() || line[0] == '#' || line[0] == '%')
                continue;

            std::vector<float> row;
            std::istringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                try
                {
                    row.push_back(std::stof(token));
                }
                catch (...)
                { /* Ignorer les valeurs non parsables */
                }
            }

            if (!row.empty())
                _data.push_back(std::move(row));
        }

        return !_data.empty();
    }

    void processFFT(NeuralState &state)
    {
        using Complex = FastFourierTransform::Complex;
        const float normFactor = 2.0f / FFT_SIZE;

        static constexpr uint16_t BIN_40HZ = static_cast<uint16_t>(40.0f * FFT_SIZE / SAMPLE_RATE);
        static constexpr uint16_t BIN_70HZ = static_cast<uint16_t>(70.0f * FFT_SIZE / SAMPLE_RATE);

        std::vector<std::vector<float>> psdChannels(BCI_CHANNELS, std::vector<float>(FFT_SIZE / 2, 0.0f));
        float alphaSum = 0.0f, betaSum = 0.0f;

        for (size_t ch = 0; ch < BCI_CHANNELS; ++ch)
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
                float freq = static_cast<float>(i) * FREQ_RES;
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
        state.alphaPower = state.alphaPower * (1.0f - sf) + (alphaSum / BCI_CHANNELS) * sf;
        state.betaPower = state.betaPower * (1.0f - sf) + (betaSum / BCI_CHANNELS) * sf;

        const float totalPower = state.alphaPower + state.betaPower + 0.0001f;
        state.concentration = state.concentration * 0.9f + (state.betaPower / totalPower) * 0.1f;
    }

    static constexpr size_t FFT_SIZE = 256u;
    static constexpr size_t UPDATE_INTERVAL = 32u;
    static constexpr float SAMPLE_RATE = 250.0f;
    static constexpr float FREQ_RES = SAMPLE_RATE / FFT_SIZE;
    static constexpr size_t MAX_SAMPLES_PER_UPDATE = 512u;

    std::string _csvPath;
    bool _loopback;
    bool _realtime;
    std::vector<std::vector<float>> _data;
    size_t _cursor;

    std::array<std::vector<float>, BCI_CHANNELS> _timeDomainBuffers;
    std::array<std::vector<FastFourierTransform::Complex>, BCI_CHANNELS> _fftInputs;
    size_t _sampleIndex;
    size_t _samplesSinceLastFFT;
    bool _running;

    std::chrono::steady_clock::time_point _lastUpdate;
};
