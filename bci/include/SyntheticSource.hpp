// File: SyntheticSource.hpp
// Description: Source BCI synthétique — génère de l'EEG simulé multi-bandes,
// applique la FFT existante et produit un NeuralState complet sans matériel.
// Fonctionne en temps réel (250 Hz simulé) ou en burst pour les tests.
//
// Pipeline : SyntheticGenerator → FFT (Cooley-Tukey) → NeuralState
//            identique au chemin réel OpenBCI, seul l'input change.
//
// Auteur: MasterLaplace

#pragma once

#include "BciSource.hpp"
#include "SignalMetrics.hpp"
#include "sim/SyntheticGenerator.hpp"
#include <chrono>

/// Source BCI basée sur le générateur synthétique.
///
/// Simule l'acquisition EEG en temps réel : à chaque appel de `update()`,
/// le nombre de samples écoulés depuis le dernier appel est calculé d'après
/// le temps réel, puis injecté dans la FFT pour produire le NeuralState.
class SyntheticSource final : public BciSource {
public:
    /// @param seed         Graine PRNG (0 = aléatoire)
    /// @param realtime     Si true, respecte le timing 250 Hz. Sinon, génère
    ///                     UPDATE_INTERVAL samples à chaque update() (mode test).
    explicit SyntheticSource(uint64_t seed = 0, bool realtime = true)
        : _gen(seed), _realtime(realtime), _sampleIndex(0), _samplesSinceLastFFT(0), _running(false)
    {
        for (size_t ch = 0; ch < SYNTH_CHANNELS; ++ch)
        {
            _timeDomainBuffers[ch].resize(FFT_SIZE, 0.0f);
            _fftInputs[ch].resize(FFT_SIZE);
        }
    }

    [[nodiscard]] bool init() override
    {
        _running = true;
        _lastUpdate = std::chrono::steady_clock::now();
        printf("[BCI] SyntheticSource initialized (seed=%lu, realtime=%s)\n", _gen.sampleIndex(),
               _realtime ? "true" : "false");
        return true;
    }

    void update(NeuralState &state) override
    {
        if (!_running)
            return;

        // Calculer combien de samples générer
        size_t samplesToGenerate;
        if (_realtime)
        {
            auto now = std::chrono::steady_clock::now();
            const float elapsed_s = std::chrono::duration<float>(now - _lastUpdate).count();
            _lastUpdate = now;
            samplesToGenerate = static_cast<size_t>(elapsed_s * SYNTH_SAMPLE_RATE);
            if (samplesToGenerate == 0)
                samplesToGenerate = 1; // Au moins 1 sample par update
            if (samplesToGenerate > MAX_SAMPLES_PER_UPDATE)
                samplesToGenerate = MAX_SAMPLES_PER_UPDATE; // Limiter les rattrapages
        }
        else
        {
            samplesToGenerate = UPDATE_INTERVAL; // Burst mode pour tests
        }

        // Générer les échantillons synthétiques
        auto samples = _gen.generate(samplesToGenerate);

        // Injecter dans les buffers circulaires et déclencher la FFT
        for (size_t t = 0; t < samplesToGenerate; ++t)
        {
            for (size_t ch = 0; ch < SYNTH_CHANNELS; ++ch)
            {
                _timeDomainBuffers[ch][_sampleIndex] = samples[t][ch];

                // Détection de clignement sur canal 0 (frontal)
                if (ch == 0)
                    state.blinkDetected = (std::abs(samples[t][0]) > 150.0f);
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
        printf("[BCI] SyntheticSource stopped\n");
    }

    [[nodiscard]] const char *name() const noexcept override { return "SyntheticSource (EEG simulé)"; }

    [[nodiscard]] BciMode mode() const noexcept override { return BciMode::Synthetic; }

    /// Accès au générateur pour modifier le profil en cours d'exécution.
    SyntheticGenerator &generator() noexcept { return _gen; }

private:
    void processFFT(NeuralState &state)
    {
        using Complex = FastFourierTransform::Complex;
        const float normFactor = 2.0f / FFT_SIZE;

        static constexpr uint16_t BIN_40HZ = static_cast<uint16_t>(40.0f * FFT_SIZE / SYNTH_SAMPLE_RATE);
        static constexpr uint16_t BIN_70HZ = static_cast<uint16_t>(70.0f * FFT_SIZE / SYNTH_SAMPLE_RATE);

        std::vector<std::vector<float>> psdChannels(SYNTH_CHANNELS, std::vector<float>(FFT_SIZE / 2, 0.0f));

        float alphaSum = 0.0f;
        float betaSum = 0.0f;

        for (size_t ch = 0; ch < SYNTH_CHANNELS; ++ch)
        {
            // Copier le buffer circulaire dans l'entrée FFT
            for (size_t i = 0; i < FFT_SIZE; ++i)
            {
                size_t idx = (_sampleIndex + i) % FFT_SIZE;
                _fftInputs[ch][i] = Complex(_timeDomainBuffers[ch][idx], 0.0f);
            }

            FastFourierTransform::apply_window(_fftInputs[ch]);
            FastFourierTransform::compute(_fftInputs[ch]);

            float chAlpha = 0.0f;
            float chBeta = 0.0f;

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

        // R(t) de Schumacher
        state.schumacherR = SignalMetrics::schumacher(psdChannels, BIN_40HZ, BIN_70HZ);

        // Lissage exponentiel (identique à OpenBCIDriver)
        const float smoothFactor = 0.1f;
        state.alphaPower = state.alphaPower * (1.0f - smoothFactor) + (alphaSum / SYNTH_CHANNELS) * smoothFactor;
        state.betaPower = state.betaPower * (1.0f - smoothFactor) + (betaSum / SYNTH_CHANNELS) * smoothFactor;

        const float totalPower = state.alphaPower + state.betaPower + 0.0001f;
        state.concentration = state.concentration * 0.9f + (state.betaPower / totalPower) * 0.1f;
    }

    static constexpr size_t FFT_SIZE = 256u;
    static constexpr size_t UPDATE_INTERVAL = 32u;
    static constexpr float FREQ_RES = SYNTH_SAMPLE_RATE / FFT_SIZE;
    static constexpr size_t MAX_SAMPLES_PER_UPDATE = 512u; // Clamp si lag

    SyntheticGenerator _gen;
    bool _realtime;

    std::array<std::vector<float>, SYNTH_CHANNELS> _timeDomainBuffers;
    std::array<std::vector<FastFourierTransform::Complex>, SYNTH_CHANNELS> _fftInputs;
    size_t _sampleIndex;
    size_t _samplesSinceLastFFT;
    bool _running;

    std::chrono::steady_clock::time_point _lastUpdate;
};
