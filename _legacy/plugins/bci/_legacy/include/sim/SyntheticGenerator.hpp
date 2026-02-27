// File: SyntheticGenerator.hpp
// Description: Générateur d'EEG synthétique multi-bandes pour simulation sans matériel.
// Génère 8 canaux à 250 Hz avec des bandes alpha (8-12 Hz), beta (13-30 Hz),
// gamma/EMG (40-70 Hz), ainsi que du bruit rose et des artéfacts de clignement.
// Déterministe pour les tests (graine fixable), stochastique pour la simulation live.
//
// Références :
//   - OpenBCI Cyton : 8 canaux, 250 Hz, résolution 24 bits → ~0.02 µV/LSB
//   - Bandes EEG standard : delta(0.5-4), theta(4-8), alpha(8-12), beta(13-30), gamma(30+)
//   - Artéfact de clignement EOG : impulsion ~150-300 µV sur canaux frontaux
//
// Auteur: MasterLaplace

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

static constexpr size_t SYNTH_CHANNELS = 8u;
static constexpr float SYNTH_SAMPLE_RATE = 250.0f;

/// Configuration d'un oscillateur sinusoïdal pour une bande de fréquence.
struct BandOscillator {
    float freq_hz;      ///< Fréquence centrale (Hz)
    float amplitude_uV; ///< Amplitude crête (µV)
    float phase_offset; ///< Déphasage initial (rad)
};

/// Profil de génération synthétique — configure les amplitudes par bande.
struct SyntheticProfile {
    /// Oscillateurs par bande (alpha, beta, gamma/EMG)
    std::vector<BandOscillator> oscillators = {
        {10.0f, 15.0f, 0.0f}, // Alpha 10 Hz — repos yeux fermés
        {20.0f, 8.0f,  0.3f}, // Beta 20 Hz — activité cognitive
        {50.0f, 3.0f,  0.7f}, // Gamma/EMG 50 Hz — tension musculaire
    };

    float noise_amplitude_uV = 2.0f;   ///< Bruit blanc gaussien (µV RMS)
    float blink_probability = 0.005f;  ///< Probabilité de clignement par sample (~1.25/sec à 250 Hz)
    float blink_amplitude_uV = 200.0f; ///< Amplitude du clignement (µV)
    float blink_duration_s = 0.15f;    ///< Durée du clignement (sec)

    /// Inter-channel phase spread (rad) — simule la propagation corticale
    float channel_phase_spread = 0.2f;
};

/// Générateur d'EEG synthétique déterministe.
///
/// Usage :
/// @code
///   SyntheticGenerator gen(42);           // Graine fixe pour tests
///   auto samples = gen.generate(256);     // 256 échantillons × 8 canaux
///   // samples[t][ch] = valeur en µV
/// @endcode
class SyntheticGenerator {
public:
    /// @param seed  Graine PRNG (0 = aléatoire basée sur le temps)
    explicit SyntheticGenerator(uint64_t seed = 0) : _profile{}, _sampleIndex(0), _blinkRemaining(0)
    {
        if (seed == 0)
            seed = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        _rng.seed(seed);
        _noiseDist = std::normal_distribution<float>(0.0f, _profile.noise_amplitude_uV);
        _blinkDist = std::uniform_real_distribution<float>(0.0f, 1.0f);
    }

    /// Modifie le profil de génération (amplitudes, fréquences, bruit).
    void setProfile(const SyntheticProfile &profile)
    {
        _profile = profile;
        _noiseDist = std::normal_distribution<float>(0.0f, _profile.noise_amplitude_uV);
    }

    [[nodiscard]] const SyntheticProfile &profile() const noexcept { return _profile; }

    /// Génère `count` échantillons multi-canaux.
    /// @return Vecteur de [count][SYNTH_CHANNELS] valeurs en µV.
    [[nodiscard]] std::vector<std::array<float, SYNTH_CHANNELS>> generate(size_t count)
    {
        std::vector<std::array<float, SYNTH_CHANNELS>> output(count);

        for (size_t t = 0; t < count; ++t)
        {
            const float time_s = static_cast<float>(_sampleIndex) / SYNTH_SAMPLE_RATE;

            // Vérifier / déclencher un clignement
            if (_blinkRemaining <= 0)
            {
                if (_blinkDist(_rng) < _profile.blink_probability)
                    _blinkRemaining = static_cast<int>(_profile.blink_duration_s * SYNTH_SAMPLE_RATE);
            }

            // Enveloppe de clignement (Hann) — max sur canaux frontaux (0, 1)
            float blinkEnvelope = 0.0f;
            if (_blinkRemaining > 0)
            {
                const float blinkSamples = _profile.blink_duration_s * SYNTH_SAMPLE_RATE;
                const float progress = 1.0f - static_cast<float>(_blinkRemaining) / blinkSamples;
                blinkEnvelope =
                    _profile.blink_amplitude_uV * 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> * progress));
                _blinkRemaining--;
            }

            for (size_t ch = 0; ch < SYNTH_CHANNELS; ++ch)
            {
                float sample = 0.0f;

                // Somme des oscillateurs de bande avec déphasage inter-canal
                for (const auto &osc : _profile.oscillators)
                {
                    const float phase = 2.0f * std::numbers::pi_v<float> * osc.freq_hz * time_s + osc.phase_offset +
                                        static_cast<float>(ch) * _profile.channel_phase_spread;
                    sample += osc.amplitude_uV * std::sin(phase);
                }

                // Bruit gaussien
                sample += _noiseDist(_rng);

                // Artéfact de clignement — atténué sur les canaux postérieurs
                const float blinkWeight = (ch < 2u) ? 1.0f : (ch < 4u) ? 0.3f : 0.05f;
                sample += blinkEnvelope * blinkWeight;

                output[t][ch] = sample;
            }

            _sampleIndex++;
        }

        return output;
    }

    /// Reset le générateur à l'état initial avec une nouvelle graine.
    void reset(uint64_t seed = 0)
    {
        if (seed == 0)
            seed = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        _rng.seed(seed);
        _sampleIndex = 0;
        _blinkRemaining = 0;
    }

    [[nodiscard]] uint64_t sampleIndex() const noexcept { return _sampleIndex; }

private:
    SyntheticProfile _profile;
    std::mt19937_64 _rng;
    std::normal_distribution<float> _noiseDist;
    std::uniform_real_distribution<float> _blinkDist;
    uint64_t _sampleIndex;
    int _blinkRemaining;
};
