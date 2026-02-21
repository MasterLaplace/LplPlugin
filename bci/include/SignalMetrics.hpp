// File: SignalMetrics.hpp
// Description: Métriques spectrales pour signaux EEG/EMG (Schumacher, RMS, baseline)
// Référence: Schumacher et al., 2015 — "Closed-loop BCI for muscle fatigue"
// Auteur: MasterLaplace

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace SignalMetrics {

/// Somme les valeurs de psd[lower_b..upper_b] (bornes incluses).
/// Correspond à ∫[lower_b, upper_b] PSD(f) df en domaine discret.
/// @param lower_b  Indice de bin inférieur (inclus)
/// @param upper_b  Indice de bin supérieur (inclus)
/// @param psd      Spectre de densité de puissance (magnitudes FFT)
[[nodiscard]] inline float integrale(uint16_t lower_b, uint16_t upper_b, const std::vector<float> &psd) noexcept
{
    if (psd.empty() || lower_b >= static_cast<uint16_t>(psd.size()))
        return 0.f;
    uint16_t end = std::min(upper_b, static_cast<uint16_t>(psd.size() - 1u));
    float area = 0.f;
    for (uint16_t i = lower_b; i <= end; ++i)
        area += psd[i];
    return area;
}

/// Convertit une fréquence Hz en indice de bin FFT.
/// bin = floor(hz * fft_size / sample_rate)
[[nodiscard]] inline uint16_t hz_to_bin(float hz, float sample_rate, size_t fft_size) noexcept
{
    return static_cast<uint16_t>(hz * static_cast<float>(fft_size) / sample_rate);
}

/// Métrique de Schumacher R(t) : puissance moyenne dans la bande [lower_b, upper_b]
/// sur l'ensemble des canaux.
/// R(t) = (1/N_ch) * Σ_i ∫[lower_b, upper_b] PSD_i(f) df
/// Utilisé comme indicateur de tension musculaire sur la bande 40-70 Hz.
/// @param channels  Vecteur de PSD par canal (chaque canal = vecteur de bins)
/// @param lower_b   Bin inférieur (défaut = 40, ~40 Hz à fs=250/N=256)
/// @param upper_b   Bin supérieur (défaut = 71, ~70 Hz à fs=250/N=256)
[[nodiscard]] inline float schumacher(const std::vector<std::vector<float>> &channels, uint16_t lower_b = 40u,
                                      uint16_t upper_b = 71u) noexcept
{
    if (channels.empty())
        return 0.f;
    float score = 0.f;
    for (const auto &ch : channels)
        score += integrale(lower_b, upper_b, ch);
    return score / static_cast<float>(channels.size());
}

/// RMS calculé sur les `window_size` derniers échantillons du vecteur.
/// Utile pour détecter les artéfacts EMG ou estimer la puissance instantanée.
[[nodiscard]] inline float sliding_window_rms(const std::vector<float> &data, size_t window_size) noexcept
{
    if (data.size() < window_size || window_size == 0u)
        return 0.f;
    float sum_sq = 0.f;
    const size_t start = data.size() - window_size;
    for (size_t i = start; i < data.size(); ++i)
        sum_sq += data[i] * data[i];
    return std::sqrt(sum_sq / static_cast<float>(window_size));
}

/// Résultat de calibration (ligne de base).
struct Baseline {
    float mean = 0.f;
    float std_dev = 0.f;
};

/// Calcule la moyenne et l'écart-type populationnel d'un vecteur.
/// Sert à établir la ligne de base pour normaliser R(t).
[[nodiscard]] inline Baseline compute_baseline(const std::vector<float> &data) noexcept
{
    if (data.empty())
        return {};
    const float mean = std::accumulate(data.begin(), data.end(), 0.f) / static_cast<float>(data.size());
    float var = 0.f;
    for (float v : data)
        var += (v - mean) * (v - mean);
    return {mean, std::sqrt(var / static_cast<float>(data.size()))};
}

} // namespace SignalMetrics
