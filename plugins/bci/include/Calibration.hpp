// File: Calibration.hpp
// Description: Machine à états de calibration automatique pour BCI.
// Collecte les valeurs de R(t) au repos pendant une durée configurable,
// puis calcule la baseline (moyenne + écart-type) via SignalMetrics::compute_baseline().
//
// États : Idle → Calibrating → Ready
//
// Pipeline d'usage :
//   1. Démarrer la calibration avec start()
//   2. À chaque tick, appeler tick(neuralState) pour accumuler R(t)
//   3. Quand state() == CalibState::Ready, baseline() est exploitable
//   4. Passer baseline().mean comme baseline_R à NeuralMetrics::from_state()
//
// Auteur: MasterLaplace

#pragma once

#include "NeuralMetrics.hpp"
#include "SignalMetrics.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

/// États de la machine de calibration.
enum class CalibState {
    Idle,        ///< Pas encore démarrée
    Calibrating, ///< Acquisition en cours — accumuler R(t)
    Ready        ///< Baseline calculée — prêt pour la boucle fermée
};

/// Retourne un nom lisible pour le state de calibration.
[[nodiscard]] inline const char *calib_state_name(CalibState s) noexcept
{
    switch (s)
    {
    case CalibState::Idle: return "Idle";
    case CalibState::Calibrating: return "Calibrating";
    case CalibState::Ready: return "Ready";
    }
    return "Unknown";
}

/// Machine à états de calibration BCI.
///
/// Accumule les valeurs de Schumacher R(t) pendant `duration_s` secondes,
/// puis calcule la baseline via SignalMetrics::compute_baseline().
///
/// Exemple :
/// @code
///   Calibration calib(30.0f);   // 30 secondes de calibration
///   calib.start();
///
///   // Dans la boucle principale :
///   bciSource->update(neuralState);
///   calib.tick(neuralState);
///
///   if (calib.state() == CalibState::Ready) {
///       auto metrics = NeuralMetrics::from_state(neuralState, calib.baseline().mean);
///   }
/// @endcode
class Calibration {
public:
    /// @param duration_s  Durée de calibration en secondes (défaut: 30s)
    /// @param min_samples Nombre minimum de samples avant d'accepter la baseline
    explicit Calibration(float duration_s = 30.0f, size_t min_samples = 50)
        : _duration(duration_s), _minSamples(min_samples), _state(CalibState::Idle)
    {
    }

    /// Démarre la calibration. Reset les données si déjà calibré.
    void start()
    {
        _rValues.clear();
        _rValues.reserve(static_cast<size_t>(_duration * 250.0f / 32.0f)); // ~estimation
        _baseline = {};
        _state = CalibState::Calibrating;
        _startTime = std::chrono::steady_clock::now();
        printf("[CALIB] Calibration started (%.0fs, min %zu samples)\n", _duration, _minSamples);
    }

    /// Accumule une mesure de R(t) si en phase de calibration.
    /// Transite automatiquement vers Ready quand la durée est écoulée.
    void tick(const NeuralState &state)
    {
        if (_state != CalibState::Calibrating)
            return;

        // Accumuler R(t) (une valeur par FFT update, pas par sample brut)
        if (state.schumacherR > 0.0f || !_rValues.empty())
            _rValues.push_back(state.schumacherR);

        // Vérifier la durée
        auto now = std::chrono::steady_clock::now();
        const float elapsed = std::chrono::duration<float>(now - _startTime).count();

        if (elapsed >= _duration && _rValues.size() >= _minSamples)
        {
            _baseline = SignalMetrics::compute_baseline(_rValues);
            _state = CalibState::Ready;
            printf("[CALIB] Calibration complete: baseline_R = %.4f ± %.4f (%zu samples)\n", _baseline.mean,
                   _baseline.std_dev, _rValues.size());
        }
    }

    /// Force la fin de la calibration immédiatement (avec les samples collectés).
    void forceComplete()
    {
        if (_state != CalibState::Calibrating)
            return;

        if (_rValues.size() >= _minSamples)
        {
            _baseline = SignalMetrics::compute_baseline(_rValues);
            _state = CalibState::Ready;
            printf("[CALIB] Forced complete: baseline_R = %.4f ± %.4f (%zu samples)\n", _baseline.mean,
                   _baseline.std_dev, _rValues.size());
        }
        else
        {
            printf("[CALIB] Cannot force complete: only %zu/%zu samples\n", _rValues.size(), _minSamples);
        }
    }

    /// Reset complet vers Idle.
    void reset()
    {
        _rValues.clear();
        _baseline = {};
        _state = CalibState::Idle;
    }

    // ─── Accesseurs ───────────────────────────────────────────────────────────

    [[nodiscard]] CalibState state() const noexcept { return _state; }

    /// Baseline calculée (valide uniquement si state == Ready).
    [[nodiscard]] const SignalMetrics::Baseline &baseline() const noexcept { return _baseline; }

    /// Progression de la calibration [0.0, 1.0].
    [[nodiscard]] float progress() const noexcept
    {
        if (_state != CalibState::Calibrating)
            return (_state == CalibState::Ready) ? 1.0f : 0.0f;

        auto now = std::chrono::steady_clock::now();
        const float elapsed = std::chrono::duration<float>(now - _startTime).count();
        return std::min(1.0f, elapsed / _duration);
    }

    /// Nombre de samples R(t) accumulés.
    [[nodiscard]] size_t sampleCount() const noexcept { return _rValues.size(); }

    /// Construit un NeuralMetrics en utilisant la baseline calibrée.
    /// Si pas encore calibré, retourne les metrics non normalisées.
    [[nodiscard]] NeuralMetrics computeMetrics(const NeuralState &state) const noexcept
    {
        const float baselineR = (_state == CalibState::Ready) ? _baseline.mean : 0.0f;
        return NeuralMetrics::from_state(state, baselineR);
    }

private:
    float _duration;
    size_t _minSamples;
    CalibState _state;
    std::chrono::steady_clock::time_point _startTime;
    std::vector<float> _rValues;
    SignalMetrics::Baseline _baseline;
};
