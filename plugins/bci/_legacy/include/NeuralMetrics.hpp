// File: NeuralMetrics.hpp
// Description: Métriques neurales normalisées pour la boucle fermée BCI
// Combine le NeuralState brut (post-FFT) avec une baseline de calibration
// pour produire des indicateurs exploitables par l'application.
// Auteur: MasterLaplace

#pragma once

#include "OpenBCIDriver.hpp"
#include "SignalMetrics.hpp"
#include <algorithm>
#include <cmath>

/// Métriques BCI interprétées, normalisées [0,1] ou booléennes.
/// À construire via NeuralMetrics::from_state() après chaque appel à
/// OpenBCIDriver::update().
struct NeuralMetrics {
    float muscle_tension = 0.0f; ///< R(t) normalisé par baseline — tension musculaire [0,1]
    float stability = 0.0f;      ///< Stabilité du signal vs baseline [0,1]
    float concentration = 0.0f;  ///< Ratio beta/(alpha+beta) — charge cognitive [0,1]
    float signal_quality = 0.0f; ///< Qualité du signal estimée (présence de puissance) [0,1]
    bool muscle_alert = false;   ///< Vrai si R(t) dépasse le seuil (artefact EMG probable)

    /// Seuil de déclenchement de l'alerte musculaire (µV).
    static constexpr float MUSCLE_ALERT_THRESHOLD = 0.5f;
    /// Puissance minimale pour considérer le signal valide.
    static constexpr float MIN_SIGNAL_POWER = 0.01f;

    /// Construit un NeuralMetrics depuis un NeuralState brut.
    ///
    /// @param state       État neural courant (sortie d'OpenBCIDriver::update)
    /// @param baseline_R  Valeur de référence R(t) au repos (0 = non calibré)
    ///
    /// Exemple d'usage :
    /// @code
    ///   bci.update(neural_state);
    ///   auto metrics = NeuralMetrics::from_state(neural_state, baseline_R);
    ///   if (metrics.muscle_alert) pause_training();
    /// @endcode
    [[nodiscard]] static NeuralMetrics from_state(const NeuralState &state, float baseline_R = 0.0f) noexcept
    {
        NeuralMetrics m;

        // Tension musculaire normalisée par la baseline de repos
        if (baseline_R > 1e-6f)
            m.muscle_tension = std::min(1.0f, state.schumacherR / baseline_R);
        else
            m.muscle_tension = state.schumacherR; // Non normalisé si pas de calibration

        // Stabilité : 1 - deviation relative depuis la baseline
        const float deviation = std::abs(state.schumacherR - baseline_R) / (baseline_R + 1e-6f);
        m.stability = std::max(0.0f, 1.0f - std::min(1.0f, deviation));

        m.concentration = state.concentration;
        m.signal_quality = (state.alphaPower + state.betaPower > MIN_SIGNAL_POWER) ? 1.0f : 0.0f;
        m.muscle_alert = (state.schumacherR > MUSCLE_ALERT_THRESHOLD);

        return m;
    }
};
