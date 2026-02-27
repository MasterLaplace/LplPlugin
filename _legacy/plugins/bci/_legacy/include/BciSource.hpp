// File: BciSource.hpp
// Description: Interface abstraite pour toute source de données BCI.
// Permet de substituer le matériel réel par une simulation synthétique,
// un flux LSL, ou un replay CSV sans modifier le reste de la pipeline.
// Auteur: MasterLaplace

#pragma once

#include "OpenBCIDriver.hpp"
#include <memory>
#include <string>

/// Mode d'acquisition BCI — sélectionnable au runtime via CLI.
enum class BciMode {
    Serial,     ///< Matériel réel (OpenBCI Cyton via /dev/ttyUSBx)
    Synthetic,  ///< Générateur synthétique multi-bandes (aucun matériel)
    Lsl,        ///< Flux LSL entrant (Lab Streaming Layer)
    CsvReplay,  ///< Replay d'un enregistrement CSV
    BrainFlow   ///< BrainFlow SDK — multi-board acquisition (OpenBCI, Muse, etc.)
};

/// Convertit un nom CLI ("serial", "synthetic", "lsl", "csv") en BciMode.
[[nodiscard]] inline BciMode bci_mode_from_string(const std::string &s) noexcept
{
    if (s == "synthetic" || s == "synth")
        return BciMode::Synthetic;
    if (s == "lsl")
        return BciMode::Lsl;
    if (s == "csv" || s == "replay")
        return BciMode::CsvReplay;
    if (s == "brainflow" || s == "bf")
        return BciMode::BrainFlow;
    return BciMode::Serial;
}

/// Retourne le nom lisible d'un BciMode.
[[nodiscard]] inline const char *bci_mode_name(BciMode m) noexcept
{
    switch (m)
    {
    case BciMode::Serial: return "Serial (OpenBCI)";
    case BciMode::Synthetic: return "Synthetic";
    case BciMode::Lsl: return "LSL Inlet";
    case BciMode::CsvReplay: return "CSV Replay";
    case BciMode::BrainFlow: return "BrainFlow";
    }
    return "Unknown";
}

/// Interface abstraite pour une source de données BCI.
///
/// Contrat :
///  1. `init()` retourne true si la source est prête.
///  2. `update(state)` remplit le NeuralState à chaque tick.
///  3. `stop()` libère les ressources (thread, fd, socket…).
///  4. `name()` retourne un identifiant lisible pour les logs.
///
/// Exemple d'usage :
/// @code
///   auto src = BciSourceFactory::create(BciMode::Synthetic, {});
///   if (src->init()) {
///       NeuralState ns;
///       src->update(ns);
///       auto metrics = NeuralMetrics::from_state(ns, baseline);
///   }
///   src->stop();
/// @endcode
class BciSource {
public:
    virtual ~BciSource() = default;

    /// Initialise la source. Retourne true en cas de succès.
    [[nodiscard]] virtual bool init() = 0;

    /// Met à jour le NeuralState avec les dernières données disponibles.
    virtual void update(NeuralState &state) = 0;

    /// Arrête proprement la source (thread, fd, socket…).
    virtual void stop() = 0;

    /// Nom lisible pour les logs.
    [[nodiscard]] virtual const char *name() const noexcept = 0;

    /// Le mode BCI de cette source.
    [[nodiscard]] virtual BciMode mode() const noexcept = 0;
};
