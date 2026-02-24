// File: BciSourceFactory.hpp
// Description: Factory pour créer la bonne BciSource selon le mode CLI.
// Parse les arguments --bci-mode et --csv-file depuis argv.
//
// Usage CLI :
//   ./visual                                → Serial (défaut, fallback synthétique)
//   ./visual --bci-mode=synthetic           → Générateur synthétique
//   ./visual --bci-mode=lsl                 → LSL inlet
//   ./visual --bci-mode=csv --csv-file=x.csv → Replay CSV
//   ./visual --bci-mode=serial              → OpenBCI Cyton série
//
// Auteur: MasterLaplace

#pragma once

#include "BciSource.hpp"
#include "BrainFlowSource.hpp"
#include "CsvReplaySource.hpp"
#include "LslSource.hpp"
#include "SerialSource.hpp"
#include "SyntheticSource.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

/// Configuration BCI extraite des arguments CLI.
struct BciConfig {
    BciMode mode = BciMode::Synthetic;       ///< Mode par défaut = synthetic (pas de matériel)
    std::string serialPort = "/dev/ttyUSB0"; ///< Port série pour Serial
    std::string csvFile;                     ///< Fichier CSV pour CsvReplay
    std::string lslStream = "OpenBCI_EEG";   ///< Nom du stream LSL
    uint64_t seed = 0;                       ///< Graine PRNG pour Synthetic (0 = aléatoire)
    float calibDuration = 30.0f;             ///< Durée de calibration (secondes)
    int brainflowBoard = -1;                 ///< BrainFlow board ID (-1 = SYNTHETIC_BOARD)
    std::string brainflowSerial;             ///< BrainFlow serial port/number
};

/// Parse les arguments CLI relatifs au BCI.
///
/// Arguments reconnus :
///   --bci-mode=MODE      serial|synthetic|lsl|csv  (défaut: synthetic)
///   --csv-file=PATH      Fichier CSV pour le mode csv
///   --lsl-stream=NAME    Nom du stream LSL (défaut: OpenBCI_EEG)
///   --serial-port=PATH   Port série (défaut: /dev/ttyUSB0)
///   --bci-seed=N         Graine PRNG pour synthetic
///   --calib-duration=N   Durée de calibration en secondes
///
/// Les arguments non reconnus sont ignorés silencieusement.
inline BciConfig bci_parse_args(int argc, char *argv[])
{
    BciConfig cfg;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);

        auto extractValue = [&](const std::string &prefix) -> std::string {
            if (arg.rfind(prefix, 0) == 0)
                return arg.substr(prefix.size());
            return {};
        };

        if (auto v = extractValue("--bci-mode="); !v.empty())
            cfg.mode = bci_mode_from_string(v);
        else if (auto v2 = extractValue("--csv-file="); !v2.empty())
            cfg.csvFile = v2;
        else if (auto v3 = extractValue("--lsl-stream="); !v3.empty())
            cfg.lslStream = v3;
        else if (auto v4 = extractValue("--serial-port="); !v4.empty())
            cfg.serialPort = v4;
        else if (auto v5 = extractValue("--bci-seed="); !v5.empty())
            cfg.seed = std::stoull(v5);
        else if (auto v6 = extractValue("--calib-duration="); !v6.empty())
            cfg.calibDuration = std::stof(v6);
        else if (auto v7 = extractValue("--brainflow-board="); !v7.empty())
            cfg.brainflowBoard = std::stoi(v7);
        else if (auto v8 = extractValue("--brainflow-serial="); !v8.empty())
            cfg.brainflowSerial = v8;
    }

    return cfg;
}

/// Factory qui instancie la bonne BciSource selon la config.
namespace BciSourceFactory {

/// Crée une BciSource selon le mode demandé.
/// En cas d'échec du mode Serial, tente un fallback vers Synthetic.
[[nodiscard]] inline std::unique_ptr<BciSource> create(const BciConfig &cfg)
{
    std::unique_ptr<BciSource> source;

    switch (cfg.mode)
    {
    case BciMode::Serial: source = std::make_unique<SerialSource>(cfg.serialPort.c_str()); break;

    case BciMode::Synthetic: source = std::make_unique<SyntheticSource>(cfg.seed, true); break;

    case BciMode::Lsl: source = std::make_unique<LslSource>(cfg.lslStream); break;

    case BciMode::CsvReplay:
        if (cfg.csvFile.empty())
        {
            printf("[BCI] CSV mode requires --csv-file=PATH\n");
            printf("[BCI] Falling back to Synthetic\n");
            source = std::make_unique<SyntheticSource>(cfg.seed, true);
        }
        else
        {
            source = std::make_unique<CsvReplaySource>(cfg.csvFile);
        }
        break;

    case BciMode::BrainFlow:
        source = std::make_unique<BrainFlowSource>(cfg.brainflowBoard, cfg.brainflowSerial);
        break;
    }

    return source;
}

/// Crée et initialise la source. Fallback vers Synthetic si init échoue.
[[nodiscard]] inline std::unique_ptr<BciSource> createAndInit(const BciConfig &cfg)
{
    auto source = create(cfg);

    printf("[BCI] Trying %s...\n", source->name());

    if (source->init())
    {
        printf("[BCI] %s initialized successfully\n", source->name());
        return source;
    }

    // Fallback vers Synthetic si le mode demandé échoue
    if (cfg.mode != BciMode::Synthetic)
    {
        printf("[BCI] %s failed — falling back to Synthetic\n", source->name());
        source = std::make_unique<SyntheticSource>(cfg.seed, true);
        if (source->init())
        {
            printf("[BCI] SyntheticSource (fallback) initialized\n");
            return source;
        }
    }

    printf("[BCI] All sources failed\n");
    return nullptr;
}

} // namespace BciSourceFactory
