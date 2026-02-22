// File: SerialSource.hpp
// Description: Adaptateur BciSource pour le driver OpenBCI Cyton série.
// Encapsule l'OpenBCIDriver existant derrière l'interface abstraite BciSource.
// Auteur: MasterLaplace

#pragma once

#include "BciSource.hpp"
#include "OpenBCIDriver.hpp"
#include <cstring>

/// Source BCI utilisant le matériel OpenBCI Cyton via port série.
///
/// Wraps l'OpenBCIDriver existant sans modification.
/// Constructeur : SerialSource("/dev/ttyUSB0")
class SerialSource final : public BciSource {
public:
    explicit SerialSource(const char *port = "/dev/ttyUSB0") { std::strncpy(_port, port, sizeof(_port) - 1); }

    [[nodiscard]] bool init() override { return _driver.init(_port); }

    void update(NeuralState &state) override { _driver.update(state); }

    void stop() override { _driver.stop(); }

    [[nodiscard]] const char *name() const noexcept override { return "SerialSource (OpenBCI Cyton)"; }

    [[nodiscard]] BciMode mode() const noexcept override { return BciMode::Serial; }

private:
    OpenBCIDriver _driver;
    char _port[64] = {};
};
