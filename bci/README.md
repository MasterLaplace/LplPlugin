# bci — Plugin BCI OpenBCI Cyton

Interface temps-réel avec le casque **OpenBCI Cyton** (8 canaux, 250 Hz) et calcul de métriques neurales pour la boucle fermée BCI.

## Contenu

```
bci/
├── include/
│   ├── OpenBCIDriver.hpp     — Driver USB-série, ring buffer, FFT multi-canal
│   ├── SignalMetrics.hpp     — Métriques spectrales (Schumacher R(t), RMS, baseline)
│   ├── RiemannianGeometry.hpp— Géométrie SPD, δ_R, distance de Mahalanobis
│   └── NeuralMetrics.hpp     — Struct normalisée pour la boucle de contrôle
├── calcul.c                  — Référence C de Schumacher/intégrale
└── tests/
    ├── test_metrics.cpp      — Tests unitaires SignalMetrics
    └── test_riemannian.cpp   — Tests unitaires RiemannianGeometry
```

## Métriques implémentées

### Schumacher R(t) — Tension musculaire

$$
R(t) = \frac{1}{N_{ch}} \sum_{i=1}^{N_{ch}} \int_{40}^{70} \mathrm{PSD}_i(f,t)\, df
$$

Indicateur de contamination EMG/artefacts haute-fréquence du signal EEG.  
Référence : Schumacher et al., *Closed-loop control of gait using BCI*, 2015.

### Distance Riemannienne δ_R — Stabilité cognitive

$$
\delta_R(C_1, C_2) = \sqrt{\sum_i \ln^2(\lambda_i)}
$$

où $\lambda_i$ sont les valeurs propres de $C_1^{-1/2} C_2 C_1^{-1/2}$.  
Invariante par congruence — robuste aux artéfacts de volume-conduit.  
Références : Moakher 2005, Arsigny et al. 2006, Blankertz et al. 2011.

### Distance de Mahalanobis D_M — Détection d'anomalie

$$
D_M(x_t) = \sqrt{(x_t - \mu_c)^T \Sigma_c^{-1} (x_t - \mu_c)}
$$

Détecte les points anormaux dans l'espace des caractéristiques EEG par rapport à un état de référence calibré.

## Build & Tests

```bash
make -C bci test   # Compile et exécute les 24 tests unitaires
```

**Résultat attendu :**
```
[OK] Tous les tests sont passes.   (SignalMetrics — 12 tests)
[OK] Tous les tests sont passes.   (RiemannianGeometry — 12 tests)
```

## Format de paquet OpenBCI Cyton

Le Cyton émet des paquets de 33 octets à 250 Hz :

| Octets | Contenu                        |
|--------|-------------------------------|
| `[0]`  | `0xA0` — marqueur de début    |
| `[1]`  | Compteur d'échantillons        |
| `[2..4]` | Canal 1 (24 bits, signé)   |
| `[5..7]` | Canal 2                    |
| …      | …                             |
| `[23..25]` | Canal 8                |
| `[26..31]` | Accéléromètre (AX,AY,AZ)|
| `[32]` | `0xC0` — marqueur de fin      |

`parse_channel(data)` récupère une valeur en microvolts via le facteur d'échelle `4.5 / 24 / 8388607 × 10⁶ µV/LSB`.

## Usage (exemple)

```cpp
#include "OpenBCIDriver.hpp"
#include "NeuralMetrics.hpp"

OpenBCIDriver bci;
bci.init("/dev/ttyUSB0");

NeuralState  ns;
float        baseline_R = 0.0f;

// Phase de calibration (2 secondes au repos)
// ...

while (running) {
    bci.update(ns);
    auto metrics = NeuralMetrics::from_state(ns, baseline_R);
    if (metrics.muscle_alert)
        pause_feedback();
}
```
