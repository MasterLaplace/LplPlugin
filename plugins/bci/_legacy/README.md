# bci — OpenBCI Cyton BCI Plugin

Real-time interface with the **OpenBCI Cyton** headset (8 channels, 250 Hz) and neural metrics computation for closed-loop BCI control.

## Contents

```
bci/
├── include/
│   ├── OpenBCIDriver.hpp     — USB-serial driver, ring buffer, multi-channel FFT
│   ├── SignalMetrics.hpp     — Spectral metrics (Schumacher R(t), RMS, baseline)
│   ├── RiemannianGeometry.hpp— SPD geometry, δ_R, Mahalanobis distance
│   └── NeuralMetrics.hpp     — Normalized struct for control loop
├── calcul.c                  — C reference for Schumacher/integral
└── tests/
    ├── test_metrics.cpp      — Unit tests SignalMetrics
    └── test_riemannian.cpp   — Unit tests RiemannianGeometry
```

## Implemented Metrics

### Schumacher R(t) — Muscle Tension

$$
R(t) = \frac{1}{N_{ch}} \sum_{i=1}^{N_{ch}} \int_{40}^{70} \mathrm{PSD}_i(f,t)\, df
$$

Indicator of EMG contamination/high-frequency artifacts in EEG signal.  
Reference: Schumacher et al., *Closed-loop control of gait using BCI*, 2015.

### Riemannian Distance δ_R — Cognitive Stability

$$
\delta_R(C_1, C_2) = \sqrt{\sum_i \ln^2(\lambda_i)}
$$

where $\lambda_i$ are eigenvalues of $C_1^{-1/2} C_2 C_1^{-1/2}$.  
Congruence-invariant — robust to volume-conduction artifacts.  
References: Moakher 2005, Arsigny et al. 2006, Blankertz et al. 2011.

### Mahalanobis Distance D_M — Anomaly Detection

$$
D_M(x_t) = \sqrt{(x_t - \mu_c)^T \Sigma_c^{-1} (x_t - \mu_c)}
$$

Detects outlier points in EEG feature space relative to a calibrated reference state.

## Build & Tests

```bash
make -C bci test   # Compile and run 24 unit tests
```

**Expected output:**
```
[OK] All tests passed.   (SignalMetrics — 12 tests)
[OK] All tests passed.   (RiemannianGeometry — 12 tests)
```

## OpenBCI Cyton Packet Format

The Cyton emits 33-byte packets at 250 Hz:

| Bytes | Content                        |
|--------|-------------------------------|
| `[0]`  | `0xA0` — start marker         |
| `[1]`  | Sample counter                |
| `[2..4]` | Channel 1 (24-bit, signed)  |
| `[5..7]` | Channel 2                   |
| …      | …                             |
| `[23..25]` | Channel 8               |
| `[26..31]` | Accelerometer (AX,AY,AZ)|
| `[32]` | `0xC0` — end marker           |

`parse_channel(data)` retrieves a value in microvolts via scale factor `4.5 / 24 / 8388607 × 10⁶ µV/LSB`.

## Usage (example)

```cpp
#include "OpenBCIDriver.hpp"
#include "NeuralMetrics.hpp"

OpenBCIDriver bci;
bci.init("/dev/ttyUSB0");

NeuralState  ns;
float        baseline_R = 0.0f;

// Calibration phase (2 seconds at rest)
// ...

while (running) {
    bci.update(ns);
    auto metrics = NeuralMetrics::from_state(ns, baseline_R);
    if (metrics.muscle_alert)
        pause_feedback();
}
```

