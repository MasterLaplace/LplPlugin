# OpenViBE Integration

This directory contains **standalone processors** and **OpenViBE Box Algorithm skeletons** for the two metrics required by the SEAMLESS team's enriched BCI feedback paradigm.

## Architecture

Each file provides:
1. **A standalone C++ processor class** — fully functional, no OpenViBE dependency. Can be used in any C++ application, including the LplPlugin visual client.
2. **A commented OpenViBE CBoxAlgorithm template** — ready to uncomment and compile against the OpenViBE SDK to become a native box in the OpenViBE Designer.

## Box Algorithms

### CBoxAlgorithmMuscleRelaxation
- **Paper**: Schumacher et al. (2015)
- **Metric**: Average spectral power in the 40-70 Hz (Gamma) band
- **Input**: Multi-channel EEG signal
- **Output**: Scalar R(t) — muscular artifact indicator
- **Core function**: `SignalMetrics::schumacher()`

### CBoxAlgorithmStabilityMonitor
- **Paper**: Sollfrank et al. (2016)
- **Metric**: Riemannian distance between consecutive covariance matrices
- **Input**: Multi-channel EEG signal
- **Output**: Scalar stability ∈ [0, 1]
- **Core function**: `RiemannianGeometry::riemannian_distance()` (or Eigen-backed variant)

## Usage Without OpenViBE

```cpp
#include "CBoxAlgorithmMuscleRelaxation.hpp"
#include "CBoxAlgorithmStabilityMonitor.hpp"

// Muscle relaxation
LplOpenViBE::MuscleRelaxationProcessor muscleProc({.sampleRate = 250.0f});
muscleProc.configure(8);
float Rt = muscleProc.compute(psdChannels);
bool alert = muscleProc.isAlert(Rt);

// Signal stability
LplOpenViBE::StabilityMonitorProcessor stabilityProc;
stabilityProc.configure(8);
float stability = stabilityProc.update(channelData);
```

## Compiling with OpenViBE SDK

1. Install the [OpenViBE SDK](http://openvibe.inria.fr/downloads/)
2. Uncomment the `CBoxAlgorithm...` classes in the `.hpp` files
3. Add the OpenViBE include paths and link against `openvibe-toolkit`
4. Register the boxes via the OpenViBE plugin descriptor mechanism
