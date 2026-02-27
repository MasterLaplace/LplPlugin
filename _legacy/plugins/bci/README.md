# LPL BCI Plugin V2

Brain-Computer Interface plugin for the LplPlugin engine.  
C++23 · Eigen3 · Boost · liblsl · Catch2  

## Architecture

```
plugins/bci/
├── include/
│   ├── core/           # Types, Constants, Error, Concepts
│   ├── dsp/            # IStage, Pipeline, Windowing, FFT, BandExtractor, RingBuffer
│   ├── source/         # ISource, OpenBCI, Synthetic, LSL, BrainFlow, CSV, Factory
│   │   ├── serial/     # Cross-platform SerialPort (POSIX + Win32)
│   │   └── sim/        # SyntheticGenerator
│   ├── math/           # Statistics, Covariance, Riemannian geometry (SPD manifold)
│   ├── metric/         # SignalMetric (Schumacher), NeuralMetric, StabilityMetric
│   ├── calibration/    # State machine (Idle → Calibrating → Ready)
│   ├── stream/         # LslOutlet (LSL broadcast)
│   └── openvibe/       # MuscleRelaxationBox, StabilityMonitorBox
├── src/        # Implementation files (.cpp)
├── tests/              # Catch2 unit tests
├── _legacy/            # V1 archived code (read-only reference)
└── xmake.lua           # Build configuration
```

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **SRP** | Sources only acquire data; DSP stages only transform; Metrics only compute |
| **OCP** | `ISource` and `IStage` interfaces allow extension without modification |
| **LSP** | All `ISource` implementations are interchangeable via `SourceFactory` |
| **ISP** | Small focused interfaces (`ISource`, `IStage`) |
| **DIP** | Pipeline depends on `IStage` abstraction, not concrete implementations |

## Error Handling

All fallible operations return `Expected<T>` (`std::expected<T, Error>`).  
Errors carry an `ErrorCode`, a human-readable message, and `source_location`.

```cpp
auto result = source->read(buffer);
if (!result)
    log("{}", result.error().format());
```

## Key Improvements over V1

- **Single FFT implementation** — V1 had `processFFT()` duplicated 5 times
- **No conditional compilation** — all dependencies are mandatory (except optional BrainFlow)
- **Proper namespace** — `bci` with sub-namespaces
- **Centralized constants** — `Constants.hpp` replaces 5 copies of `BCI_CHANNELS`
- **RAII everywhere** — `std::jthread`, `std::unique_ptr`, PIMPL for serial ports
- **Fixed bugs** — V1's `StabilityMonitor::_distanceHistory` was never populated
- **Riemannian geometry** — Eigen-only, no manual Jacobi fallback

## Dependencies

| Library | Purpose | Required |
|---------|---------|----------|
| Eigen3 | Linear algebra, SPD geometry | Yes |
| Boost | `lockfree::spsc_queue` for RingBuffer | Yes |
| liblsl | Lab Streaming Layer I/O | Yes |
| BrainFlow | Multi-board acquisition SDK | Optional |
| Catch2 | Unit testing | Dev only |

## Build

> **Note:** the project uses a custom xmake package repository. You can
> either register the local copy shipped in the workspace or let xmake fetch
> from GitHub. The local mirror contains the liblsl/brainflow recipes and is
> updated via `git pull`.
>
> ```bash
> # register workspace copy (preferred for offline/CI environments)
> xmake repo -a /home/masterlaplace/LplPlugin/xmake-repo
> # or, clone from GitHub manually if you renamed the fork:
> # git clone https://github.com/MasterLaplace/xmake-local-repo.git \
> #     /home/masterlaplace/LplPlugin/xmake-local-repo
> ```

```bash
xmake config --mode=debug
xmake build lpl-bci
xmake build lpl-bci-tests
xmake run lpl-bci-tests
```

Enable BrainFlow support:
```bash
xmake config --with_brainflow=y
```

The unit-test suite now covers all public modules including *serial*,
*OpenBciSource*, *LslSource* and conditional *BrainFlowSource* backends.

> **LSL tests** configure a 0.1‑second resolve timeout and gracefully succeed
> if no sample arrives; the library may spin up network discovery ports while
> running which can produce verbose logs.


## Module Overview

### `core/` — Vocabulary Types
`Sample`, `SignalBlock`, `FrequencyBand`, `BandPower`, `NeuralState`, `Baseline`, `AcquisitionMode`

### `dsp/` — Digital Signal Processing
Fluent `PipelineBuilder` composes stages: `HannWindow → FftProcessor → BandExtractor`

### `source/` — Data Acquisition
`ISource` implementations: `OpenBciSource` (serial), `SyntheticSource`, `LslSource`, `BrainFlowSource`, `CsvReplaySource`

### `math/` — Mathematical Foundations
- `Statistics`: PSD integration, Hz→bin, sliding RMS, baseline
- `Covariance`: Batch + Welford online estimation, Ledoit-Wolf regularization
- `Riemannian`: `matrixSqrt`, `matrixLog`, geodesic distance, Fréchet mean on SPD(n)

### `metric/` — BCI Metrics
- `SignalMetric`: Schumacher β/(α+θ) ratio
- `NeuralMetric`: Baseline-normalized state in [0,1]
- `StabilityMetric`: Temporal consistency via Riemannian distances

### `calibration/` — Baseline Acquisition
Three-state machine with Observer pattern for state-change notifications.

### `stream/` — Network Output
`LslOutlet` broadcasts `NeuralState` or raw samples over LSL.

### `openvibe/` — OpenViBE Integration
`MuscleRelaxationBox` (gamma artifact detection) and `StabilityMonitorBox` (EMA-smoothed Riemannian stability).

## References

- Schumacher et al. (2015) — Muscular relaxation in MI-BCI
- Sollfrank et al. (2016) — Multimodal enriched feedback for SMR-BCI
- Barachant et al. (2012) — Multiclass BCI by Riemannian geometry
- Congedo et al. (2017) — Riemannian geometry for EEG-based BCI

## License

See [LICENSE](../../LICENSE) for details.
