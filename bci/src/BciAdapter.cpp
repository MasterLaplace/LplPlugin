/**
 * @file BciAdapter.cpp
 * @brief BciAdapter implementation with full DSP pipeline.
 *
 * Pipeline: poll() → RingBuffer → Hann Window → FFT → BandExtractor →
 *           NeuralMetric → NeuralSafetyChain → NeuralInputState
 *
 * Ported from plugins/bci/ DSP pipeline into the engine adapter pattern.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/bci/BciAdapter.hpp>
#include <lpl/bci/NeuralSafetyChain.hpp>
#include <lpl/bci/calibration/Calibration.hpp>
#include <lpl/bci/metric/NeuralMetric.hpp>
#include <lpl/bci/dsp/Pipeline.hpp>
#include <lpl/bci/dsp/Windowing.hpp>
#include <lpl/bci/dsp/FftProcessor.hpp>
#include <lpl/bci/dsp/BandExtractor.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <cmath>
#include <optional>
#include <vector>

namespace lpl::bci {

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct BciAdapter::Impl
{
    std::unique_ptr<IBciDriver>    driver;
    BciAdapterConfig               config{};
    NeuralSafetyChain              safetyChain;
    calibration::Calibration       calibration;
    metric::NeuralMetric           neuralMetric;
    core::u32                      tickSequence{0};
    bool                           calibrated{false};

    // DSP pipeline state
    std::optional<dsp::Pipeline>    pipeline;
    std::vector<std::vector<float>> sampleBuffer;  // Per-channel accumulator
    core::u32                       samplesAccumulated{0};

    // Per-window band powers (for calibration feeding)
    std::vector<float>              lastAlphaPowers;
    std::vector<float>              lastBetaPowers;
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

BciAdapter::BciAdapter(std::unique_ptr<IBciDriver> driver,
                       const BciAdapterConfig& config)
    : _impl{std::make_unique<Impl>()}
{
    LPL_ASSERT(driver != nullptr);
    _impl->driver = std::move(driver);
    _impl->config = config;

    // Initialize safety chain with default checks
    _impl->safetyChain.addCheck(std::make_unique<AmplitudeBoundsCheck>(1000.0f));
    _impl->safetyChain.addCheck(std::make_unique<ConfidenceCheck>(
        config.confidenceThreshold));
    _impl->safetyChain.addCheck(std::make_unique<RateOfChangeCheck>(500.0f));
}

BciAdapter::~BciAdapter() { stop(); }

core::Expected<void> BciAdapter::start()
{
    auto res = _impl->driver->connect();
    if (!res) { return res; }

    // Pre-allocate sample buffers for FFT window
    const core::u32 channels = input::NeuralInputState::kChannels;
    _impl->sampleBuffer.resize(channels);
    for (auto& ch : _impl->sampleBuffer)
    {
        ch.reserve(_impl->config.fftSize);
    }
    _impl->samplesAccumulated = 0;

    std::vector<FrequencyBand> bands = {
        FrequencyBand{8.0f, 13.0f}, // Alpha
        FrequencyBand{13.0f, 30.0f} // Beta
    };

    _impl->pipeline = dsp::Pipeline::builder()
        .add<dsp::HannWindow>(_impl->config.fftSize)
        .add<dsp::FftProcessor>(_impl->config.fftSize)
        .add<dsp::BandExtractor>(bands, _impl->config.sampleRateHz, _impl->config.fftSize)
        .build();

    core::Log::info("BciAdapter: DSP pipeline initialized");
    return _impl->driver->startStream();
}

void BciAdapter::stop()
{
    if (_impl && _impl->driver)
    {
        _impl->driver->stopStream();
        _impl->driver->disconnect();
    }
}

core::Expected<input::NeuralInputState> BciAdapter::update()
{
    // Poll raw samples from driver
    auto sampleResult = _impl->driver->poll();
    if (!sampleResult)
    {
        return core::Unexpected(sampleResult.error());
    }

    const auto& raw = sampleResult.value();
    const core::usize channelCount = std::min(
        static_cast<core::usize>(raw.channelCount),
        static_cast<core::usize>(input::NeuralInputState::kChannels));

    // Accumulate samples for FFT window
    for (core::usize ch = 0; ch < channelCount; ++ch)
    {
        _impl->sampleBuffer[ch].push_back(raw.channels[ch]);
    }
    _impl->samplesAccumulated++;

    // Check if we have enough samples for a full FFT window
    input::NeuralInputState state{};
    state.sequence = ++_impl->tickSequence;

    if (_impl->samplesAccumulated >= _impl->config.fftSize)
    {
        // ── DSP Pipeline ─────────────────────────────────────────────
        SignalBlock inputBlock;
        inputBlock.data = _impl->sampleBuffer;
        inputBlock.sampleRate = _impl->config.sampleRateHz;
        inputBlock.channelCount = channelCount;
        inputBlock.timestamp = std::chrono::steady_clock::now();

        auto result = _impl->pipeline->process(inputBlock);
        if (!result)
        {
            core::Log::error("BciAdapter: Pipeline failed: {}", result.error().message);
            for (auto& ch : _impl->sampleBuffer) ch.clear();
            _impl->samplesAccumulated = 0;
            return core::makeError(core::ErrorCode::InvalidState, result.error().message);
        }

        const auto& bandPowers = result.value();
        float totalAlpha = 0.0f;
        float totalBeta  = 0.0f;

        for (core::usize ch = 0; ch < channelCount; ++ch)
        {
            float alphaPower = bandPowers.data[0][ch];
            float betaPower  = bandPowers.data[1][ch];

            totalAlpha += alphaPower;
            totalBeta  += betaPower;

            float totalPower = alphaPower + betaPower;
            float activation = (totalPower > 0.001f) ? betaPower / totalPower : 0.0f;
            state.channels[ch] = math::Fixed32::fromFloat(
                std::clamp(activation, 0.0f, 1.0f));
        }

        float totalPower = totalAlpha + totalBeta;
        float confidence = (totalPower > 0.001f)
            ? std::clamp(totalBeta / totalPower, 0.0f, 1.0f)
            : 0.0f;
        state.confidence = math::Fixed32::fromFloat(confidence);

        // ── Calibration feeding ──────────────────────────────────────
        // Store per-channel band powers for calibration
        _impl->lastAlphaPowers.resize(channelCount);
        _impl->lastBetaPowers.resize(channelCount);

        _impl->lastAlphaPowers.assign(bandPowers.data[0].begin(), bandPowers.data[0].begin() + channelCount);
        _impl->lastBetaPowers.assign(bandPowers.data[1].begin(), bandPowers.data[1].begin() + channelCount);

        // Feed calibration if active
        if (_impl->calibration.state() == calibration::CalibrationState::kCalibrating)
        {
            auto trial = _impl->calibration.addTrial(
                _impl->lastAlphaPowers, _impl->lastBetaPowers);
            (void)trial;

            // Check if calibration just completed
            if (_impl->calibration.state() == calibration::CalibrationState::kReady)
            {
                auto baselines = _impl->calibration.baselines();
                if (baselines.has_value())
                {
                    _impl->neuralMetric.setBaselines(baselines.value());
                    _impl->calibrated = true;
                    core::Log::info("BciAdapter: calibration complete, NeuralMetric active");
                }
            }
        }

        // ── NeuralMetric normalization (if calibrated) ───────────────
        if (_impl->calibrated)
        {
            auto neuralState = _impl->neuralMetric.compute(
                _impl->lastAlphaPowers, _impl->lastBetaPowers);

            // Map normalized alpha/beta to engine channels (alpha dominant)
            for (core::usize ch = 0; ch < channelCount && ch < neuralState.channelAlpha.size(); ++ch)
            {
                // Use alpha as primary activation channel
                state.channels[ch] = math::Fixed32::fromFloat(neuralState.channelAlpha[ch]);
            }
            state.confidence = math::Fixed32::fromFloat(neuralState.concentration);
        }

        // Reset sample buffer for next window
        for (auto& ch : _impl->sampleBuffer)
        {
            ch.clear();
        }
        _impl->samplesAccumulated = 0;
    }
    else
    {
        // Not enough samples yet — return raw conversion
        for (core::usize i = 0; i < channelCount; ++i)
        {
            state.channels[i] = math::Fixed32::fromFloat(
                std::clamp(raw.channels[i], 0.0f, 1.0f));
        }
        state.confidence = math::Fixed32{0};
    }

    // ── Safety Chain validation ───────────────────────────────────────
    SafetyVerdict verdict = _impl->safetyChain.evaluate(state);
    state.validated = (verdict != SafetyVerdict::Reject);

    if (!state.validated)
    {
        // Safety chain rejected — zero out the state for safety
        for (auto& ch : state.channels)
        {
            ch = math::Fixed32{0};
        }
        state.confidence = math::Fixed32{0};
    }

    return state;
}

core::Expected<void> BciAdapter::startCalibration()
{
    auto result = _impl->calibration.start();
    if (!result)
    {
        return core::makeError(core::ErrorCode::InvalidState,
                               "Failed to start BCI calibration");
    }
    _impl->calibrated = false;
    core::Log::info("BciAdapter: calibration started");
    return {};
}

bool BciAdapter::isCalibrated() const noexcept
{
    return _impl->calibrated;
}

const IBciDriver& BciAdapter::driver() const noexcept
{
    return *_impl->driver;
}

} // namespace lpl::bci
