/**
 * @file Transfer.cpp
 * @brief The ramp, and the rescaling that keeps it meaning the same thing at every level.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Transfer.hpp>

namespace lpl::voxel {

namespace {

constexpr core::f32 clamp01(core::f32 v) noexcept { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

void fill(TransferFunction &tf, core::f32 lowEdge, core::f32 highEdge, core::f32 peakAlpha) noexcept
{
    if (highEdge <= lowEdge)
        highEdge = lowEdge + 1.0f;
    tf.lowEdge = lowEdge;
    tf.highEdge = highEdge;
    tf.peakAlpha = peakAlpha;

    tf.firstVisible = 255u;
    for (core::u32 v = 0u; v < 256u; ++v)
    {
        const core::f32 t = clamp01((static_cast<core::f32>(v) - lowEdge) / (highEdge - lowEdge));
        // Smoothstep rather than linear: the edge of a sheet is where the scan is least certain,
        // and a linear ramp puts its steepest change exactly there. This puts the steep part in
        // the middle of the band, where the samples are least ambiguous.
        const core::f32 s = t * t * (3.0f - 2.0f * t);

        tf.alpha[v] = s * peakAlpha;
        // Cold and dim in the medium, warm and bright on a sheet. Two ends of one hue rather
        // than a rainbow: a colour scale that runs through hues invites the reader to see
        // categories the data does not have.
        tf.red[v] = 0.32f + 0.68f * s;
        tf.green[v] = 0.30f + 0.52f * s;
        tf.blue[v] = 0.34f + 0.20f * s;

        if (tf.firstVisible == 255u && tf.alpha[v] > 1e-4f)
            tf.firstVisible = static_cast<core::u8>(v);
    }
}

} // namespace

TransferFunction rampTransfer(core::u8 floorSample, core::u8 sheetSample, core::f32 peakAlpha) noexcept
{
    TransferFunction tf{};
    fill(tf, static_cast<core::f32>(floorSample), static_cast<core::f32>(sheetSample), peakAlpha);
    return tf;
}

core::f32 transparencyAfter(core::f32 perSample, core::f32 samples) noexcept
{
    if (perSample <= 0.0f || samples <= 0.0f)
        return 1.0f;
    if (perSample >= 1.0f)
        return 0.0f;

    // Repeated squaring on the whole samples. An exp/log pair would be one line and would put a
    // transcendental on the hot path of every sample, which this tree does not spend.
    const core::f32 keep = 1.0f - perSample;
    core::f32 left = 1.0f;
    core::f32 power = keep;
    for (core::u32 whole = static_cast<core::u32>(samples); whole != 0u; whole >>= 1u)
    {
        if ((whole & 1u) != 0u)
            left *= power;
        power *= power;
    }
    const core::f32 fraction = samples - static_cast<core::f32>(static_cast<core::u32>(samples));
    return left * (1.0f - fraction * (1.0f - keep));
}

core::f32 alphaForOpaqueAfter(core::f32 samples, core::f32 opacity) noexcept
{
    if (!(samples > 0.0f))
        return 1.0f;
    if (opacity <= 0.0f)
        return 0.0f;
    if (opacity >= 1.0f)
        return 1.0f;

    // Solve 1 - (1-a)^n = opacity for a, by bisection. A log/exp pair would be shorter and would
    // put a transcendental in a module that is otherwise free of them; this runs once, at setup.
    core::f32 lo = 0.0f;
    core::f32 hi = 1.0f;
    for (int i = 0; i < 40; ++i)
    {
        const core::f32 mid = 0.5f * (lo + hi);
        if (1.0f - transparencyAfter(mid, samples) < opacity)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5f * (lo + hi);
}

TransferFunction TransferFunction::forLevel(const DensityProfile &profile, core::u32 level) const noexcept
{
    if (level == 0u || level >= kMaxPyramidLevels)
        return *this;

    const core::f32 ratio = profile.spreadRatio[level];
    if (!(ratio > 0.0f) || ratio >= 1.0f)
        return *this;

    // Squeeze the window toward the mean by the measured ratio: the samples have moved toward the
    // mean, so the window has to move with them to keep the same matter visible. The edges are the
    // stored ones.
    const core::f32 m = profile.mean;
    TransferFunction scaled{};
    fill(scaled, m + (lowEdge - m) * ratio, m + (highEdge - m) * ratio, peakAlpha);
    return scaled;
}

} // namespace lpl::voxel
