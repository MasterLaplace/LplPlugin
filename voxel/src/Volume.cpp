/**
 * @file Volume.cpp
 * @brief Measuring what a volume's samples actually look like.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Volume.hpp>

namespace lpl::voxel {

namespace {

/// Histogram of non-empty samples. Zero is the mask, not a density, so it never counts.
struct Histogram final {
    core::u64 bins[256]{};
    core::u64 total{0};

    void add(const BrickView &brick) noexcept
    {
        if (!brick.valid())
            return;
        for (core::u32 i = 0u; i < kBrickVoxels; ++i)
        {
            const core::u8 v = brick.voxels[i];
            if (v == 0u)
                continue;
            ++bins[v];
            ++total;
        }
    }

    [[nodiscard]] core::f32 mean() const noexcept
    {
        if (total == 0)
            return 0.0f;
        core::f64 acc = 0.0;
        for (core::u32 v = 1u; v < 256u; ++v)
            acc += static_cast<core::f64>(v) * static_cast<core::f64>(bins[v]);
        return static_cast<core::f32>(acc / static_cast<core::f64>(total));
    }

    [[nodiscard]] core::f32 deviation(core::f32 m) const noexcept
    {
        if (total == 0)
            return 1.0f;
        core::f64 acc = 0.0;
        for (core::u32 v = 1u; v < 256u; ++v)
        {
            const core::f64 d = static_cast<core::f64>(v) - static_cast<core::f64>(m);
            acc += d * d * static_cast<core::f64>(bins[v]);
        }
        const core::f64 var = acc / static_cast<core::f64>(total);
        // A degenerate volume -- every sample identical -- would otherwise divide by zero in
        // every spread ratio downstream. One is the value that makes the correction a no-op.
        core::f64 r = var > 0.0 ? var : 1.0;
        core::f64 g = r;
        for (int i = 0; i < 24; ++i)
            g = 0.5 * (g + r / g); // Newton; no libm, the same discipline as the rest of the tree.
        return static_cast<core::f32>(g);
    }

    /// @return Smallest density such that the fraction of matter at or above it is <= @p fraction.
    [[nodiscard]] core::u8 upperQuantile(core::f32 fraction) const noexcept
    {
        if (total == 0)
            return 255u;
        const core::u64 want = static_cast<core::u64>(static_cast<core::f64>(total) * fraction);
        core::u64 acc = 0;
        for (core::u32 v = 255u; v >= 1u; --v)
        {
            acc += bins[v];
            if (acc >= want)
                return static_cast<core::u8>(v);
        }
        return 1u;
    }

    [[nodiscard]] core::u8 lowerQuantile(core::f32 fraction) const noexcept
    {
        if (total == 0)
            return 0u;
        const core::u64 want = static_cast<core::u64>(static_cast<core::f64>(total) * fraction);
        core::u64 acc = 0;
        for (core::u32 v = 1u; v < 256u; ++v)
        {
            acc += bins[v];
            if (acc >= want)
                return static_cast<core::u8>(v);
        }
        return 255u;
    }
};

} // namespace

DensityProfile measureProfile(const BrickView *bricks, core::u32 count, core::f32 quantile,
                              core::f32 windowDeviations) noexcept
{
    DensityProfile profile{};
    if (bricks == nullptr || count == 0u)
        return profile;

    Histogram h{};
    for (core::u32 i = 0u; i < count; ++i)
        h.add(bricks[i]);
    if (h.total == 0)
        return profile;

    profile.mean = h.mean();
    profile.deviation = h.deviation(profile.mean);

    if (quantile <= 0.0f)
        quantile = 0.133f;
    if (quantile > 1.0f)
        quantile = 1.0f;

    // The window: its upper edge is where the requested fraction of matter starts, and its floor
    // sits a stated number of measured deviations below. A second quantile would let the window
    // invert on a narrow distribution, and an inverted window renders black without saying why;
    // a width in raw sample values would be calibrated on one instrument and wrong on the next.
    profile.sheetSample = h.upperQuantile(quantile);
    if (windowDeviations <= 0.0f)
        windowDeviations = 1.0f;
    core::i32 width = static_cast<core::i32>(profile.deviation * windowDeviations + 0.5f);
    if (width < 2)
        width = 2;
    core::i32 floorSample = static_cast<core::i32>(profile.sheetSample) - width;

    // ⚠ **The floor is never allowed below the bulk, whatever width was asked for.** The
    // distribution of a carbonised roll is narrow -- matter spans about three deviations -- so a
    // window of one deviation can still reach past the mean and start painting the medium. It did:
    // the first real frames had a floor at 143 against a mean of 145, so more than half of every
    // sample painted and the picture saturated within a metre. Whatever else the width says, the
    // medium is what sits near the mean, and the medium must stay out of the way.
    const core::i32 bulk = static_cast<core::i32>(profile.bulkCeiling(0.5f) + 0.5f);
    if (floorSample < bulk)
        floorSample = bulk;
    if (floorSample >= static_cast<core::i32>(profile.sheetSample))
        floorSample = static_cast<core::i32>(profile.sheetSample) - 2;
    profile.floorSample = static_cast<core::u8>(floorSample < 1 ? 1 : floorSample);
    if (profile.sheetSample <= profile.floorSample)
        profile.sheetSample = static_cast<core::u8>(profile.floorSample + 1);

    for (core::u32 lv = 0u; lv < kMaxPyramidLevels; ++lv)
        profile.spreadRatio[lv] = 1.0f;
    return profile;
}

void measureSpread(DensityProfile &profile, const BrickView *perLevel, core::u32 count) noexcept
{
    if (perLevel == nullptr || count == 0u)
        return;

    Histogram base{};
    base.add(perLevel[0]);
    if (base.total == 0)
        return;
    const core::f32 baseDeviation = base.deviation(base.mean());
    if (baseDeviation <= 0.0f)
        return;

    core::f32 previous = 1.0f;
    profile.spreadRatio[0] = 1.0f;
    for (core::u32 lv = 1u; lv < kMaxPyramidLevels; ++lv)
    {
        if (lv >= count || !perLevel[lv].valid())
        {
            // Carrying the previous ratio forward degrades; inventing one would lie.
            profile.spreadRatio[lv] = previous;
            continue;
        }
        Histogram h{};
        h.add(perLevel[lv]);
        if (h.total == 0)
        {
            profile.spreadRatio[lv] = previous;
            continue;
        }
        previous = h.deviation(h.mean()) / baseDeviation;
        profile.spreadRatio[lv] = previous;
    }
}

void measureSpreadOverBox(DensityProfile &profile, const BrickMosaic &mosaic, const core::i64 centre[3],
                          core::i64 halfExtent) noexcept
{
    if (centre == nullptr || halfExtent <= 0)
        return;

    Histogram perLevel[kMaxPyramidLevels]{};
    for (core::u32 i = 0u; i < mosaic.count(); ++i)
    {
        const BrickView &brick = mosaic.at(i);
        if (!brick.valid() || brick.key.level >= kMaxPyramidLevels)
            continue;

        const core::u32 level = brick.key.level;
        const core::i64 step = static_cast<core::i64>(1) << level;
        const core::i64 span = brickSpanInBaseSamples(level);
        const core::i64 origin[3]{static_cast<core::i64>(brick.key.z) * span,
                                  static_cast<core::i64>(brick.key.y) * span,
                                  static_cast<core::i64>(brick.key.x) * span};

        // Only the part of this brick that falls inside the box: that is what makes the
        // comparison between levels a comparison of the SAME region.
        core::i64 lo[3];
        core::i64 hi[3];
        bool overlaps = true;
        for (core::u32 a = 0u; a < 3u; ++a)
        {
            const core::i64 boxLow = centre[a] - halfExtent;
            const core::i64 boxHigh = centre[a] + halfExtent;
            lo[a] = (boxLow - origin[a] + step - 1) / step;
            hi[a] = (boxHigh - origin[a]) / step;
            if (lo[a] < 0)
                lo[a] = 0;
            if (hi[a] > static_cast<core::i64>(kBrickEdge))
                hi[a] = static_cast<core::i64>(kBrickEdge);
            if (lo[a] >= hi[a])
                overlaps = false;
        }
        if (!overlaps)
            continue;

        Histogram &h = perLevel[level];
        for (core::i64 z = lo[0]; z < hi[0]; ++z)
        {
            for (core::i64 y = lo[1]; y < hi[1]; ++y)
            {
                for (core::i64 x = lo[2]; x < hi[2]; ++x)
                {
                    const core::u8 v =
                        brick.at(static_cast<core::u32>(z), static_cast<core::u32>(y), static_cast<core::u32>(x));
                    if (v == 0u)
                        continue;
                    ++h.bins[v];
                    ++h.total;
                }
            }
        }
    }

    // The finest level with anything in the box is the reference; below it there is nothing to
    // compare against, and inventing a reference would be worse than leaving the ratios alone.
    core::u32 base = kMaxPyramidLevels;
    for (core::u32 lv = 0u; lv < kMaxPyramidLevels; ++lv)
    {
        if (perLevel[lv].total > 0)
        {
            base = lv;
            break;
        }
    }
    if (base == kMaxPyramidLevels)
        return;
    const core::f32 reference = perLevel[base].deviation(perLevel[base].mean());
    if (!(reference > 0.0f))
        return;

    core::f32 previous = 1.0f;
    for (core::u32 lv = 0u; lv < kMaxPyramidLevels; ++lv)
    {
        if (lv < base || perLevel[lv].total == 0)
        {
            profile.spreadRatio[lv] = previous;
            continue;
        }
        previous = perLevel[lv].deviation(perLevel[lv].mean()) / reference;
        profile.spreadRatio[lv] = previous;
    }
}

} // namespace lpl::voxel
