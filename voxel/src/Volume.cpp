/**
 * @file Volume.cpp
 * @brief Measuring what a volume's samples actually look like.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/std/cmath.hpp>
#include <lpl/voxel/Volume.hpp>

namespace lpl::voxel {

namespace {

/// Histogram of non-empty samples. Zero is the mask, not a density, so it never counts.
struct Histogram final {
    core::u64 bins[256]{};
    core::u64 total{0};

    void addSample(core::u8 v) noexcept
    {
        if (v == 0u)
            return;
        ++bins[v];
        ++total;
    }

    void add(const BrickView &brick) noexcept
    {
        if (!brick.valid())
            return;
        for (core::u32 i = 0u; i < kBrickVoxels; ++i)
            addSample(brick.voxels[i]);
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
        return static_cast<core::f32>(pmr::sqrt(var > 0.0 ? var : 1.0));
    }

    /// @return Largest density such that at least @p fraction of the matter is at or above it.
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

    // The floor is never allowed below the bulk, whatever width was asked for. The distribution of
    // a carbonised roll is narrow -- matter spans about three deviations -- so a window of one
    // deviation can still reach past the mean and start painting the medium. It did: the first
    // real frames had a floor at 143 against a mean of 145, so more than half of every sample
    // painted and the picture saturated within a metre. Whatever else the width says, the medium is
    // what sits near the mean, and the medium must stay out of the way.
    const core::i32 bulk = static_cast<core::i32>(profile.bulkCeiling(0.5f) + 0.5f);
    if (floorSample < bulk)
        floorSample = bulk;
    if (floorSample >= static_cast<core::i32>(profile.sheetSample))
        floorSample = static_cast<core::i32>(profile.sheetSample) - 2;
    profile.floorSample = static_cast<core::u8>(floorSample < 1 ? 1 : floorSample);
    if (profile.sheetSample <= profile.floorSample)
        profile.sheetSample = static_cast<core::u8>(profile.floorSample + 1);
    return profile;
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

        // Only the part of this brick that falls inside the box: that is what makes the
        // comparison between levels a comparison of the SAME region.
        core::i64 lo[3];
        core::i64 hi[3];
        bool overlaps = true;
        for (core::u32 a = 0u; a < 3u; ++a)
        {
            const core::i64 origin = brickOriginInBaseSamples(brick.key, a);
            const core::i64 boxLow = centre[a] - halfExtent;
            const core::i64 boxHigh = centre[a] + halfExtent;
            lo[a] = (boxLow - origin + step - 1) / step;
            hi[a] = (boxHigh - origin) / step;
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
                    h.addSample(
                        brick.at(static_cast<core::u32>(z), static_cast<core::u32>(y), static_cast<core::u32>(x)));
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
