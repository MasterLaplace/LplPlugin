#include <lpl/codec/Prng.hpp>

#include <lpl/math/FixedMath.hpp>

namespace lpl::codec {

namespace {

constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief The label that turns a droplet seed into its own stream.
 *
 * Part of the wire format: it decides which source blocks a seed names. Changing it
 * changes what every seed means, and therefore what every strand already written
 * decodes to.
 */
constexpr core::u32 kDropletSalt = 0xD0B1E7u;

/**
 * @brief ln(2) in Q16.16, the bridge from the base-two logarithm this project has to
 *        the natural one the distribution is written with.
 *
 * round(0.6931472 * 65536).
 */
constexpr core::i32 kLn2Raw = 45426;

/**
 * @brief Natural logarithm of a positive integer, in Q16.16, without libm.
 *
 * ln(x) = log2(x) * ln(2). @ref lpl::math::fixedLog2 is piecewise linear between
 * powers of two, which is coarser than a series expansion and is the right trade
 * here: R only sets the scale of the robust component, and the distribution is
 * normalised afterwards, so an error of a fraction of a percent in R moves no
 * decision. What matters is that both targets make the SAME error.
 *
 * @param value Argument; 0 and 1 both yield 0.
 * @return ln(value) in Q16.16.
 */
[[nodiscard]] math::Fixed32 fixedLn(core::u32 value) noexcept
{
    if (value <= 1u)
        return math::Fixed32::zero();
    return math::fixedLog2(value) * math::Fixed32::fromRaw(kLn2Raw);
}

/**
 * @brief One over @p value, as a Q32 weight.
 *
 * Q32 rather than Q16.16 for the reason SolitonTable documents: the ideal soliton's
 * tail is 1/(d(d-1)) and reaches the Q16.16 quantum at d = 256, so a Q16.16 table
 * would silently truncate the high degrees to zero.
 *
 * @param value Denominator; 0 yields 0.
 * @return round(2^32 / value).
 */
[[nodiscard]] core::u64 inverseQ32(core::u64 value) noexcept
{
    if (value == 0u)
        return 0u;
    return (core::u64{1} << 32) / value;
}

} // namespace

void SolitonTable::build(const SolitonParams &params)
{
    _sourceBlocks = params.sourceBlocks;
    _cumulative.clear();
    _total = 0u;
    _spikeDegree = 0u;
    _robustScale = math::Fixed32::zero();

    if (_sourceBlocks == 0u)
        return;

    const core::u32 k = _sourceBlocks;

    const core::u32 deltaRaw = params.delta.raw() > 0 ? static_cast<core::u32>(params.delta.raw()) : 1u;
    const core::u64 ratio = (static_cast<core::u64>(k) << 16) / deltaRaw;
    const core::u32 boundedRatio = ratio > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<core::u32>(ratio);

    _robustScale =
        params.c * fixedLn(boundedRatio) * math::fixedSqrt(math::Fixed32::fromInt(static_cast<core::i32>(k)));

    core::u32 spike = k;
    if (_robustScale.raw() > 0)
    {
        const core::u64 scaled = (static_cast<core::u64>(k) << 16) / static_cast<core::u64>(_robustScale.raw());
        spike = scaled == 0u ? 1u : (scaled > k ? k : static_cast<core::u32>(scaled));
    }
    _spikeDegree = spike;

    const core::u32 robustInteger = _robustScale.raw() > 0 ? static_cast<core::u32>(_robustScale.raw() >> 16) : 0u;
    const core::u64 spikeRatio = robustInteger == 0u ? 0u : (static_cast<core::u64>(robustInteger) << 16) / deltaRaw;
    const math::Fixed32 spikeWeight =
        _robustScale * fixedLn(spikeRatio > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<core::u32>(spikeRatio));

    _cumulative.resize(k, core::u64{0});
    for (core::u32 degree = 1u; degree <= k; ++degree)
    {
        core::u64 weight = degree == 1u ? inverseQ32(k) : inverseQ32(static_cast<core::u64>(degree) * (degree - 1u));

        if (degree < spike)
        {
            const core::u64 numerator = static_cast<core::u64>(robustInteger) << 32;
            const core::u64 denominator = static_cast<core::u64>(degree) * k;
            weight += denominator == 0u ? 0u : numerator / denominator;
        }
        else if (degree == spike)
        {
            const core::u64 spikeQ32 =
                spikeWeight.raw() > 0 ? (static_cast<core::u64>(spikeWeight.raw()) << 16) / k : 0u;
            weight += spikeQ32;
        }

        _total += weight;
        _cumulative[degree - 1u] = _total;
    }
}

core::u32 SolitonTable::drawDegree(math::Random &stream) const noexcept
{
    if (_sourceBlocks == 0u || _total == 0u)
        return 1u;

    const core::u32 draw = stream.next();
    const core::u64 target = ((_total >> 32) * draw) + (((_total & 0xFFFFFFFFull) * draw) >> 32);

    core::u32 lowDegree = 0u;
    core::u32 highDegree = _sourceBlocks - 1u;
    while (lowDegree < highDegree)
    {
        const core::u32 middle = lowDegree + (highDegree - lowDegree) / 2u;
        if (_cumulative[middle] > target)
            highDegree = middle;
        else
            lowDegree = middle + 1u;
    }
    return lowDegree + 1u;
}

core::u32 SolitonTable::fold(core::u32 seed) const noexcept
{
    core::u32 hash = seed;
    for (core::usize i = 0u; i < _cumulative.size(); ++i)
    {
        hash = (hash ^ static_cast<core::u32>(_cumulative[i] & 0xFFFFFFFFu)) * kFnv1aPrime;
        hash = (hash ^ static_cast<core::u32>(_cumulative[i] >> 32)) * kFnv1aPrime;
    }
    return hash;
}

void expandDroplet(core::u32 seed, const SolitonTable &table, DropletPlan &out)
{
    out.seed = seed;
    out.indices.clear();
    out.degree = 0u;

    const core::u32 blocks = table.degrees();
    if (blocks == 0u)
        return;

    math::Random stream = math::deriveStream(seed, kDropletSalt);
    core::u32 degree = table.drawDegree(stream);
    if (degree > blocks)
        degree = blocks;

    const core::u32 attemptBudget = degree * 8u + 16u;
    for (core::u32 attempt = 0u; attempt < attemptBudget && out.indices.size() < degree; ++attempt)
    {
        const core::u32 candidate = stream.below(blocks);
        bool seen = false;
        for (core::usize i = 0u; i < out.indices.size(); ++i)
            seen = seen || out.indices[i] == candidate;
        if (!seen)
            out.indices.push_back(candidate);
    }

    for (core::usize i = 1u; i < out.indices.size(); ++i)
    {
        const core::u32 held = out.indices[i];
        core::usize j = i;
        while (j > 0u && out.indices[j - 1u] > held)
        {
            out.indices[j] = out.indices[j - 1u];
            --j;
        }
        out.indices[j] = held;
    }

    out.degree = static_cast<core::u32>(out.indices.size());
}

} // namespace lpl::codec
