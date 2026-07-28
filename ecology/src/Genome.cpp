/**
 * @file Genome.cpp
 * @brief Implementation of heredity, drift and the anomaly test.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecology/Genome.hpp>

#include <lpl/procgen/FixedMath.hpp>

namespace lpl::ecology {

namespace {

/// Advances a caller-owned xorshift stream. Every draw in this file goes through
/// it, so a creature's genetics depend on its own stream and never on the order
/// creatures happened to be processed in — which is what a global generator would
/// give, and what desynchronises a networked simulation.
core::u32 nextRandom(core::u32 &stream)
{
    stream ^= stream << 13;
    stream ^= stream >> 17;
    stream ^= stream << 5;
    if (stream == 0u)
        stream = 0x9E3779B9u;
    return stream;
}

/// A value in [0, 1) as Fixed32.
math::Fixed32 unitRandom(core::u32 &stream)
{
    return math::Fixed32::fromRaw(static_cast<core::i32>(nextRandom(stream) & 0xFFFFu));
}

/// Lowest a gene may fall. A gene at zero is a creature with no speed at all, and
/// once there, multiplication can never lift it again.
constexpr core::i32 kGeneFloor = 0x1999; // 0.1

math::Fixed32 mutateGene(math::Fixed32 value, core::u32 chance16, math::Fixed32 amplitude, core::u32 &stream)
{
    if ((nextRandom(stream) & 0xFu) >= chance16)
        return value;

    // Multiplicative, not additive: a gene worth 100 and one worth 1 should both
    // move by a comparable PROPORTION, or the same amplitude is a rounding error
    // for one and a catastrophe for the other.
    const math::Fixed32 swing = (unitRandom(stream) * math::Fixed32::fromInt(2) - math::Fixed32::one()) * amplitude;
    math::Fixed32 mutated = value * (math::Fixed32::one() + swing);
    if (mutated.raw() < kGeneFloor)
        mutated = math::Fixed32::fromRaw(kGeneFloor);
    return mutated;
}

} // namespace

Genome crossover(const Genome &a, const Genome &b, core::u32 &stream)
{
    const auto blend = [&](math::Fixed32 x, math::Fixed32 y) {
        // A mix weight in [0.3, 0.7] rather than [0, 1]: at the extremes blending
        // degenerates into copying one parent, and a population that copies stops
        // exploring the space between its members.
        const math::Fixed32 t = math::Fixed32::fromRaw(0x4CCC) + unitRandom(stream) * math::Fixed32::fromRaw(0x6666);
        return x * t + y * (math::Fixed32::one() - t);
    };

    Genome child;
    child.maxSpeed = blend(a.maxSpeed, b.maxSpeed);
    child.vision = blend(a.vision, b.vision);
    child.strength = blend(a.strength, b.strength);
    child.absorption = blend(a.absorption, b.absorption);
    child.size = blend(a.size, b.size);
    return child;
}

Genome mutate(const Genome &genome, core::u32 chance16, core::f32 amplitude, core::u32 &stream)
{
    const math::Fixed32 swing = math::Fixed32::fromFloat(amplitude);
    Genome mutated;
    mutated.maxSpeed = mutateGene(genome.maxSpeed, chance16, swing, stream);
    mutated.vision = mutateGene(genome.vision, chance16, swing, stream);
    mutated.strength = mutateGene(genome.strength, chance16, swing, stream);
    mutated.absorption = mutateGene(genome.absorption, chance16, swing, stream);
    mutated.size = mutateGene(genome.size, chance16, swing, stream);
    return mutated;
}

bool inMutationalMeltdown(math::Fixed32 local, math::Fixed32 capacity, const HeredityParams &params)
{
    if (capacity.raw() <= 0)
        return false;
    // A SHARE of what the habitat supports, never a head count. Four survivors is
    // a collapse in a valley that held forty and unremarkable in one that held
    // four thousand, and only the ratio knows the difference.
    const math::Fixed32 share = local / capacity;
    return share < math::Fixed32::fromRaw(static_cast<core::i32>(params.collapseShare16) * 0x1000);
}

PopulationStats strengthStats(const Genome *genomes, core::u32 count)
{
    PopulationStats stats;
    if (genomes == nullptr || count == 0u)
        return stats;

    stats.count = count;
    math::Fixed32 total{};
    for (core::u32 i = 0u; i < count; ++i)
        total = total + genomes[i].strength;
    stats.mean = total / math::Fixed32::fromInt(static_cast<core::i32>(count));

    math::Fixed32 variance{};
    for (core::u32 i = 0u; i < count; ++i)
    {
        const math::Fixed32 d = genomes[i].strength - stats.mean;
        variance = variance + d * d;
    }
    variance = variance / math::Fixed32::fromInt(static_cast<core::i32>(count));
    stats.deviation = procgen::fixedSqrt(variance);
    return stats;
}

bool isAnomaly(const Genome &genome, const PopulationStats &stats, const HeredityParams &params)
{
    // A species of one has no distribution to be exceptional against, and calling
    // the last survivor a boss because it is its own mean plus zero would make an
    // anomaly out of every extinction.
    if (stats.count < 4u || stats.deviation.raw() <= 0)
        return false;
    const math::Fixed32 threshold = stats.mean + stats.deviation * math::Fixed32::fromFloat(params.anomalySigma);
    return genome.strength > threshold;
}

Genome breed(const Genome &a, const Genome &b, math::Fixed32 local, math::Fixed32 capacity,
             const HeredityParams &params, core::u32 &stream)
{
    const Genome child = crossover(a, b, stream);

    // The one conditional the whole boss mechanic rests on. Under collapse the
    // population stops being averaged back toward its mean and starts wandering,
    // so most children are worse and one, occasionally, is something the species
    // could never have produced while it was healthy.
    if (inMutationalMeltdown(local, capacity, params))
        return mutate(child, params.meltdownChance16, params.meltdownAmplitude, stream);
    return mutate(child, params.mutationChance16, params.mutationAmplitude, stream);
}

} // namespace lpl::ecology
