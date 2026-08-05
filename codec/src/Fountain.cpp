/**
 * @file Fountain.cpp
 * @brief The LT encoder and the in-silico constraint filter.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/Fountain.hpp>

namespace lpl::codec {

bool satisfiesBiologicalLimits(const core::u8 *bytes, core::u32 count, const BiologicalLimits &limits) noexcept
{
    if (bytes == nullptr || count == 0u)
        return false;

    const core::u32 bases = count * 4u;
    core::u32 gc = 0u;
    core::u32 run = 1u;
    core::u32 longestRun = 1u;
    core::u32 previous = 4u; // no base has this value, so the first comparison fails

    for (core::u32 i = 0u; i < bases; ++i)
    {
        const core::u8 byte = bytes[i / 4u];
        const core::u32 shift = 6u - 2u * (i % 4u);
        const core::u32 base = static_cast<core::u32>((byte >> shift) & 0x3u);

        // C and G are the two that pair with three hydrogen bonds, and their share is
        // what sets the melting temperature.
        if (base == 1u || base == 2u)
            ++gc;

        if (base == previous)
        {
            ++run;
            if (run > longestRun)
                longestRun = run;
        }
        else
        {
            run = 1u;
        }
        previous = base;
    }

    if (longestRun > limits.maxHomopolymer)
        return false;

    const core::u32 gcPermille = (gc * 1000u) / bases;
    return gcPermille >= limits.minGcPermille && gcPermille <= limits.maxGcPermille;
}

Fountain::Fountain(const SourceView &source, const SolitonParams &tuning) : _source(source)
{
    SolitonParams params = tuning;
    params.sourceBlocks = source.blockCount;
    _table.build(params);
}

void Fountain::emit(core::u32 seed, Droplet &out) const
{
    out.seed = seed;
    out.payload.clear();
    if (_source.bytes == nullptr || _source.blockBytes == 0u || _source.blockCount == 0u)
        return;

    out.payload.resize(_source.blockBytes, core::u8{0});

    DropletPlan plan;
    expandDroplet(seed, _table, plan);

    // The combination itself: over GF(2) adding is XOR, so a droplet of degree d is d
    // blocks folded together and nothing more. Byte-wise rather than through
    // XorKernel because a block is not word-aligned in general — the payload comes
    // from the caller's buffer — and a droplet payload is tens of bytes, not the
    // kilobyte rows where the vector path earns its keep.
    for (core::usize i = 0u; i < plan.indices.size(); ++i)
    {
        const core::u8 *const block = _source.bytes + static_cast<core::usize>(plan.indices[i]) * _source.blockBytes;
        for (core::u32 b = 0u; b < _source.blockBytes; ++b)
            out.payload[b] = static_cast<core::u8>(out.payload[b] ^ block[b]);
    }
}

core::u32 Fountain::emitValid(core::u32 count, const BiologicalLimits &limits, core::u32 firstSeed,
                              lpl::pmr::vector<Droplet> &out) const
{
    out.clear();
    if (count == 0u)
        return 0u;

    core::u32 examined = 0u;
    Droplet candidate;

    // Bounded, because "the fountain never runs dry" is a statement about the code and
    // not about this loop: limits strict enough to reject everything would spin here
    // forever, and a budget turns that into a short output the caller can notice.
    const core::u32 budget = count * 64u + 1024u;
    for (core::u32 seed = firstSeed; out.size() < count && examined < budget; ++seed)
    {
        ++examined;
        emit(seed, candidate);
        if (!satisfiesBiologicalLimits(candidate.payload.data(), static_cast<core::u32>(candidate.payload.size()),
                                       limits))
            continue;

        Droplet kept;
        kept.seed = candidate.seed;
        kept.payload = candidate.payload;
        out.push_back(kept);
    }

    return examined;
}

void simulateDecay(lpl::pmr::vector<Droplet> &pool, core::u32 years, const DecayParams &params, math::Random &stream,
                   DecayReport &outReport)
{
    outReport = DecayReport{};
    outReport.strands = static_cast<core::u32>(pool.size());

    const core::u32 centuries = years / 100u;
    if (centuries == 0u)
    {
        outReport.intact = outReport.strands;
        return;
    }

    // Rates compound over centuries rather than multiply: a strand that survives ten
    // centuries survived each of them, and a linear rate would pass one at a few
    // hundred years. The loop IS the compounding, which also keeps every draw an
    // integer comparison.
    lpl::pmr::vector<Droplet> survivors;
    for (core::usize d = 0u; d < pool.size(); ++d)
    {
        bool lost = false;
        core::u32 substitutions = 0u;

        for (core::u32 century = 0u; century < centuries && !lost; ++century)
        {
            if (stream.below(1000000u) < params.breakPerMillionPerCentury)
                lost = true;
            else if (stream.below(1000000u) < params.dropoutPerMillionPerCentury)
                lost = true;
        }

        if (lost)
        {
            ++outReport.lost;
            continue;
        }

        Droplet aged;
        aged.seed = pool[d].seed;
        aged.payload = pool[d].payload;

        const core::u32 bases = static_cast<core::u32>(aged.payload.size()) * 4u;
        for (core::u32 century = 0u; century < centuries; ++century)
            for (core::u32 base = 0u; base < bases; ++base)
            {
                if (stream.below(1000000u) >= params.substitutionPerMillionPerCentury)
                    continue;
                // Flip one of the two bits of that base: a substitution is a base read
                // as a DIFFERENT base, so the value has to change.
                const core::u32 shift = 6u - 2u * (base % 4u);
                const core::u8 delta = static_cast<core::u8>((1u + stream.below(3u)) << shift);
                aged.payload[base / 4u] = static_cast<core::u8>(aged.payload[base / 4u] ^ delta);
                ++substitutions;
            }

        outReport.substitutions += substitutions;
        ++outReport.intact;
        survivors.push_back(aged);
    }

    pool.clear();
    for (core::usize i = 0u; i < survivors.size(); ++i)
        pool.push_back(survivors[i]);
}

} // namespace lpl::codec
