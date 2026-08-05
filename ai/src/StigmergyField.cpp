/**
 * @file StigmergyField.cpp
 * @brief Implementation of the shared stigmergy substrate.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ai/StigmergyField.hpp>

namespace lpl::ai {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

} // namespace

StigmergyField::StigmergyField(core::u32 width, core::u32 depth, core::u32 channels)
    : _width(width), _depth(depth), _channels(channels > kMaxStigmergyChannels ? kMaxStigmergyChannels : channels)
{
    if (_width == 0u || _depth == 0u || _channels == 0u)
    {
        _width = 0u;
        _depth = 0u;
        _channels = 0u;
        return;
    }
    const core::usize total = static_cast<core::usize>(_width) * _depth * _channels;
    _cells.resize(total, math::Fixed32::zero());
    _scratch.resize(total, math::Fixed32::zero());
    _blocked.resize(static_cast<core::usize>(_width) * _depth, core::u8{0});
}

void StigmergyField::setObstacles(const procgen::Grid<core::u8> &blocked)
{
    if (blocked.width() != _width || blocked.depth() != _depth)
        return;
    for (core::u32 i = 0u; i < _blocked.size(); ++i)
        _blocked[i] = blocked[i] != 0u ? 1u : 0u;
}

void StigmergyField::clear()
{
    for (core::u32 i = 0u; i < _cells.size(); ++i)
        _cells[i] = math::Fixed32::zero();
}

void StigmergyField::deposit(core::u32 channel, core::u32 x, core::u32 z, math::Fixed32 amount)
{
    if (channel >= _channels || x >= _width || z >= _depth)
        return;
    if (_blocked[z * _width + x] != 0u)
        return; // Nothing stands inside a wall, so nothing deposits there.
    _cells[index(channel, x, z)] = _cells[index(channel, x, z)] + amount;
}

void StigmergyField::depositTrail(core::u32 channel, const core::u32 *cells, core::u32 count, math::Fixed32 quality)
{
    if (channel >= _channels || cells == nullptr || count == 0u)
        return;

    // Q / L. The whole of "shorter is better" lives on this line: a route half as
    // long lays twice the trail per cell, so the field prefers it after enough
    // traversals without any agent ever comparing two routes.
    const math::Fixed32 perCell = quality / math::Fixed32::fromInt(static_cast<core::i32>(count));
    for (core::u32 i = 0u; i < count; ++i)
    {
        const core::u32 cell = cells[i];
        if (cell >= _width * _depth)
            continue;
        deposit(channel, cell % _width, cell / _width, perCell);
    }
}

void StigmergyField::step(const StigmergyParams &params)
{
    if (empty())
        return;

    const math::Fixed32 retain = math::Fixed32::fromFloat(params.evaporation);
    const math::Fixed32 spread = math::Fixed32::fromFloat(params.diffusion);
    const math::Fixed32 ceiling = math::Fixed32::fromFloat(params.maximum);
    const math::Fixed32 floor = math::Fixed32::fromFloat(params.floor);
    const math::Fixed32 quarter = spread / math::Fixed32::fromInt(4);
    const math::Fixed32 kept = math::Fixed32::one() - spread;

    for (core::u32 c = 0u; c < _channels; ++c)
    {
        // ── Evaporate, into the scratch buffer ──────────────────────────────
        for (core::u32 z = 0u; z < _depth; ++z)
            for (core::u32 x = 0u; x < _width; ++x)
            {
                math::Fixed32 v = _cells[index(c, x, z)] * retain;
                if (v > ceiling)
                    v = ceiling;
                // The floor. Without it a value walks down to one Q16.16 tick and
                // stays there forever, so a stale trail never actually clears.
                if (v < floor)
                    v = math::Fixed32::zero();
                _scratch[index(c, x, z)] = v;
            }

        // ── Diffuse, reading scratch and writing cells ──────────────────────
        //
        // Two buffers, always. A stencil done in place reads neighbours the same
        // sweep has already updated, so the answer depends on whether the loop
        // ran left-to-right — which is a desynchronisation, not a rounding error.
        for (core::u32 z = 0u; z < _depth; ++z)
        {
            for (core::u32 x = 0u; x < _width; ++x)
            {
                if (_blocked[z * _width + x] != 0u)
                {
                    _cells[index(c, x, z)] = math::Fixed32::zero();
                    continue;
                }

                math::Fixed32 inflow = math::Fixed32::zero();
                math::Fixed32 outflow = math::Fixed32::zero();
                const math::Fixed32 here = _scratch[index(c, x, z)];

                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor4Z[n];
                    const bool inside = nx >= 0 && nz >= 0 && static_cast<core::u32>(nx) < _width &&
                                        static_cast<core::u32>(nz) < _depth;
                    // A wall blocks in BOTH directions: nothing flows in from it,
                    // and nothing flows out into it. Letting it absorb would make
                    // walls into sinks, so a corridor would read as less scented
                    // than an open room carrying the same traffic.
                    const bool open =
                        inside && _blocked[static_cast<core::u32>(nz) * _width + static_cast<core::u32>(nx)] == 0u;
                    if (!open)
                        continue;
                    inflow = inflow + _scratch[index(c, static_cast<core::u32>(nx), static_cast<core::u32>(nz))];
                    outflow = outflow + quarter;
                }

                math::Fixed32 v = here - here * outflow + inflow * quarter;
                if (v > ceiling)
                    v = ceiling;
                if (v < floor)
                    v = math::Fixed32::zero();
                _cells[index(c, x, z)] = v;
            }
        }
        (void) kept;
    }
}

math::Fixed32 StigmergyField::value(core::u32 channel, core::u32 x, core::u32 z) const
{
    if (channel >= _channels || x >= _width || z >= _depth)
        return math::Fixed32::zero();
    return _cells[index(channel, x, z)];
}

core::u32 StigmergyField::gradientDirection(core::u32 channel, core::u32 x, core::u32 z, bool uphill) const
{
    if (channel >= _channels || x >= _width || z >= _depth)
        return kNoDirection;

    const math::Fixed32 here = _cells[index(channel, x, z)];
    core::u32 best = kNoDirection;
    math::Fixed32 bestValue = here;

    for (core::u32 n = 0u; n < 8u; ++n)
    {
        const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor8X[n];
        const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor8Z[n];
        if (nx < 0 || nz < 0 || static_cast<core::u32>(nx) >= _width || static_cast<core::u32>(nz) >= _depth)
            continue;
        if (_blocked[static_cast<core::u32>(nz) * _width + static_cast<core::u32>(nx)] != 0u)
            continue;

        const math::Fixed32 v = _cells[index(channel, static_cast<core::u32>(nx), static_cast<core::u32>(nz))];
        // Strictly better, so a plateau leaves the agent where it is rather than
        // letting the neighbour order decide — the same tie rule the rest of the
        // engine uses, and the reason two targets agree.
        const bool better = uphill ? v > bestValue : v < bestValue;
        if (better)
        {
            bestValue = v;
            best = n;
        }
    }
    return best;
}

core::u32 StigmergyField::palateDirection(const ScentPalate &palate, core::u32 x, core::u32 z) const
{
    if (palate.count == 0u || x >= _width || z >= _depth)
        return kNoDirection;

    // Scores a cell against everything this animal cares about. Attraction and
    // repulsion are the same sum with opposite signs, which is why a wolf and a
    // deer can share one field and one line of code.
    const auto score = [&](core::u32 cx, core::u32 cz) {
        math::Fixed32 total{};
        for (core::u32 t = 0u; t < palate.count; ++t)
        {
            const ScentAffinity &term = palate.terms[t];
            if (term.channel >= _channels)
                continue;
            total = total + _cells[index(term.channel, cx, cz)] * term.weight;
        }
        return total;
    };

    core::u32 best = kNoDirection;
    math::Fixed32 bestValue = score(x, z);

    for (core::u32 n = 0u; n < 8u; ++n)
    {
        const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor8X[n];
        const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor8Z[n];
        if (nx < 0 || nz < 0 || static_cast<core::u32>(nx) >= _width || static_cast<core::u32>(nz) >= _depth)
            continue;
        if (_blocked[static_cast<core::u32>(nz) * _width + static_cast<core::u32>(nx)] != 0u)
            continue;

        // Strictly better, so a plateau leaves the agent where it is rather than
        // letting the neighbour order decide — the same tie rule gradientDirection
        // uses, and the reason two targets agree.
        if (const math::Fixed32 v = score(static_cast<core::u32>(nx), static_cast<core::u32>(nz)); v > bestValue)
        {
            bestValue = v;
            best = n;
        }
    }
    return best;
}

core::u32 StigmergyField::fold() const
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < _cells.size(); ++i)
    {
        const core::u32 raw = static_cast<core::u32>(_cells[i].raw());
        hash = (hash ^ (raw & 0xFFu)) * kFnv1aPrime;
        hash = (hash ^ ((raw >> 8) & 0xFFu)) * kFnv1aPrime;
        hash = (hash ^ ((raw >> 16) & 0xFFu)) * kFnv1aPrime;
        hash = (hash ^ ((raw >> 24) & 0xFFu)) * kFnv1aPrime;
    }
    return hash;
}

} // namespace lpl::ai
