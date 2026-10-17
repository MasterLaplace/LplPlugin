#ifndef LPL_AI_ANT_COLONY_INL
#define LPL_AI_ANT_COLONY_INL

#include <lpl/procgen/Heightfield.hpp>

namespace lpl::ai {

inline void AntColony::reset(StigmergyField &field, core::u32 width, core::u32 depth, const AntColonyParams &params,
                             core::u32 nestX, core::u32 nestZ)
{
    _params = params;
    _width = width;
    _depth = depth;
    _returns = 0u;
    _explored = 0u;

    _nestX = width == 0u ? 0u : (nestX < width ? nestX : width / 2u);
    _nestZ = depth == 0u ? 0u : (nestZ < depth ? nestZ : depth / 2u);

    _x.clear();
    _z.clear();
    if (width == 0u || depth == 0u)
        return;

    for (core::u32 i = 0u; i < _params.agents; ++i)
    {
        _x.push_back(_nestX);
        _z.push_back(_nestZ);
    }
    _stream = _params.seed ^ 0xA57E0022u;

    const core::u32 nest = _nestZ * _width + _nestX;
    seedPheromoneField(field, _params.ants.channel, &nest, 1u, math::Fixed32::fromInt(60));
}

inline void AntColony::step(StigmergyField &field)
{
    if (_width == 0u || _x.empty())
        return;
    _explored = 0u;

    for (core::u32 i = 0u; i < _x.size(); ++i)
    {
        bool explored = false;
        const core::u32 direction = chooseAntMove(field, _params.ants, _x[i], _z[i], _stream, explored);
        if (explored)
            ++_explored;
        if (direction != StigmergyField::kNoDirection)
        {
            const core::i32 nx = static_cast<core::i32>(_x[i]) + procgen::kNeighbor8X[direction];
            const core::i32 nz = static_cast<core::i32>(_z[i]) + procgen::kNeighbor8Z[direction];
            if (nx >= 0 && nz >= 0 && static_cast<core::u32>(nx) < _width && static_cast<core::u32>(nz) < _depth)
            {
                _x[i] = static_cast<core::u32>(nx);
                _z[i] = static_cast<core::u32>(nz);
            }
        }

        const core::i32 dx = static_cast<core::i32>(_x[i]) - static_cast<core::i32>(_nestX);
        const core::i32 dz = static_cast<core::i32>(_z[i]) - static_cast<core::i32>(_nestZ);
        const core::i32 range = static_cast<core::i32>(_params.forageRange);
        if (dx * dx + dz * dz > range * range)
        {
            _x[i] = _nestX;
            _z[i] = _nestZ;
            ++_returns;
        }

        const core::u32 cell = _z[i] * _width + _x[i];
        field.depositTrail(_params.ants.channel, &cell, 1u, _params.ants.depositQuality);
    }
}

} // namespace lpl::ai

#endif // LPL_AI_ANT_COLONY_INL
