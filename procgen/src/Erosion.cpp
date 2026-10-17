#include <lpl/procgen/Erosion.hpp>

namespace lpl::procgen {

namespace {

/**
 * @brief Clamps a fraction parameter into [0, 1]; the models diverge outside it.
 */
math::Fixed32 asFraction(core::f32 value)
{
    math::Fixed32 f = math::Fixed32::fromFloat(value);
    if (f < math::Fixed32::zero())
        return math::Fixed32::zero();
    if (f > math::Fixed32::one())
        return math::Fixed32::one();
    return f;
}

} // namespace

math::Fixed32 thermalErode(Heightfield &field, const ThermalErosionParams &params)
{
    if (field.empty() || params.iterations == 0u)
        return math::Fixed32::zero();

    const math::Fixed32 talus = math::Fixed32::fromFloat(params.talus);
    const math::Fixed32 carry = asFraction(params.carryFraction);
    math::Fixed32 moved = math::Fixed32::zero();

    Heightfield delta{field.width(), field.depth(), math::Fixed32::zero()};

    for (core::u32 pass = 0u; pass < params.iterations; ++pass)
    {
        delta.fill(math::Fixed32::zero());

        for (core::u32 z = 0u; z < field.depth(); ++z)
        {
            for (core::u32 x = 0u; x < field.width(); ++x)
            {
                const math::Fixed32 here = field.at(x, z);

                math::Fixed32 totalExcess = math::Fixed32::zero();
                math::Fixed32 largestExcess = math::Fixed32::zero();
                math::Fixed32 excess[4] = {};
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                    if (!field.contains(nx, nz))
                        continue;
                    const math::Fixed32 drop = here - field.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                    if (drop > talus)
                    {
                        excess[n] = drop - talus;
                        totalExcess = totalExcess + excess[n];
                        if (excess[n] > largestExcess)
                            largestExcess = excess[n];
                    }
                }
                if (totalExcess.raw() == 0)
                    continue;

                const math::Fixed32 amount = largestExcess * carry;
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    if (excess[n].raw() == 0)
                        continue;
                    const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                    const math::Fixed32 share = amount * (excess[n] / totalExcess);
                    delta.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) =
                        delta.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) + share;
                    delta.at(x, z) = delta.at(x, z) - share;
                    moved = moved + share;
                }
            }
        }

        for (core::u32 i = 0u; i < field.cellCount(); ++i)
            field[i] = field[i] + delta[i];
    }
    return moved;
}

math::Fixed32 hydraulicErode(Heightfield &field, const HydraulicErosionParams &params)
{
    if (field.empty() || params.iterations == 0u)
        return math::Fixed32::zero();

    const math::Fixed32 rain = math::Fixed32::fromFloat(params.rainAmount);
    const math::Fixed32 solubility = asFraction(params.solubility);
    const math::Fixed32 evaporation = asFraction(params.evaporation);
    const math::Fixed32 capacity = math::Fixed32::fromFloat(params.sedimentCapacity);
    const math::Fixed32 deposition = asFraction(params.deposition);
    const math::Fixed32 minSlope = math::Fixed32::fromFloat(params.minSlope);

    Heightfield water{field.width(), field.depth(), math::Fixed32::zero()};
    Heightfield sediment{field.width(), field.depth(), math::Fixed32::zero()};
    Heightfield waterDelta{field.width(), field.depth(), math::Fixed32::zero()};
    Heightfield sedimentDelta{field.width(), field.depth(), math::Fixed32::zero()};

    math::Fixed32 displaced = math::Fixed32::zero();

    for (core::u32 pass = 0u; pass < params.iterations; ++pass)
    {
        for (core::u32 i = 0u; i < field.cellCount(); ++i)
            water[i] = water[i] + rain;

        for (core::u32 z = 0u; z < field.depth(); ++z)
        {
            for (core::u32 x = 0u; x < field.width(); ++x)
            {
                const core::u32 index = field.index(x, z);
                if (water[index].raw() == 0)
                    continue;

                math::Fixed32 slope = slopeAt(field, x, z);
                if (slope < minSlope)
                    slope = minSlope;

                const math::Fixed32 holdable = capacity * slope * water[index];
                if (sediment[index] < holdable)
                {
                    const math::Fixed32 dissolved = (holdable - sediment[index]) * solubility;
                    field[index] = field[index] - dissolved;
                    sediment[index] = sediment[index] + dissolved;
                    displaced = displaced + dissolved;
                }
                else
                {
                    const math::Fixed32 dropped = (sediment[index] - holdable) * deposition;
                    field[index] = field[index] + dropped;
                    sediment[index] = sediment[index] - dropped;
                    displaced = displaced + dropped;
                }
            }
        }

        waterDelta.fill(math::Fixed32::zero());
        sedimentDelta.fill(math::Fixed32::zero());

        for (core::u32 z = 0u; z < field.depth(); ++z)
        {
            for (core::u32 x = 0u; x < field.width(); ++x)
            {
                const math::Fixed32 here = field.at(x, z) + water.at(x, z);
                if (water.at(x, z).raw() == 0)
                    continue;

                math::Fixed32 totalDrop = math::Fixed32::zero();
                math::Fixed32 drops[4] = {};
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                    if (!field.contains(nx, nz))
                        continue;
                    const math::Fixed32 there = field.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) +
                                                water.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                    if (here > there)
                    {
                        drops[n] = here - there;
                        totalDrop = totalDrop + drops[n];
                    }
                }
                if (totalDrop.raw() == 0)
                    continue;

                math::Fixed32 movable = water.at(x, z);
                if (totalDrop < movable)
                    movable = totalDrop;

                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    if (drops[n].raw() == 0)
                        continue;
                    const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                    const math::Fixed32 fraction = drops[n] / totalDrop;
                    const math::Fixed32 movedWater = movable * fraction;
                    const math::Fixed32 movedSediment = water.at(x, z).raw() != 0 ?
                                                            sediment.at(x, z) * (movedWater / water.at(x, z)) :
                                                            math::Fixed32::zero();

                    waterDelta.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) =
                        waterDelta.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) + movedWater;
                    waterDelta.at(x, z) = waterDelta.at(x, z) - movedWater;

                    sedimentDelta.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) =
                        sedimentDelta.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) + movedSediment;
                    sedimentDelta.at(x, z) = sedimentDelta.at(x, z) - movedSediment;
                }
            }
        }

        for (core::u32 i = 0u; i < field.cellCount(); ++i)
        {
            water[i] = water[i] + waterDelta[i];
            sediment[i] = sediment[i] + sedimentDelta[i];
        }

        for (core::u32 i = 0u; i < field.cellCount(); ++i)
            water[i] = water[i] - water[i] * evaporation;
    }

    for (core::u32 i = 0u; i < field.cellCount(); ++i)
        field[i] = field[i] + sediment[i];

    return displaced;
}

} // namespace lpl::procgen
