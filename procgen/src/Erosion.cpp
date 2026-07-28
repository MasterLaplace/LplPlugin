/**
 * @file Erosion.cpp
 * @brief Implementation of the thermal and hydraulic erosion passes.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Erosion.hpp>

namespace lpl::procgen {

namespace {

/// Clamps a fraction parameter into [0, 1]; the models diverge outside it.
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

    // Deltas accumulate into a separate buffer and are applied at the end of
    // each pass. Writing back immediately would make a cell's fate depend on
    // whether its neighbour had already been visited this pass, turning a
    // symmetric physical rule into a scan-order artefact (visible as a diagonal
    // grain across the terrain).
    Heightfield delta{field.width(), field.depth(), math::Fixed32::zero()};

    for (core::u32 pass = 0u; pass < params.iterations; ++pass)
    {
        delta.fill(math::Fixed32::zero());

        for (core::u32 z = 0u; z < field.depth(); ++z)
        {
            for (core::u32 x = 0u; x < field.width(); ++x)
            {
                const math::Fixed32 here = field.at(x, z);

                // First sweep: how much excess is there, and where can it go?
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

                // Second sweep: share the moved material in proportion to each
                // neighbour's excess, so the steepest side receives the most.
                //
                // The amount is a fraction of the LARGEST excess, not of their
                // sum. A peak with four steep sides has four excesses, and
                // moving a fraction of the total would strip up to four times
                // what any single slope justifies — enough to dig the peak below
                // its neighbours and invert the slope it was supposed to relax.
                // The terrain then flattens toward its mean instead of relaxing
                // toward the talus angle, which measurably INCREASES roughness
                // and leaves no gradient for water to follow.
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
        // ── Rain ────────────────────────────────────────────────────────────
        for (core::u32 i = 0u; i < field.cellCount(); ++i)
            water[i] = water[i] + rain;

        // ── Erode or deposit, according to the transport capacity ────────────
        //
        // C = Kc * |slope| * water. Below it the flow tears material out of the
        // bed; above it the flow is saturated and lets material go. Both are
        // rate-limited so the surface converges instead of oscillating between
        // the two states on alternate passes.
        for (core::u32 z = 0u; z < field.depth(); ++z)
        {
            for (core::u32 x = 0u; x < field.width(); ++x)
            {
                const core::u32 index = field.index(x, z);
                if (water[index].raw() == 0)
                    continue;

                math::Fixed32 slope = slopeAt(field, x, z);
                // A floor on the slope keeps standing water slightly erosive
                // rather than perfectly inert, which is what stops a filled
                // basin from becoming a permanent sediment trap.
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

        // ── Flow: water moves to lower neighbours, carrying its sediment ────
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

                // Never move more water than the cell holds, nor more than
                // would level it with its neighbours: either would oscillate.
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
                    // Sediment travels with the water that carries it.
                    const math::Fixed32 movedSediment =
                        water.at(x, z).raw() != 0 ? sediment.at(x, z) * (movedWater / water.at(x, z))
                                                  : math::Fixed32::zero();

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

        // ── Evaporate ───────────────────────────────────────────────────────
        //
        // Losing water lowers the capacity, so what the next pass finds above it
        // is deposited then. Dropping the excess here as well would deposit twice
        // per cycle and starve the erosion side of the equation.
        for (core::u32 i = 0u; i < field.cellCount(); ++i)
            water[i] = water[i] - water[i] * evaporation;
    }

    // Any sediment still suspended when the rain stops settles where it is,
    // otherwise the pass would quietly destroy material.
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
        field[i] = field[i] + sediment[i];

    return displaced;
}

} // namespace lpl::procgen
