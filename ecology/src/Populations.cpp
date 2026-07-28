/**
 * @file Populations.cpp
 * @brief Implementation of the bounded predator-prey web.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecology/Populations.hpp>

namespace lpl::ecology {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

} // namespace

core::u32 TrophicWeb::add(const SpeciesParams &params, math::Fixed32 initial, core::u32 preyIndex)
{
    Species entry;
    entry.params = params;
    entry.population = initial;
    entry.preyIndex = preyIndex;
    species.push_back(entry);
    return static_cast<core::u32>(species.size()) - 1u;
}

void TrophicWeb::step(core::u32 steps)
{
    for (core::u32 s = 0u; s < steps; ++s)
    {
        // Every species is stepped from the SAME snapshot. Updating in place
        // would let a predator eat prey that the same sweep has already grown,
        // so the result would depend on the order species were registered in.
        lpl::pmr::vector<math::Fixed32> before(species.size(), math::Fixed32::zero());
        for (core::u32 i = 0u; i < species.size(); ++i)
            before[i] = species[i].population;

        for (core::u32 i = 0u; i < species.size(); ++i)
        {
            Species &self = species[i];
            const SpeciesParams &p = self.params;
            const math::Fixed32 n = before[i];

            math::Fixed32 delta{};

            if (self.preyIndex == Species::kNoPrey)
            {
                // Producers and unfed prey grow logistically. The (1 - N/K) term
                // is what turns a neutrally stable orbit into a stable one: a
                // population knocked off equilibrium comes BACK instead of
                // oscillating at its new amplitude forever.
                const math::Fixed32 crowding =
                    p.capacity.raw() != 0 ? math::Fixed32::one() - n / p.capacity : math::Fixed32::zero();
                delta = p.growth * n * crowding;
            }
            else
            {
                // Predators: mass action against the prey ABOVE its refuge, less
                // starvation. The conversion factor is where the ten-percent rule
                // enters the dynamics rather than being a comment about them.
                const math::Fixed32 preyTotal = before[self.preyIndex];
                const math::Fixed32 preyParams = species[self.preyIndex].params.refuge;
                math::Fixed32 available = preyTotal - preyParams;
                if (available < math::Fixed32::zero())
                    available = math::Fixed32::zero();

                const math::Fixed32 eaten = p.predation * n * available;
                delta = p.conversion * eaten - p.mortality * n;

                // The same meal, taken off the prey. Symmetry matters: energy
                // that appears in a predator without leaving its prey is energy
                // the world creates from nothing, and the pyramid stops meaning
                // anything.
                Species &prey = species[self.preyIndex];
                math::Fixed32 remaining = prey.population - eaten;
                if (remaining < prey.params.refuge)
                    remaining = prey.params.refuge;
                prey.population = remaining;
            }

            math::Fixed32 next = self.population + delta;

            // Two clamps, and both are load-bearing. The refuge stops the
            // pseudo-extinction the classical model walks into; the capacity
            // stops a fixed-point overflow from a runaway exponential.
            if (next < p.refuge)
                next = p.refuge;
            if (p.capacity.raw() != 0 && next > p.capacity)
                next = p.capacity;
            self.population = next;
        }
    }
}

math::Fixed32 TrophicWeb::populationOf(core::u32 index) const
{
    if (index >= species.size())
        return math::Fixed32::zero();
    return species[index].population;
}

math::Fixed32 TrophicWeb::levelTotal(TrophicLevel level) const
{
    math::Fixed32 total{};
    for (core::u32 i = 0u; i < species.size(); ++i)
        if (species[i].params.level == level)
            total = total + species[i].population;
    return total;
}

void TrophicWeb::extirpate(core::u32 index)
{
    if (index >= species.size())
        return;

    // Zero AND refuge-zero: a species that has been hunted out must not be
    // resurrected by its own floor on the next step. That floor exists to survive
    // predation, not to survive removal.
    species[index].population = math::Fixed32::zero();
    species[index].params.refuge = math::Fixed32::zero();
    species[index].params.growth = math::Fixed32::zero();
}

core::u32 TrophicWeb::fold() const
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < species.size(); ++i)
    {
        const core::u32 raw = static_cast<core::u32>(species[i].population.raw());
        hash = (hash ^ (raw & 0xFFFFu)) * kFnv1aPrime;
        hash = (hash ^ (raw >> 16)) * kFnv1aPrime;
    }
    return hash;
}

} // namespace lpl::ecology
