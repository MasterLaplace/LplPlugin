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
                const math::Fixed32 crowding =
                    p.capacity.raw() != 0 ? math::Fixed32::one() - n / p.capacity : math::Fixed32::zero();
                delta = p.growth * n * crowding;
            }
            else
            {
                const math::Fixed32 preyTotal = before[self.preyIndex];
                const math::Fixed32 preyParams = species[self.preyIndex].params.refuge;
                math::Fixed32 available = preyTotal - preyParams;
                if (available < math::Fixed32::zero())
                    available = math::Fixed32::zero();

                const math::Fixed32 eaten = p.predation * n * available;
                delta = p.conversion * eaten - p.mortality * n;

                Species &prey = species[self.preyIndex];
                math::Fixed32 remaining = prey.population - eaten;
                if (remaining < prey.params.refuge)
                    remaining = prey.params.refuge;
                prey.population = remaining;
            }

            math::Fixed32 next = self.population + delta;

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
