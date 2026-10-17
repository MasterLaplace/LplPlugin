#ifndef LPL_AI_AFFORDANCE_INL
#define LPL_AI_AFFORDANCE_INL

namespace lpl::ai {

inline bool AffordanceRegistry::best(core::u16 kinds, core::u32 x, core::u32 z, core::u32 width, math::Fixed32 need,
                                     core::u32 &outIndex) const
{
    bool found = false;
    math::Fixed32 bestValue{};
    for (core::u32 i = 0u; i < _offers.size(); ++i)
    {
        const Affordance &offer = _offers[i];
        if ((offer.kinds & kinds) == 0u)
            continue;
        if (need < offer.requiredNeed)
            continue;

        const core::u32 ox = offer.cell % width;
        const core::u32 oz = offer.cell / width;
        const core::u32 dx = ox > x ? ox - x : x - ox;
        const core::u32 dz = oz > z ? oz - z : z - oz;
        if ((dx > dz ? dx : dz) > offer.radius)
            continue;

        if (!found || offer.value > bestValue)
        {
            found = true;
            bestValue = offer.value;
            outIndex = i;
        }
    }
    return found;
}

} // namespace lpl::ai

#endif // LPL_AI_AFFORDANCE_INL
