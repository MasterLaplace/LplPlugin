#ifndef LPL_AI_PERSONALITY_INL
#define LPL_AI_PERSONALITY_INL

namespace lpl::ai {

constexpr PersonalityTraits personalityOf(core::u32 id, core::u32 species) noexcept
{
    const auto avalanche = [](core::u32 v) -> core::u32 {
        v ^= v >> 16;
        v *= 0x7FEB352Du;
        v ^= v >> 15;
        v *= 0x846CA68Bu;
        v ^= v >> 16;
        return v;
    };

    const auto trait = [&](core::u32 salt) -> math::Fixed32 {
        const core::u32 h = avalanche((id ^ (species * 0x9E3779B1u)) * salt + salt);
        return math::Fixed32::fromRaw(static_cast<core::i32>(h & 0xFFFFu));
    };

    PersonalityTraits traits;
    traits.aggression = trait(0x1B873593u);
    traits.bravery = trait(0xCC9E2D51u);
    traits.dominance = trait(0x85EBCA6Bu);
    traits.energy = trait(0xC2B2AE35u);
    traits.nervousness = trait(0x27D4EB2Fu);
    traits.sympathy = trait(0x165667B1u);
    return traits;
}

constexpr math::Fixed32 fleeThreshold(const PersonalityTraits &traits) noexcept
{
    math::Fixed32 t = math::Fixed32::half() + (traits.bravery - traits.nervousness) * math::Fixed32::half();
    if (t < math::Fixed32::zero())
        t = math::Fixed32::zero();
    if (t > math::Fixed32::one())
        t = math::Fixed32::one();
    return t;
}

} // namespace lpl::ai

#endif // LPL_AI_PERSONALITY_INL
