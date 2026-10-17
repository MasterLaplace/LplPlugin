#ifndef LPL_ECOLOGY_GENOME_INL
#define LPL_ECOLOGY_GENOME_INL

namespace lpl::ecology {

inline math::Fixed32 Fitness::score() const noexcept
{
    return math::Fixed32::fromInt(static_cast<core::i32>(age)) +
           math::Fixed32::fromInt(static_cast<core::i32>(kills)) * math::Fixed32::fromInt(10) +
           energyAbsorbed / math::Fixed32::fromInt(10) - damageTaken / math::Fixed32::fromInt(2);
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_GENOME_INL
