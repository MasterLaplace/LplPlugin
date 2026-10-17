#ifndef LPL_ECOLOGY_POPULATIONS_INL
#define LPL_ECOLOGY_POPULATIONS_INL

namespace lpl::ecology {

constexpr math::Fixed32 transferEfficiency(TrophicLevel level) noexcept
{
    (void) level;
    return math::Fixed32::fromRaw(0x199A);
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_POPULATIONS_INL
