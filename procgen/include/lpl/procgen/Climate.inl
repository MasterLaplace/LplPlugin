#ifndef LPL_PROCGEN_CLIMATE_INL
#define LPL_PROCGEN_CLIMATE_INL

namespace lpl::procgen {

inline ClimateVector ClimateField::at(core::u32 x, core::u32 z) const noexcept
{
    ClimateVector v{};
    for (core::u32 i = 0u; i < kClimateAxisCount; ++i)
        v.axis[i] = axes[i].at(x, z);
    return v;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CLIMATE_INL
