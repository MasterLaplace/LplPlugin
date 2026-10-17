#ifndef LPL_AI_STIGMERGYFIELD_INL
#define LPL_AI_STIGMERGYFIELD_INL

namespace lpl::ai {

constexpr void ScentPalate::add(ScentChannel channel, math::Fixed32 weight) noexcept
{
    if (count < kMaxStigmergyChannels)
        terms[count++] = ScentAffinity{static_cast<core::u32>(channel), weight};
}

} // namespace lpl::ai

#endif // LPL_AI_STIGMERGYFIELD_INL
