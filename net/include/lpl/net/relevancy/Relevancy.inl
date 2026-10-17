#ifndef LPL_NET_RELEVANCY_RELEVANCY_INL
#define LPL_NET_RELEVANCY_RELEVANCY_INL

namespace lpl::net::relevancy {

inline float priority(float distanceSq, core::u32 ticksSinceLastSent) noexcept
{
    const float proximity = 1.0f / (1.0f + distanceSq);
    const float staleness = static_cast<float>(ticksSinceLastSent);
    return kProximityWeight * proximity + staleness;
}

inline core::u32 lodUpdateInterval(float distanceSq, float nearRadiusSq, core::u32 farInterval) noexcept
{
    if (nearRadiusSq <= 0.0f || distanceSq <= nearRadiusSq)
        return 1;
    return farInterval < 1 ? 1 : farInterval;
}

} // namespace lpl::net::relevancy

#endif // LPL_NET_RELEVANCY_RELEVANCY_INL
