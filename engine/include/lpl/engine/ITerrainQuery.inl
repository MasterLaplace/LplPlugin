#ifndef LPL_ENGINE_ITERRAINQUERY_INL
#define LPL_ENGINE_ITERRAINQUERY_INL

namespace lpl::engine {

inline bool ITerrainQuery::recoveryDirection(math::Fixed32 x, math::Fixed32 z, math::Fixed32 &outX,
                                             math::Fixed32 &outZ) const
{
    (void) x;
    (void) z;
    (void) outX;
    (void) outZ;
    return false;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_ITERRAINQUERY_INL
