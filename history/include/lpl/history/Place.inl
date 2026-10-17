#ifndef LPL_HISTORY_PLACE_INL
#define LPL_HISTORY_PLACE_INL

namespace lpl::history {

inline core::u32 IPlaceResolver::linkedPlaces(core::u32 id, core::u32 *out, core::u32 capacity) const
{
    (void) id;
    (void) out;
    (void) capacity;
    return 0u;
}

constexpr bool existsInYear(const Place &place, core::i32 year) noexcept
{
    if (place.minYear == 0 && place.maxYear == 0)
        return true;
    return year >= place.minYear && year <= place.maxYear;
}

inline core::u32 IRouteResolver::route(core::u32 fromPlace, core::u32 toPlace, RouteLeg *out, core::u32 capacity) const
{
    (void) fromPlace;
    (void) toPlace;
    (void) out;
    (void) capacity;
    return 0u;
}

} // namespace lpl::history

#endif // LPL_HISTORY_PLACE_INL
