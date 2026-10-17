#ifndef LPL_HISTORY_CALENDAR_INL
#define LPL_HISTORY_CALENDAR_INL

namespace lpl::history {

constexpr core::i32 dayOfDate(core::i32 year, core::u32 month, core::u32 day) noexcept
{
    const core::i32 shifted = year - (month <= 2u ? 1 : 0);
    const core::i32 era = (shifted >= 0 ? shifted : shifted - 399) / 400;
    const core::u32 yearOfEra = static_cast<core::u32>(shifted - era * 400);
    const core::u32 dayOfYear = (153u * (month + (month > 2u ? -3u : 9u)) + 2u) / 5u + day - 1u;
    const core::u32 dayOfEra = yearOfEra * 365u + yearOfEra / 4u - yearOfEra / 100u + dayOfYear;
    return era * 146097 + static_cast<core::i32>(dayOfEra) - kEraShift;
}

constexpr core::i32 yearOfDay(core::i32 day) noexcept
{
    const core::i32 shifted = day + kEraShift;
    const core::i32 era = (shifted >= 0 ? shifted : shifted - 146096) / 146097;
    const core::u32 dayOfEra = static_cast<core::u32>(shifted - era * 146097);
    const core::u32 yearOfEra = (dayOfEra - dayOfEra / 1460u + dayOfEra / 36524u - dayOfEra / 146096u) / 365u;
    const core::i32 candidate = static_cast<core::i32>(yearOfEra) + era * 400;
    const core::u32 dayOfYear = dayOfEra - (365u * yearOfEra + yearOfEra / 4u - yearOfEra / 100u);
    const core::u32 shiftedMonth = (5u * dayOfYear + 2u) / 153u;
    return shiftedMonth >= 10u ? candidate + 1 : candidate;
}

} // namespace lpl::history

#endif // LPL_HISTORY_CALENDAR_INL
