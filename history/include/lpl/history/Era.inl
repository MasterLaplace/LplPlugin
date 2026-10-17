#ifndef LPL_HISTORY_ERA_INL
#define LPL_HISTORY_ERA_INL

namespace lpl::history {

constexpr Era Era::ofYears(core::i32 firstYear, core::i32 lastYear, core::u32 ticksPerYear) noexcept
{
    Era era;
    era.startDay = firstDayOfYear(firstYear);
    era.endDay = lastDayOfYear(lastYear);
    era.daysPerTick = ticksPerYear == 0u ? 365u : (365u / ticksPerYear == 0u ? 1u : 365u / ticksPerYear);
    return era;
}

constexpr core::u32 Era::totalTicks() const noexcept
{
    const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
    return (days() + step - 1u) / step;
}

constexpr core::i32 Era::dayOfTick(core::u32 tick) const noexcept
{
    const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
    const core::i64 day = static_cast<core::i64>(startDay) + static_cast<core::i64>(tick) * step;
    return day > endDay ? endDay : static_cast<core::i32>(day);
}

constexpr void Era::spanOfTick(core::u32 tick, core::i32 &outFrom, core::i32 &outTo) const noexcept
{
    const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
    const core::i64 from = static_cast<core::i64>(startDay) + static_cast<core::i64>(tick) * step;
    const core::i64 to = from + static_cast<core::i64>(step) - 1;
    outFrom = from > endDay ? endDay : static_cast<core::i32>(from);
    outTo = to > endDay ? endDay : static_cast<core::i32>(to);
}

constexpr core::u32 Era::firstTickOfDay(core::i32 day) const noexcept
{
    const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
    const core::i32 clamped = day < startDay ? startDay : (day > endDay ? endDay : day);
    return static_cast<core::u32>(clamped - startDay) / step;
}

} // namespace lpl::history

#endif // LPL_HISTORY_ERA_INL
