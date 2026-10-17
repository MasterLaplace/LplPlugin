#ifndef LPL_HISTORY_PREDICATE_INL
#define LPL_HISTORY_PREDICATE_INL

namespace lpl::history {

constexpr bool isDeed(Predicate predicate) noexcept
{
    return predicate == Predicate::BornAt || predicate == Predicate::DweltAt || predicate == Predicate::TravelledTo ||
           predicate == Predicate::DiedAt || predicate == Predicate::Founded || predicate == Predicate::Abandoned;
}

constexpr bool objectIsPlace(Predicate predicate) noexcept
{
    return predicate == Predicate::BornAt || predicate == Predicate::DweltAt || predicate == Predicate::TravelledTo ||
           predicate == Predicate::DiedAt || predicate == Predicate::Founded || predicate == Predicate::Abandoned ||
           predicate == Predicate::Ruled;
}

constexpr const char *predicateName(Predicate predicate) noexcept
{
    switch (predicate)
    {
    case Predicate::DiedOf: return "died-of";
    case Predicate::Exists: return "exists";
    case Predicate::BornIn: return "born-in";
    case Predicate::DiedIn: return "died-in";
    case Predicate::Ruled: return "ruled";
    case Predicate::Wrote: return "wrote";
    case Predicate::BornAt: return "born-at";
    case Predicate::DweltAt: return "dwelt-at";
    case Predicate::TravelledTo: return "travelled-to";
    case Predicate::DiedAt: return "died-at";
    case Predicate::Founded: return "founded";
    case Predicate::Abandoned: return "abandoned";
    case Predicate::None:
    case Predicate::Count:
    default: return "unknown";
    }
}

constexpr bool predicateByName(const char *text, core::u32 bytes, Predicate &out) noexcept
{
    constexpr Predicate kAll[] = {Predicate::DiedOf,      Predicate::Exists, Predicate::BornIn,  Predicate::DiedIn,
                                  Predicate::Ruled,       Predicate::Wrote,  Predicate::BornAt,  Predicate::DweltAt,
                                  Predicate::TravelledTo, Predicate::DiedAt, Predicate::Founded, Predicate::Abandoned};
    for (const Predicate candidate : kAll)
    {
        const char *name = predicateName(candidate);
        core::u32 i = 0u;
        while (i < bytes && name[i] != '\0' && name[i] == text[i])
            ++i;
        if (i == bytes && name[i] == '\0')
        {
            out = candidate;
            return true;
        }
    }
    return false;
}

} // namespace lpl::history

#endif // LPL_HISTORY_PREDICATE_INL
