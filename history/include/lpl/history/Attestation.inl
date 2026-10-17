#ifndef LPL_HISTORY_ATTESTATION_INL
#define LPL_HISTORY_ATTESTATION_INL

namespace lpl::history {

constexpr const char *causeName(Cause cause) noexcept
{
    switch (cause)
    {
    case Cause::Constraint: return "constraint";
    case Cause::Emergent: return "emergent";
    case Cause::Sovereign: return "sovereign";
    case Cause::Count: break;
    }
    return "?";
}

} // namespace lpl::history

#endif // LPL_HISTORY_ATTESTATION_INL
