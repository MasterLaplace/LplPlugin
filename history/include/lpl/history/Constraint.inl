#ifndef LPL_HISTORY_CONSTRAINT_INL
#define LPL_HISTORY_CONSTRAINT_INL

namespace lpl::history {

constexpr const char *constraintKindName(ConstraintKind kind) noexcept
{
    switch (kind)
    {
    case ConstraintKind::Seed: return "seed";
    case ConstraintKind::Force: return "force";
    case ConstraintKind::Score: return "score";
    case ConstraintKind::Count: break;
    }
    return "score";
}

} // namespace lpl::history

#endif // LPL_HISTORY_CONSTRAINT_INL
