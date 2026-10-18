#ifndef LPL_EDITOR_JSON_INL
#define LPL_EDITOR_JSON_INL

namespace lpl::editor::detail {

inline const JVal *JVal::find(std::string_view key) const
{
    for (const auto &kv : obj)
        if (kv.first == key)
            return &kv.second;
    return nullptr;
}

inline double JVal::numOr(std::string_view key, double fallback) const
{
    const JVal *v = find(key);
    return (v != nullptr && v->t == T::Num) ? v->num : fallback;
}

} // namespace lpl::editor::detail

#endif // LPL_EDITOR_JSON_INL
