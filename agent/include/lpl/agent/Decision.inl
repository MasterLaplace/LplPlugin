#ifndef LPL_AGENT_DECISION_INL
#define LPL_AGENT_DECISION_INL

namespace lpl::agent {

constexpr bool alphabetOffers(const char *alphabet, core::u32 bytes, const char *name, core::u32 nameBytes) noexcept
{
    if (alphabet == nullptr || name == nullptr || nameBytes == 0u)
        return false;

    core::u32 cursor = 0u;
    while (cursor < bytes)
    {
        while (cursor < bytes && alphabet[cursor] == ' ')
            ++cursor;
        const core::u32 start = cursor;
        while (cursor < bytes && alphabet[cursor] != ' ')
            ++cursor;
        if (cursor - start == nameBytes)
        {
            bool match = true;
            for (core::u32 i = 0u; i < nameBytes && match; ++i)
                match = alphabet[start + i] == name[i];
            if (match)
                return true;
        }
        if (cursor == start)
            break;
    }
    return false;
}

} // namespace lpl::agent

#endif // LPL_AGENT_DECISION_INL
