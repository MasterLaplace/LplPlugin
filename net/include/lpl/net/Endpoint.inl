#ifndef LPL_NET_ENDPOINT_INL
#define LPL_NET_ENDPOINT_INL

namespace lpl::net {

constexpr Endpoint Endpoint::fromOctets(core::u8 a, core::u8 b, core::u8 c, core::u8 d, core::u16 port) noexcept
{
    return Endpoint{(static_cast<core::u32>(a) << 24) | (static_cast<core::u32>(b) << 16) |
                        (static_cast<core::u32>(c) << 8) | static_cast<core::u32>(d),
                    port};
}

constexpr bool Endpoint::parse(const char *text, core::u16 port, Endpoint &outEndpoint) noexcept
{
    if (text == nullptr)
        return false;

    core::u32 address = 0;
    core::u32 octet = 0;
    core::u32 digits = 0;
    core::u32 octetsSeen = 0;

    for (const char *cursor = text;; ++cursor)
    {
        const char character = *cursor;

        if (character >= '0' && character <= '9')
        {
            if (++digits > 3)
                return false;
            octet = octet * 10 + static_cast<core::u32>(character - '0');
            if (octet > 255)
                return false;
            continue;
        }

        if (character != '.' && character != '\0')
            return false;
        if (digits == 0)
            return false;

        address = (address << 8) | octet;
        ++octetsSeen;
        octet = 0;
        digits = 0;

        if (character == '\0')
            break;
        if (octetsSeen == 4)
            return false;
    }

    if (octetsSeen != 4)
        return false;

    outEndpoint = Endpoint{address, port};
    return true;
}

} // namespace lpl::net

#endif // LPL_NET_ENDPOINT_INL
