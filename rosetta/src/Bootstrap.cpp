#include <lpl/rosetta/Bootstrap.hpp>

#include <lpl/rosetta/SelfDescribing.hpp>

namespace lpl::rosetta {

namespace {

/**
 * @brief Appends a NUL-terminated literal to a byte vector.
 * @param out  Destination.
 * @param text What to append, terminator included.
 */
void appendText(lpl::pmr::vector<core::u8> &out, const char *text)
{
    for (core::u32 i = 0u; text[i] != '\0'; ++i)
        out.push_back(static_cast<core::u8>(text[i]));
    out.push_back(core::u8{0});
}

} // namespace

core::u32 Bootstrap::totalBytes() const noexcept
{
    core::u32 total = 0u;
    for (core::u32 i = 0u; i < static_cast<core::u32>(BootstrapLevel::Count) - 1u; ++i)
        total += static_cast<core::u32>(level[i].size());
    return total;
}

Bootstrap standardBootstrap()
{
    Bootstrap bootstrap;

    appendText(bootstrap.level[0], "H1:1420405751.768");
    appendText(bootstrap.level[0], "ALPHA-1:137.035999");
    appendText(bootstrap.level[0], "MP/ME:1836.152673");

    appendText(bootstrap.level[1], "CELL:8H1");
    appendText(bootstrap.level[1], "ROWS:MSB-FIRST");
    appendText(bootstrap.level[1], "ORIGIN:TOP-LEFT");

    appendText(bootstrap.level[2], "U8:8BIT-UNSIGNED");
    appendText(bootstrap.level[2], "U32:4xU8-LITTLE-ENDIAN");
    appendText(bootstrap.level[2], "TEXT:U8-NUL-TERMINATED");

    bootstrap.level[3].resize(kSpecificationBytes, core::u8{0});
    const core::u32 written = emitSpecification(bootstrap.level[3].data(), kSpecificationBytes);
    bootstrap.level[3].resize(written, core::u8{0});

    return bootstrap;
}

} // namespace lpl::rosetta
