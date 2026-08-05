/**
 * @file Bootstrap.cpp
 * @brief The four levels a reader needs before the payload means anything.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

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

    // Level 0 — calibration. Ratios, not units: a metre is a convention and the fine
    // structure constant is not. A finder checks their instrument against the plate.
    appendText(bootstrap.level[0], "H1:1420405751.768");
    appendText(bootstrap.level[0], "ALPHA-1:137.035999");
    appendText(bootstrap.level[0], "MP/ME:1836.152673");

    // Level 1 — read-out. How a mark becomes a bit, in the units level 0 established.
    appendText(bootstrap.level[1], "CELL:8H1");
    appendText(bootstrap.level[1], "ROWS:MSB-FIRST");
    appendText(bootstrap.level[1], "ORIGIN:TOP-LEFT");

    // Level 2 — primitives. The few types everything below is written in. Kept to
    // three because every entry is something a reader has to be convinced of.
    appendText(bootstrap.level[2], "U8:8BIT-UNSIGNED");
    appendText(bootstrap.level[2], "U32:4xU8-LITTLE-ENDIAN");
    appendText(bootstrap.level[2], "TEXT:U8-NUL-TERMINATED");

    // Level 3 — the machine, as bytes an interpreter is rebuilt from. NOT a prose
    // restatement: a description nothing can execute is a description nothing can
    // check.
    bootstrap.level[3].resize(kSpecificationBytes, core::u8{0});
    const core::u32 written = emitSpecification(bootstrap.level[3].data(), kSpecificationBytes);
    bootstrap.level[3].resize(written, core::u8{0});

    return bootstrap;
}

} // namespace lpl::rosetta
