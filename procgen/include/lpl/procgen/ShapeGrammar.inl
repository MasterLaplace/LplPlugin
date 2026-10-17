#ifndef LPL_PROCGEN_SHAPEGRAMMAR_INL
#define LPL_PROCGEN_SHAPEGRAMMAR_INL

namespace lpl::procgen {

constexpr core::u8 materialForSymbol(char letter) noexcept
{
    if (letter >= 'A' && letter <= 'Z')
        return static_cast<core::u8>(letter - 'A' + 1);
    if (letter >= 'a' && letter <= 'z')
        return static_cast<core::u8>(letter - 'a' + 1);
    return 0u;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_SHAPEGRAMMAR_INL
