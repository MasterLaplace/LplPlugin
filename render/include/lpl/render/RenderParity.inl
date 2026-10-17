#ifndef LPL_RENDER_RENDERPARITY_INL
#define LPL_RENDER_RENDERPARITY_INL

namespace lpl::render {

namespace detail {

inline core::u32 fnv1aStep(core::u32 hash, core::u32 value) noexcept
{
    for (core::u32 i = 0; i < 4u; ++i)
    {
        hash ^= (value >> (i * 8u)) & 0xFFu;
        hash *= kFnv1aPrime;
    }
    return hash;
}

} // namespace detail

} // namespace lpl::render

#endif // LPL_RENDER_RENDERPARITY_INL
