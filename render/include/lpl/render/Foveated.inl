#ifndef LPL_RENDER_FOVEATED_INL
#define LPL_RENDER_FOVEATED_INL

namespace lpl::render {

namespace detail {

inline core::u32 shadeRateForRing(core::u32 ring) noexcept
{
    if (ring == 0u)
        return 1u;
    if (ring == 1u)
        return 2u;
    return 4u;
}

} // namespace detail

inline FoveatedResult foveatedShade(core::u32 *color, core::u32 width, core::u32 height, core::u32 gazeX,
                                    core::u32 gazeY)
{
    FoveatedResult out{};
    out.width = width;
    out.height = height;
    out.full_fragments = width * height;
    if (color == nullptr || width == 0u || height == 0u)
        return out;

    constexpr core::u32 kTile = 8u;
    const core::u32 gazeTileX = gazeX / kTile;
    const core::u32 gazeTileY = gazeY / kTile;

    const auto shadePixel = [&](core::u32 x, core::u32 y) -> core::u32 {
        const core::f32 fx = static_cast<core::f32>(x) / static_cast<core::f32>(width);
        const core::f32 fy = static_cast<core::f32>(y) / static_cast<core::f32>(height);
        const core::u32 r = static_cast<core::u32>(detail::saturate(fx) * 255.0f + 0.5f);
        const core::u32 g = static_cast<core::u32>(detail::saturate(fy) * 255.0f + 0.5f);
        const core::u32 b = static_cast<core::u32>(detail::saturate(1.0f - fx * fy) * 255.0f + 0.5f);
        return (r << 16) | (g << 8) | b;
    };

    for (core::u32 ty = 0; ty * kTile < height; ++ty)
        for (core::u32 tx = 0; tx * kTile < width; ++tx)
        {
            const core::u32 dx = tx > gazeTileX ? tx - gazeTileX : gazeTileX - tx;
            const core::u32 dy = ty > gazeTileY ? ty - gazeTileY : gazeTileY - ty;
            const core::u32 ring = dx > dy ? dx : dy;
            const core::u32 rate = detail::shadeRateForRing(ring);

            const core::u32 x0 = tx * kTile;
            const core::u32 y0 = ty * kTile;
            for (core::u32 cy = 0; cy < kTile && (y0 + cy) < height; cy += rate)
                for (core::u32 cx = 0; cx < kTile && (x0 + cx) < width; cx += rate)
                {
                    const core::u32 rgb = shadePixel(x0 + cx, y0 + cy);
                    ++out.shaded_fragments;
                    for (core::u32 fy = 0; fy < rate && (y0 + cy + fy) < height; ++fy)
                        for (core::u32 fx = 0; fx < rate && (x0 + cx + fx) < width; ++fx)
                            color[static_cast<core::usize>(y0 + cy + fy) * width + (x0 + cx + fx)] = rgb;
                }
        }

    core::u32 hash = detail::kFnv1aOffsetBasis;
    for (core::u32 i = 0; i < width * height; ++i)
        hash = detail::fnv1aStep(hash, color[i]);
    out.image_signature = hash;
    return out;
}

} // namespace lpl::render

#endif // LPL_RENDER_FOVEATED_INL
