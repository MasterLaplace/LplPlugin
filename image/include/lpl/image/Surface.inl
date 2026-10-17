#ifndef LPL_IMAGE_SURFACE_INL
#define LPL_IMAGE_SURFACE_INL

namespace lpl::image {

inline void blitToFramebuffer(const Image &source, core::u32 *dst, core::u32 dstWidth, core::u32 dstHeight,
                              core::u32 dstPitchBytes, core::i32 dstX, core::i32 dstY) noexcept
{
    if (dst == nullptr || dstPitchBytes == 0u)
        return;

    for (core::u32 sy = 0u; sy < source.height(); ++sy)
    {
        const core::i32 py = dstY + static_cast<core::i32>(sy);
        if (py < 0 || static_cast<core::u32>(py) >= dstHeight)
            continue;
        core::u32 *row = reinterpret_cast<core::u32 *>(reinterpret_cast<core::u8 *>(dst) +
                                                       static_cast<core::usize>(py) * dstPitchBytes);
        for (core::u32 sx = 0u; sx < source.width(); ++sx)
        {
            const core::i32 px = dstX + static_cast<core::i32>(sx);
            if (px < 0 || static_cast<core::u32>(px) >= dstWidth)
                continue;
            row[px] = source.at(static_cast<core::i32>(sx), static_cast<core::i32>(sy));
        }
    }
}

} // namespace lpl::image

#endif // LPL_IMAGE_SURFACE_INL
