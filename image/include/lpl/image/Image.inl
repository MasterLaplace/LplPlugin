#ifndef LPL_IMAGE_IMAGE_INL
#define LPL_IMAGE_IMAGE_INL

namespace lpl::image {

inline void Image::resize(core::u32 width, core::u32 height)
{
    _width = width;
    _height = height;
    _pixels.clear();
    _pixels.resize(static_cast<core::usize>(width) * height, 0u);
}

inline void Image::fill(Rgba color)
{
    for (auto &pixel : _pixels)
        pixel = color;
}

inline Rgba Image::at(core::i32 x, core::i32 y) const noexcept
{
    if (_width == 0u || _height == 0u)
        return 0u;
    const core::u32 cx = clampCoord(x, _width);
    const core::u32 cy = clampCoord(y, _height);
    return _pixels[static_cast<core::usize>(cy) * _width + cx];
}

inline void Image::set(core::i32 x, core::i32 y, Rgba color) noexcept
{
    if (x < 0 || y < 0 || static_cast<core::u32>(x) >= _width || static_cast<core::u32>(y) >= _height)
        return;
    _pixels[static_cast<core::usize>(y) * _width + static_cast<core::u32>(x)] = color;
}

inline Rgba Image::sampleNearest(core::u32 u, core::u32 v) const noexcept
{
    const core::i32 x = static_cast<core::i32>((static_cast<core::u64>(u) * _width) >> 16);
    const core::i32 y = static_cast<core::i32>((static_cast<core::u64>(v) * _height) >> 16);
    return at(x, y);
}

inline core::u32 Image::clampCoord(core::i32 v, core::u32 size) noexcept
{
    if (v < 0)
        return 0u;
    if (static_cast<core::u32>(v) >= size)
        return size - 1u;
    return static_cast<core::u32>(v);
}

} // namespace lpl::image

#endif // LPL_IMAGE_IMAGE_INL
