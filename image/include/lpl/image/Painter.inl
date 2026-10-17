#ifndef LPL_IMAGE_PAINTER_INL
#define LPL_IMAGE_PAINTER_INL

namespace lpl::image {

inline core::u32 foldSignature(const Image &image) noexcept
{
    core::u32 acc = 2166136261u;
    for (core::u32 y = 0u; y < image.height(); ++y)
        for (core::u32 x = 0u; x < image.width(); ++x)
            acc = (acc ^ image.at(static_cast<core::i32>(x), static_cast<core::i32>(y))) * 16777619u;
    return acc;
}

inline void paintParityScene(Image &image) noexcept
{
    Painter painter(image);
    image.fill(packRgba(10, 20, 30));
    painter.fillRect(2, 2, 10, 6, packRgba(200, 40, 40));
    painter.drawLine(0, 0, static_cast<core::i32>(image.width()) - 1, static_cast<core::i32>(image.height()) - 1,
                     packRgba(0, 200, 0));
    painter.drawCircle(8, 8, 5, packRgba(0, 0, 220));
    painter.fillCircle(20, 20, 4, packRgba(255, 255, 0, 128));
}

} // namespace lpl::image

#endif // LPL_IMAGE_PAINTER_INL
