#include <lpl/image/Codec.hpp>
#include <lpl/image/Image.hpp>
#include <lpl/image/Painter.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(image);

namespace {

[[nodiscard]] bool isNear(lpl::core::u8 value, lpl::core::u8 target)
{
    return static_cast<lpl::core::u32>(value > target ? value - target : target - value) <= 2u;
}

[[nodiscard]] bool sameColour(lpl::image::Rgba lhs, lpl::image::Rgba rhs)
{
    return (lhs & 0x00FFFFFFu) == (rhs & 0x00FFFFFFu);
}

} // namespace

LPL_TEST(channels_pack_and_unpack)
{
    const lpl::image::Rgba colour = lpl::image::packRgba(0x12u, 0x34u, 0x56u, 0x78u);

    test.check(lpl::image::redOf(colour) == 0x12u && lpl::image::greenOf(colour) == 0x34u &&
                   lpl::image::blueOf(colour) == 0x56u && lpl::image::alphaOf(colour) == 0x78u,
               "each channel reads back what was packed");
}

/**
 * @brief Gate P4 image, colour half: the primaries land on their hues and convert back exactly.
 */
LPL_TEST(primaries_and_gray_convert_exactly)
{
    const lpl::image::Rgba red = lpl::image::packRgba(255, 0, 0);
    const lpl::image::Hsb redHsb = lpl::image::rgbToHsb(red);
    const lpl::image::Hsb grayHsb = lpl::image::rgbToHsb(lpl::image::packRgba(128, 128, 128));
    const lpl::image::Rgba grayBack = lpl::image::hsbToRgb(grayHsb);

    test.check(redHsb.hue == 0u, "red is at hue 0");
    test.check(lpl::image::rgbToHsb(lpl::image::packRgba(0, 255, 0)).hue == 120u, "green at 120");
    test.check(lpl::image::rgbToHsb(lpl::image::packRgba(0, 0, 255)).hue == 240u, "blue at 240");
    test.check(redHsb.saturation == 255u && redHsb.brightness == 255u, "red is fully saturated and bright");
    test.check(grayHsb.saturation == 0u && grayHsb.brightness == 128u, "gray has no saturation");
    test.check(lpl::image::redOf(grayBack) == 128u && lpl::image::greenOf(grayBack) == 128u &&
                   lpl::image::blueOf(grayBack) == 128u,
               "gray converts back exactly");
    test.check(lpl::image::hsbToRgb(redHsb) == red, "and so does red");
}

LPL_TEST(luminance_spans_black_to_white)
{
    test.check(lpl::image::luminanceOf(lpl::image::packRgba(255, 255, 255)) == 255u, "white has luminance 255");
    test.check(lpl::image::luminanceOf(lpl::image::packRgba(0, 0, 0)) == 0u, "black has luminance 0");
}

LPL_TEST(histogram_counts_every_pixel)
{
    lpl::image::Image picture(4u, 4u);

    picture.fill(lpl::image::packRgba(255, 0, 0));

    const lpl::image::Histogram histogram = picture.histogram();

    test.check(histogram.red[255] == 16u && histogram.green[0] == 16u && histogram.blue[0] == 16u,
               "sixteen red pixels count sixteen times in each channel");
    test.check(histogram.luminance[76] == 16u, "at the luminance of red, 0.299 of 255 rounded");
}

/**
 * @brief Gate P4 image, sampling half: the centre of a 2x2 gradient averages its corners.
 */
LPL_TEST(bilinear_centre_averages_the_corners)
{
    lpl::image::Image gradient(2u, 2u);

    gradient.set(0, 0, lpl::image::packRgba(0, 0, 0));
    gradient.set(1, 0, lpl::image::packRgba(255, 0, 0));
    gradient.set(0, 1, lpl::image::packRgba(0, 255, 0));
    gradient.set(1, 1, lpl::image::packRgba(0, 0, 255));

    const lpl::image::Rgba centre = gradient.sampleBilinear(0x8000u, 0x8000u);

    test.check(isNear(lpl::image::redOf(centre), 63u) && isNear(lpl::image::greenOf(centre), 63u) &&
                   isNear(lpl::image::blueOf(centre), 63u),
               "the centre is the average of the four corners");
    test.check(gradient.sampleNearest(0u, 0u) == lpl::image::packRgba(0, 0, 0), "the nearest sample of a corner is it");

    test.measureHexadecimal("centre_colour", centre & 0x00FFFFFFu);
}

LPL_TEST(painter_primitives_cover_what_they_say)
{
    const lpl::image::Rgba red = lpl::image::packRgba(255, 0, 0);
    const lpl::image::Rgba green = lpl::image::packRgba(0, 255, 0);
    const lpl::image::Rgba blue = lpl::image::packRgba(0, 0, 255);
    lpl::image::Image canvas(16u, 16u);
    lpl::image::Painter painter(canvas);

    painter.fillRect(2, 2, 4, 4, red);
    test.check(canvas.at(2, 2) == red && canvas.at(5, 5) == red && canvas.at(6, 6) != red,
               "a rectangle covers [2, 6) on both axes and no more");

    painter.drawLine(0, 0, 15, 15, blue);
    test.check(canvas.at(0, 0) == blue && canvas.at(7, 7) == blue && canvas.at(15, 15) == blue,
               "a diagonal line covers both ends and its middle");

    lpl::image::Image disc(11u, 11u);
    lpl::image::Painter discPainter(disc);

    discPainter.fillCircle(5, 5, 4, red);
    test.check(disc.at(5, 5) == red && disc.at(1, 5) == red && disc.at(9, 5) == red && disc.at(0, 0) != red,
               "a filled circle covers its centre and its width, not the corners");

    lpl::image::Image ring(11u, 11u);
    lpl::image::Painter ringPainter(ring);

    ringPainter.drawCircle(5, 5, 4, blue);
    test.check(ring.at(5, 5) != blue && ring.at(9, 5) == blue && ring.at(1, 5) == blue, "a drawn circle is an outline");

    lpl::image::Image patch(3u, 3u);
    lpl::image::Image target(8u, 8u);
    lpl::image::Painter targetPainter(target);

    patch.fill(green);
    targetPainter.blit(patch, 2, 2);
    test.check(target.at(2, 2) == green && target.at(4, 4) == green && target.at(0, 0) == 0u,
               "a blit copies an opaque patch and leaves the rest");

    lpl::image::Image blend(2u, 2u);
    lpl::image::Painter blendPainter(blend);

    blend.fill(blue);
    blendPainter.blendPixel(0, 0, lpl::image::packRgba(255, 0, 0, 128));

    const lpl::image::Rgba mixed = blend.at(0, 0);

    test.check(isNear(lpl::image::redOf(mixed), 128u) && lpl::image::greenOf(mixed) == 0u &&
                   isNear(lpl::image::blueOf(mixed), 127u),
               "half red over blue blends to half of each");
}

LPL_TEST(ppm_round_trip_keeps_the_pixels)
{
    lpl::image::Image original(24u, 16u);
    lpl::pmr::vector<lpl::core::u8> encoded;
    lpl::image::Image decoded;

    lpl::image::paintParityScene(original);
    if (!test.check(lpl::image::writePpm(original, encoded), "the picture encodes") ||
        !test.check(lpl::image::readPpm(encoded.data(), encoded.size(), decoded), "and decodes"))
        return;
    test.check(decoded.width() == 24u && decoded.height() == 16u, "with its size");

    bool same = true;

    for (lpl::core::u32 y = 0u; y < 16u; ++y)
    {
        for (lpl::core::u32 x = 0u; x < 24u; ++x)
            same = same && sameColour(original.at(x, y), decoded.at(x, y));
    }
    test.check(same, "and every pixel's colour");
}

LPL_TEST(ppm_reader_skips_a_comment)
{
    const lpl::core::u8 redPixel[] = {'P',  '6', '\n', '#', 'c',  '\n', '1', ' ', '1',
                                      '\n', '2', '5',  '5', '\n', 255u, 0u,  0u};
    lpl::image::Image one;

    test.check(lpl::image::readPpm(redPixel, sizeof(redPixel), one) &&
                   sameColour(one.at(0, 0), lpl::image::packRgba(255, 0, 0)),
               "a one-pixel red PPM with a comment line reads as red");
}

/**
 * @brief Gate P4 image: the parity scene, painted with integer rasterisers, folds the same on both
 *        targets, before and after a PPM round trip.
 */
LPL_TEST(parity_scene_folds_the_same)
{
    lpl::image::Image scene(32u, 32u);
    lpl::pmr::vector<lpl::core::u8> encoded;
    lpl::image::Image decoded;

    lpl::image::paintParityScene(scene);

    const lpl::core::u32 painted = lpl::image::foldSignature(scene);

    if (!test.check(lpl::image::writePpm(scene, encoded) &&
                        lpl::image::readPpm(encoded.data(), encoded.size(), decoded),
                    "the scene goes through PPM"))
        return;

    const lpl::core::u32 roundTripped = lpl::image::foldSignature(decoded);

    test.check(painted == roundTripped, "the scene is opaque, so the round trip gives it back unchanged");

    test.measureHexadecimal("painter_signature", painted);
    test.measureHexadecimal("ppm_signature", roundTripped);
}
