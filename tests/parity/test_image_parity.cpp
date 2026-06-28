/**
 * @file test_image_parity.cpp
 * @brief Parity test: deterministic integer color/image operations.
 *
 * Verifies the lpl::image color-space conversions, luminance, histogram and
 * bilinear sampling produce exact, reproducible integer results — the same on
 * the Linux oracle and (once wired) the freestanding kernel build.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/image/Codec.hpp>
#include <lpl/image/Image.hpp>
#include <lpl/image/Painter.hpp>

using namespace lpl;

static int failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

int main()
{
    std::printf("== image color/HSB ==\n");

    // Pack/unpack roundtrip.
    const image::Rgba c = image::packRgba(0x12u, 0x34u, 0x56u, 0x78u);
    check(image::redOf(c) == 0x12u && image::greenOf(c) == 0x34u && image::blueOf(c) == 0x56u &&
              image::alphaOf(c) == 0x78u,
          "packRgba / channel accessors roundtrip");

    // Primary hues.
    check(image::rgbToHsb(image::packRgba(255, 0, 0)).hue == 0u, "red hue == 0");
    check(image::rgbToHsb(image::packRgba(0, 255, 0)).hue == 120u, "green hue == 120");
    check(image::rgbToHsb(image::packRgba(0, 0, 255)).hue == 240u, "blue hue == 240");

    {
        const image::Hsb red = image::rgbToHsb(image::packRgba(255, 0, 0));
        check(red.saturation == 255u && red.brightness == 255u, "red sat/brightness saturated");
    }

    // Achromatic gray: saturation 0, exact RGB roundtrip.
    {
        const image::Hsb gray = image::rgbToHsb(image::packRgba(128, 128, 128));
        check(gray.saturation == 0u && gray.brightness == 128u, "gray sat==0 brightness==128");
        const image::Rgba back = image::hsbToRgb(gray);
        check(image::redOf(back) == 128u && image::greenOf(back) == 128u && image::blueOf(back) == 128u,
              "gray RGB->HSB->RGB exact");
    }

    // Primary roundtrip through HSB.
    check(image::hsbToRgb(image::rgbToHsb(image::packRgba(255, 0, 0))) == image::packRgba(255, 0, 0),
          "red RGB->HSB->RGB exact");

    // Luminance bounds.
    check(image::luminanceOf(image::packRgba(255, 255, 255)) == 255u, "white luminance == 255");
    check(image::luminanceOf(image::packRgba(0, 0, 0)) == 0u, "black luminance == 0");

    std::printf("== image histogram ==\n");
    {
        image::Image img(4u, 4u);
        img.fill(image::packRgba(255, 0, 0));
        const image::Histogram h = img.histogram();
        // luminance of pure red = round(0.299*255) = 76.
        check(h.red[255] == 16u && h.green[0] == 16u && h.blue[0] == 16u && h.luminance[76] == 16u,
              "16x red pixels histogram counts");
    }

    std::printf("== image bilinear ==\n");
    {
        image::Image img(2u, 2u);
        img.set(0, 0, image::packRgba(0, 0, 0));
        img.set(1, 0, image::packRgba(255, 0, 0));
        img.set(0, 1, image::packRgba(0, 255, 0));
        img.set(1, 1, image::packRgba(0, 0, 255));
        // Centre sample averages all four corners (per channel ~63/63/63).
        const image::Rgba mid = img.sampleBilinear(0x8000u, 0x8000u);
        const auto near = [](core::u8 v, core::u8 t) { return static_cast<core::u32>(v > t ? v - t : t - v) <= 2u; };
        check(near(image::redOf(mid), 63u) && near(image::greenOf(mid), 63u) && near(image::blueOf(mid), 63u),
              "centre bilinear ~= corner average");
        // Top-left corner sample returns the top-left texel exactly.
        check(img.sampleNearest(0u, 0u) == image::packRgba(0, 0, 0), "nearest top-left exact");
    }

    std::printf("== painter primitives ==\n");
    {
        const image::Rgba red = image::packRgba(255, 0, 0);
        const image::Rgba blue = image::packRgba(0, 0, 255);

        image::Image img(16u, 16u);
        image::Painter painter(img);

        painter.fillRect(2, 2, 4, 4, red);
        check(img.at(2, 2) == red && img.at(5, 5) == red && img.at(6, 6) != red, "fillRect covers [2,6)x[2,6) only");

        painter.drawLine(0, 0, 15, 15, blue);
        check(img.at(0, 0) == blue && img.at(7, 7) == blue && img.at(15, 15) == blue,
              "drawLine diagonal hits endpoints + midpoint");

        image::Image disc(11u, 11u);
        image::Painter discPainter(disc);
        discPainter.fillCircle(5, 5, 4, red);
        check(disc.at(5, 5) == red && disc.at(1, 5) == red && disc.at(9, 5) == red && disc.at(0, 0) != red,
              "fillCircle fills centre + horizontal extent, not corners");

        image::Image outline(11u, 11u);
        image::Painter outlinePainter(outline);
        outlinePainter.drawCircle(5, 5, 4, blue);
        check(outline.at(5, 5) != blue && outline.at(9, 5) == blue && outline.at(1, 5) == blue,
              "drawCircle is an outline (centre empty, rim set)");

        // Blit a 3x3 opaque green patch; transparent source leaves dst intact.
        image::Image patch(3u, 3u);
        patch.fill(image::packRgba(0, 255, 0));
        image::Image canvas(8u, 8u);
        image::Painter canvasPainter(canvas);
        canvasPainter.blit(patch, 2, 2);
        check(canvas.at(2, 2) == image::packRgba(0, 255, 0) && canvas.at(4, 4) == image::packRgba(0, 255, 0) &&
                  canvas.at(0, 0) == 0u,
              "blit copies opaque patch, leaves rest untouched");

        // Alpha blend: 50% red over opaque blue -> ~ (127,0,127).
        image::Image blend(2u, 2u);
        blend.fill(image::packRgba(0, 0, 255));
        image::Painter blendPainter(blend);
        blendPainter.blendPixel(0, 0, image::packRgba(255, 0, 0, 128));
        const image::Rgba mixed = blend.at(0, 0);
        const auto near = [](core::u8 v, core::u8 t) { return static_cast<core::u32>(v > t ? v - t : t - v) <= 2u; };
        check(near(image::redOf(mixed), 128u) && image::greenOf(mixed) == 0u && near(image::blueOf(mixed), 127u),
              "blendPixel 50% red over blue ~= (128,0,127)");
    }

    std::printf("== ppm codec ==\n");
    {
        // Round-trip: paint a scene, encode to PPM, decode, compare (RGB only).
        image::Image original(24u, 16u);
        image::paintParityScene(original);
        pmr::vector<core::u8> encoded;
        check(image::writePpm(original, encoded), "writePpm succeeds");

        image::Image decoded;
        check(image::readPpm(encoded.data(), encoded.size(), decoded), "readPpm succeeds");
        check(decoded.width() == 24u && decoded.height() == 16u, "decoded dimensions match");

        bool same = true;
        for (core::u32 y = 0u; y < 16u && same; ++y)
            for (core::u32 x = 0u; x < 24u && same; ++x)
                same = ((original.at(x, y) & 0x00FFFFFFu) == (decoded.at(x, y) & 0x00FFFFFFu));
        check(same, "PPM round-trip preserves RGB pixels");

        // Parse a hand-built 1x1 red PPM (with a comment line).
        const core::u8 redPpm[] = {'P',  '6', '\n', '#', 'c',  '\n', '1', ' ', '1',
                                   '\n', '2', '5',  '5', '\n', 255u, 0u,  0u};
        image::Image one;
        check(image::readPpm(redPpm, sizeof(redPpm), one) &&
                  ((one.at(0, 0) & 0x00FFFFFFu) == (image::packRgba(255, 0, 0) & 0x00FFFFFFu)),
              "readPpm parses 1x1 red with comment");
    }

    // Cross-target signature: this must equal the kernel P4 image smoke's
    // paint_sig field (both fold the same paintParityScene via integer rasterisers).
    {
        image::Image scene(32u, 32u);
        image::paintParityScene(scene);
        std::printf("== painter parity signature == 0x%08X (compare with kernel paint_sig)\n",
                    image::foldSignature(scene));

        pmr::vector<core::u8> encoded;
        image::Image decoded;
        if (image::writePpm(scene, encoded) && image::readPpm(encoded.data(), encoded.size(), decoded))
            std::printf("== ppm round-trip signature == 0x%08X (compare with kernel ppm_sig)\n",
                        image::foldSignature(decoded));
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
