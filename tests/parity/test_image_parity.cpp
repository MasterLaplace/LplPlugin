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
#include <lpl/image/Image.hpp>

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
        const auto near = [](core::u8 v, core::u8 t) {
            return static_cast<core::u32>(v > t ? v - t : t - v) <= 2u;
        };
        check(near(image::redOf(mid), 63u) && near(image::greenOf(mid), 63u) && near(image::blueOf(mid), 63u),
              "centre bilinear ~= corner average");
        // Top-left corner sample returns the top-left texel exactly.
        check(img.sampleNearest(0u, 0u) == image::packRgba(0, 0, 0), "nearest top-left exact");
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
