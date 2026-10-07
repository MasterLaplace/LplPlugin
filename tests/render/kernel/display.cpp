#include <lpl/image/Image.hpp>
#include <lpl/image/Painter.hpp>
#include <lpl/image/Surface.hpp>
#include <lpl/platform/kernel/KernelPlatform.hpp>
#include <lpl/render/SoftwareRasterizer.hpp>
#include <lpl/render/kernel/KernelDisplayRenderer.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(display);

namespace {

constexpr lpl::core::u32 kBackground = 0x00001040u;
constexpr lpl::core::u32 kPresentedWidth = 320u;
constexpr lpl::core::u32 kPresentedHeight = 240u;

/** The target that is presented, static because a kernel stack cannot hold it. */
lpl::core::u32 gPresentedColour[kPresentedWidth * kPresentedHeight];
lpl::core::f32 gPresentedDepth[kPresentedWidth * kPresentedHeight];

[[nodiscard]] lpl::core::u32 colourOf(lpl::core::u32 pixel) { return pixel & 0x00FFFFFFu; }

[[nodiscard]] bool hasSurface(lpl::platform::IDisplayBackend &display, lpl::platform::SurfaceDescriptor &surface)
{
    return display.querySurface(surface) && surface.buffer != nullptr && surface.width != 0u && surface.height != 0u;
}

[[nodiscard]] lpl::core::u32 pitchInBytes(const lpl::platform::SurfaceDescriptor &surface)
{
    return (surface.pitch != 0u) ? surface.pitch : surface.width * 4u;
}

void paintScene(lpl::image::Painter &painter, lpl::core::u32 width, lpl::core::u32 height)
{
    const lpl::core::i32 w = static_cast<lpl::core::i32>(width);
    const lpl::core::i32 h = static_cast<lpl::core::i32>(height);

    for (lpl::core::u32 y = 0u; y < height; ++y)
        painter.fillRect(0, static_cast<lpl::core::i32>(y), w, 1, lpl::image::packRgba(20, (y * 180u) / height, 90));
    painter.fillRect(40, 40, 160, 100, lpl::image::packRgba(220, 60, 60));
    painter.fillCircle(w / 2, h / 2, 70, lpl::image::packRgba(60, 200, 120, 200));
    painter.drawRect(40, 40, 160, 100, lpl::image::packRgba(255, 255, 255));
    painter.drawLine(0, 0, w - 1, h - 1, lpl::image::packRgba(255, 230, 0));
}

void upscaleOntoSurface(const lpl::platform::SurfaceDescriptor &surface)
{
    const lpl::core::u32 pitchInPixels = pitchInBytes(surface) / 4u;

    for (lpl::core::u32 y = 0u; y < surface.height; ++y)
    {
        const lpl::core::u32 *source = gPresentedColour + ((y * kPresentedHeight) / surface.height) * kPresentedWidth;
        lpl::core::u32 *destination = surface.buffer + y * pitchInPixels;

        for (lpl::core::u32 x = 0u; x < surface.width; ++x)
            destination[x] = source[(x * kPresentedWidth) / surface.width];
    }
}

} // namespace

/**
 * @brief Gate P3 render: five frames of the kernel's renderer paint its triangle over the
 *        background, and a pixel written to the surface reads back through the HAL.
 */
LPL_TEST(renderer_paints_its_triangle)
{
    lpl::platform::kernel::KernelPlatform platform;
    lpl::platform::IDisplayBackend &display = platform.display();
    lpl::render::kernel::KernelDisplayRenderer renderer{display};
    lpl::platform::SurfaceDescriptor surface;

    if (!hasSurface(display, surface))
    {
        test.skip("no display surface on this boot");
        return;
    }
    (void) renderer.init(surface.width, surface.height);
    for (lpl::core::u32 frame = 0u; frame < 5u; ++frame)
    {
        renderer.tick();
        renderer.beginFrame();
        renderer.endFrame();
    }

    const lpl::core::u32 centreX = surface.width / 2u;
    const lpl::core::u32 centreY = surface.height / 2u;

    test.check(display.readPixel(centreX, centreY) != kBackground, "the centre of the screen is painted over");

    surface.buffer[centreY * (pitchInBytes(surface) / 4u) + centreX] = 0x00ABCDEFu;
    test.check(display.readPixel(centreX, centreY) == 0x00ABCDEFu, "a pixel written to the surface reads back");
}

/**
 * @brief Gate P4 image present: a scene painted by the image module is copied onto the surface
 *        and presented.
 */
LPL_TEST(painted_image_is_presented)
{
    lpl::platform::kernel::KernelPlatform platform;
    lpl::platform::IDisplayBackend &display = platform.display();
    lpl::platform::SurfaceDescriptor surface;

    if (!hasSurface(display, surface))
    {
        test.skip("no display surface on this boot");
        return;
    }

    lpl::image::Image scene(surface.width, surface.height);
    lpl::image::Painter painter(scene);

    paintScene(painter, surface.width, surface.height);
    lpl::image::blitToFramebuffer(scene, surface.buffer, surface.width, surface.height, pitchInBytes(surface), 0, 0);
    display.present();

    const lpl::core::i32 centreX = static_cast<lpl::core::i32>(surface.width / 2u);
    const lpl::core::i32 centreY = static_cast<lpl::core::i32>(surface.height / 2u);

    test.check(colourOf(display.readPixel(surface.width / 2u, surface.height / 2u)) ==
                   colourOf(scene.at(centreX, centreY)),
               "the surface shows the painted scene");
}

/**
 * @brief Gate P5 render present: the viewports, rendered at 320 by 240, are scaled onto the surface
 *        and presented.
 */
LPL_TEST(rendered_viewports_are_presented)
{
    lpl::platform::kernel::KernelPlatform platform;
    lpl::platform::IDisplayBackend &display = platform.display();
    lpl::platform::SurfaceDescriptor surface;
    lpl::render::RenderTarget target{gPresentedColour, gPresentedDepth, kPresentedWidth, kPresentedHeight};

    if (!hasSurface(display, surface))
    {
        test.skip("no display surface on this boot");
        return;
    }
    lpl::render::renderMultiViewport(target);
    upscaleOntoSurface(surface);
    display.present();
    test.check(colourOf(display.readPixel(0u, 0u)) == colourOf(gPresentedColour[0]),
               "the corner of the surface shows the corner of the render");
}
