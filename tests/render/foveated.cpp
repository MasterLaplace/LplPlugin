#include <lpl/render/Foveated.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(foveated);

namespace {

constexpr lpl::core::u32 kWidth = 128u;
constexpr lpl::core::u32 kHeight = 96u;

/** The shaded image, static because a kernel stack cannot hold it. */
lpl::core::u32 gImage[kWidth * kHeight];

} // namespace

/**
 * @brief Gate P6 foveation: looking at the centre shades fewer fragments than a full pass.
 */
LPL_TEST(the_periphery_is_shaded_coarser)
{
    const auto shaded = lpl::render::foveatedShade(gImage, kWidth, kHeight, 64u, 48u);

    test.check(shaded.shaded_fragments > 0u, "foveation shades fragments");
    test.check(shaded.shaded_fragments < shaded.full_fragments, "fewer than a full pass would");

    test.measure("shaded_fragments", shaded.shaded_fragments);
    test.measure("full_fragments", shaded.full_fragments);
    test.measureHexadecimal("image_signature", shaded.image_signature);
}
