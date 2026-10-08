#include <lpl/ecs/Registry.hpp>
#include <lpl/render/SoftwareRasterizer.hpp>
#include <lpl/samples/CubePile.hpp>
#include <lpl/testing/Test.hpp>

#include <new>

LPL_TEST_SUITE(cube_pile);

namespace {

constexpr lpl::core::u32 kWidth = 192u;
constexpr lpl::core::u32 kHeight = 120u;

/** The target the pile is drawn into, static because a kernel stack cannot hold it. */
lpl::core::u32 gColour[kWidth * kHeight];
lpl::core::f32 gDepth[kWidth * kHeight];

/**
 * @brief Storage of the registry and the pile a run builds, in BSS: a pile carries a 4 KiB tint
 *        table and its registry drives 1024 heap chunks, too much for a kernel stack.
 */
alignas(lpl::ecs::Registry) unsigned char gRegistryStorage[sizeof(lpl::ecs::Registry)];
alignas(lpl::samples::CubePile) unsigned char gPileStorage[sizeof(lpl::samples::CubePile)];

/**
 * @struct CubePileFold
 * @brief The cube pile after some ticks: its authoritative state and the image of it, folded.
 */
struct CubePileFold {
    lpl::core::u32 stateSignature{0u}; /**< Fold of the pile's Fixed32 state. */
    lpl::core::u32 imageSignature{0u}; /**< Fold of the image rendered of it. */
};

/**
 * @brief Seeds a fresh pile, advances it @p ticks deterministic steps, renders it into @p target,
 *        and folds both.
 *
 * @details A fresh registry and pile on each call, so every run starts from the same state. The
 *          registry is a throwaway one; on the live engine path the pile runs on the hosting
 *          World's registry instead.
 */
[[nodiscard]] CubePileFold runCubePileAndFold(const lpl::render::RenderTarget &target, lpl::core::u32 ticks)
{
    lpl::ecs::Registry *registry = ::new (static_cast<void *>(gRegistryStorage)) lpl::ecs::Registry();
    lpl::samples::CubePile *pile = ::new (static_cast<void *>(gPileStorage)) lpl::samples::CubePile(*registry);

    pile->init();
    for (lpl::core::u32 tick = 0u; tick < ticks; ++tick)
        pile->step();
    pile->render(target, lpl::samples::CubePile::Camera{});

    const CubePileFold fold{pile->stateSignature(), lpl::render::foldTarget(target)};

    pile->~CubePile();
    registry->~Registry();
    return fold;
}

} // namespace

/**
 * @brief The cube pile the client runs: its Fixed32 state and the image of it, after 8 and after
 *        64 ticks, are the same on both targets.
 *
 * @details Not run twice: 64 more ticks are most of the boot's test time in ring 0, and the two
 *          targets agreeing is the stronger statement of determinism.
 */
LPL_TEST(pile_settles_the_same_way)
{
    lpl::render::RenderTarget target{gColour, gDepth, kWidth, kHeight};
    const CubePileFold early = runCubePileAndFold(target, 8u);
    const CubePileFold late = runCubePileAndFold(target, 64u);

    test.check(early.stateSignature != late.stateSignature, "the state moves between tick 8 and tick 64");
    test.check(early.imageSignature != late.imageSignature, "and so does the image of it");

    test.measureHexadecimal("state_signature_at_8", early.stateSignature);
    test.measureHexadecimal("image_signature_at_8", early.imageSignature);
    test.measureHexadecimal("state_signature_at_64", late.stateSignature);
    test.measureHexadecimal("image_signature_at_64", late.imageSignature);
}
