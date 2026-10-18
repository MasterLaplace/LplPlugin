#include <cstdio>
#include <vector>

#include <lpl/ecology/LivingRecipe.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/engine/ResourceManager.hpp>
#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/platform/linux/LinuxPlatform.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/samples/TerrainWorld.hpp>

namespace {

int gChecks = 0;
int gFailures = 0;

void check(const char *what, bool ok)
{
    ++gChecks;
    if (ok)
        return;
    ++gFailures;
    std::printf("  (fail) %s\n", what);
}

/**
 * @brief What a TerrainWorld needs to run without a window: a platform nobody looks at, a cache, an arena and a
 *        bounded configuration.
 */
struct Host {
    lpl::platform::linux_host::LinuxPlatform platform{256u, 256u};
    lpl::engine::ResourceManager resources;
    lpl::engine::Config config = lpl::engine::Config::Builder{}.tickRate(60u).build();
    std::vector<lpl::core::u8> arenaBytes = std::vector<lpl::core::u8>(1u << 20);
    lpl::memory::ArenaAllocator arena{arenaBytes.data(), arenaBytes.size()};
    lpl::engine::WorldContext context{platform, resources, arena, config, nullptr};
};

/**
 * @brief Finds the first world position, scanning the grid row by row, where a plant stands within a cell.
 *
 * @param world The world, initialised.
 * @param half  Half the grid's side: positions run from -half to half - 1 on each axis.
 * @param outX  Receives the position's x when one is found.
 * @param outZ  Receives the position's z when one is found.
 * @return true when a plant was found, and eaten.
 */
bool eatTheFirstPlant(lpl::samples::TerrainWorld &world, lpl::core::i32 half, lpl::core::i32 &outX,
                      lpl::core::i32 &outZ)
{
    for (lpl::core::i32 z = -half; z < half; ++z)
        for (lpl::core::i32 x = -half; x < half; ++x)
            if (world.consumePlantAt(x, z))
            {
                outX = x;
                outZ = z;
                return true;
            }
    return false;
}

/**
 * @brief A grazed plant grows back once its regrowth has run out.
 *
 * @details The doc of TerrainWorld::onFixedStep says the vegetation regrows, and the world held a regrowth counter
 *          on every plant it grazed; nothing counted it down, so a cell grazed once stayed bare for the rest of the
 *          run, and the producer level of the food web never followed what stood.
 *
 *          The cell is grazed bare first, then checked bare, so the last check cannot pass on a plant that was never
 *          eaten. The regrowth is short so the run is a second of world time, not fifteen.
 */
void checkAGrazedPlantGrowsBack()
{
    std::printf("-- a grazed plant grows back\n");
    const lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    lpl::ecology::LivingRecipe living = lpl::ecology::parityLivingRecipe();
    living.regrowthTicks = 30u;

    Host host;
    lpl::samples::TerrainWorld world{recipe, living};
    check("the world initialises", world.onInit(host.context).has_value());
    check("and its scheduler is well formed", world.scheduler().buildGraph().has_value());

    const lpl::core::i32 half = static_cast<lpl::core::i32>(recipe.width / 2u);
    lpl::core::i32 x = 0;
    lpl::core::i32 z = 0;
    check("a plant stands somewhere on the grid", eatTheFirstPlant(world, half, x, z));

    lpl::core::u32 eaten = 1u;
    while (world.consumePlantAt(x, z))
        ++eaten;
    check("the cell and its neighbours are grazed bare", !world.consumePlantAt(x, z));

    for (lpl::core::u32 tick = 0u; tick <= living.regrowthTicks; ++tick)
    {
        world.onFixedStep(1.0f / 60.0f);
        world.registry().swapAllBuffers();
    }

    std::printf("     %u plants eaten around (%d, %d), then %u ticks\n", eaten, x, z, living.regrowthTicks + 1u);
    check("a plant stands there again once its regrowth ran out", world.consumePlantAt(x, z));
}

} // namespace

/**
 * @brief The sample world, run without a window.
 *
 * @details `test-terrain-render` only constructs a TerrainWorld, which proves its members compile; what one does
 *          when it ticks had no host test, and the kernel's client profile was the only thing that ran it.
 */
int main()
{
    std::printf("== the terrain world ==\n");

    checkAGrazedPlantGrowsBack();

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
