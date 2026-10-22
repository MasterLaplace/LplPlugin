#include <lpl/engine/Config.hpp>
#include <lpl/engine/TerrainShadows.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(terrain_shadows);

namespace {

/**
 * @brief The side of a chunk, in cells: the client's.
 */
constexpr lpl::core::u32 kChunkSize = 32u;

/**
 * @brief Steps along the ray, from the shaded cell to the cells raised in its way.
 */
constexpr lpl::core::f32 kRiseDistance = 4.0f;

/**
 * @brief Cells either side of the centre of the raised block.
 *
 * @details A block rather than one cell, so the ray meets it whatever the rounding of its steps.
 */
constexpr lpl::core::i32 kRiseHalf = 2;

/**
 * @brief How much the block is raised: far above any ray that starts on the ground beside it.
 */
constexpr lpl::core::i32 kRise = 64;

/**
 * @brief A shadow mask's value for a cell in full shadow.
 */
constexpr lpl::core::u8 kFullShadow = 255u;

/**
 * @struct Shadowed
 * @brief A streamed world with its surface, the way the ring-0 client keeps them, minus the display.
 */
struct Shadowed {
    lpl::engine::TerrainStreamer streamer{};
    lpl::engine::TerrainSurface surface{};

    Shadowed()
    {
        const lpl::procgen::EndlessPlan plan =
            lpl::procgen::endlessPlanFromRecipe(lpl::procgen::parityWorldRecipe(), kChunkSize);

        lpl::procgen::StreamingParams stream;
        stream.generateRadius = 1u;
        stream.maxGeneratePerTick = 16u;
        stream.maxReleasePerTick = 4u;
        streamer.configure(plan.chunk, plan.rivers, stream, 16u, plan.rule);

        const lpl::engine::Config config = lpl::engine::Config::Builder{}.enableTerrainShadows(true).build();
        lpl::engine::TerrainSurfaceParams look;
        look.seaLevel = plan.rule.seaLevel;
        surface.configure(config, look, plan.chunk.worldSeed);

        for (lpl::core::u32 tick = 0u; tick < 16u; ++tick)
            streamer.update(0.0f, 0.0f, 0.0f, 0.0f, [](lpl::engine::TerrainChunk &) {});
    }

    /**
     * @brief The resident chunk at the origin, or nullptr.
     */
    [[nodiscard]] lpl::engine::TerrainChunk *origin()
    {
        for (lpl::core::u32 i = 0u; i < streamer.size(); ++i)
            if (streamer.at(i).coord.x == 0 && streamer.at(i).coord.z == 0)
                return &streamer.at(i);
        return nullptr;
    }

    /**
     * @brief Casts @p chunk's shadows, with no props.
     */
    void cast(lpl::engine::TerrainChunk &chunk) const
    {
        lpl::engine::castChunkShadows(streamer, surface, chunk, [](auto &&) {});
    }
};

/**
 * @brief The offset, in cells, of the point a ray toward the sun reaches after @p distance steps.
 *
 * @details The direction of render::terrainShadow: the sun's horizontal part, normalised.
 */
void stepTowardTheSun(const lpl::render::SunState &sun, lpl::core::f32 distance, lpl::core::i32 &outX,
                      lpl::core::i32 &outZ)
{
    const lpl::core::f32 horizontal = sun.x * sun.x + sun.z * sun.z;
    lpl::core::f32 inverse = 1.0f;
    for (lpl::core::u32 i = 0u; i < 3u; ++i)
        inverse = inverse * (1.5f - 0.5f * horizontal * inverse * inverse);
    outX = static_cast<lpl::core::i32>(sun.x * inverse * distance);
    outZ = static_cast<lpl::core::i32>(sun.z * inverse * distance);
}

/**
 * @brief Whether the block centred @p (centreX, centreZ), in local cells, lies inside the chunk.
 */
[[nodiscard]] bool blockFits(lpl::core::i32 centreX, lpl::core::i32 centreZ)
{
    const lpl::core::i32 last = static_cast<lpl::core::i32>(kChunkSize) - 1;
    return centreX - kRiseHalf >= 0 && centreZ - kRiseHalf >= 0 && centreX + kRiseHalf <= last &&
           centreZ + kRiseHalf <= last;
}

/**
 * @brief The first lit cell whose raised block, toward the sun, lies inside the chunk.
 *
 * @details Found rather than named, so the cell stays lit and the block stays resident whatever the
 *          world's terrain and the sun's place.
 *
 * @return true when one was found.
 */
[[nodiscard]] bool findALitCell(const lpl::engine::TerrainChunk &chunk, lpl::core::i32 stepX, lpl::core::i32 stepZ,
                                lpl::core::u32 &outX, lpl::core::u32 &outZ)
{
    for (lpl::core::u32 z = 0u; z < kChunkSize; ++z)
        for (lpl::core::u32 x = 0u; x < kChunkSize; ++x)
        {
            if (chunk.shade.at(x, z) != 0u ||
                !blockFits(static_cast<lpl::core::i32>(x) + stepX, static_cast<lpl::core::i32>(z) + stepZ))
                continue;
            outX = x;
            outZ = z;
            return true;
        }
    return false;
}

} // namespace

/**
 * @brief A ridge the resident chunk holds casts its shadow, though the world's raw height has none there.
 *
 * @details The rays used to read the world's raw height: the ground before erosion, the rivers and the
 *          measured relief, so the shadows were cast by a ground that is not the one drawn (#566). Here
 *          the resident heights alone are raised, and the cell they stand in front of has to fall dark.
 */
LPL_TEST(a_ridge_of_the_resident_ground_casts_its_shadow)
{
    Shadowed world;
    lpl::engine::TerrainChunk *chunk = world.origin();

    test.check(chunk != nullptr && !chunk->shade.empty() && !chunk->height.empty(),
               "the chunk at the origin is resident, with heights and a shadow mask");
    if (chunk == nullptr || chunk->shade.empty() || chunk->height.empty())
        return;

    const lpl::render::SunState &sun = world.surface.sun();
    test.check(sun.elevation > 0.12f, "the sun is up");

    lpl::core::i32 stepX = 0;
    lpl::core::i32 stepZ = 0;
    stepTowardTheSun(sun, kRiseDistance, stepX, stepZ);

    world.cast(*chunk);
    lpl::core::u32 litX = 0u;
    lpl::core::u32 litZ = 0u;
    const bool found = findALitCell(*chunk, stepX, stepZ, litX, litZ);
    test.check(found, "a lit cell has room for the ridge toward the sun");
    if (!found)
        return;

    const lpl::core::i32 centreX = static_cast<lpl::core::i32>(litX) + stepX;
    const lpl::core::i32 centreZ = static_cast<lpl::core::i32>(litZ) + stepZ;
    for (lpl::core::i32 z = centreZ - kRiseHalf; z <= centreZ + kRiseHalf; ++z)
        for (lpl::core::i32 x = centreX - kRiseHalf; x <= centreX + kRiseHalf; ++x)
        {
            lpl::math::Fixed32 &height =
                chunk->height.at(static_cast<lpl::core::u32>(x), static_cast<lpl::core::u32>(z));
            height = height + lpl::math::Fixed32::fromInt(kRise);
        }

    world.cast(*chunk);
    test.check(chunk->shade.at(litX, litZ) == kFullShadow, "the ridge puts the cell in its shadow");
}

/**
 * @brief A hint from another chunk, or past the resident chunks, still reads the cell's own chunk.
 *
 * @details The rays look the ground up from the chunk they found last. A hint that points elsewhere has
 *          to cost a walk over the resident chunks, never a height read from the wrong one.
 */
LPL_TEST(a_stale_hint_still_reads_the_cells_own_chunk)
{
    Shadowed world;
    const lpl::core::u32 resident = world.streamer.size();

    test.check(resident > 1u, "several chunks are resident");

    lpl::core::u32 wrongHeights = 0u;
    lpl::core::u32 wrongHints = 0u;
    for (lpl::core::u32 i = 0u; i < resident; ++i)
    {
        const lpl::engine::TerrainChunk &chunk = world.streamer.at(i);
        const lpl::core::i32 x = chunk.coord.x * static_cast<lpl::core::i32>(kChunkSize) + 5;
        const lpl::core::i32 z = chunk.coord.z * static_cast<lpl::core::i32>(kChunkSize) + 7;
        const lpl::core::u32 hints[] = {0u, resident - 1u, resident + 5u, i};

        for (const lpl::core::u32 given : hints)
        {
            lpl::core::u32 hint = given;
            if (world.streamer.groundAt(x, z, hint) != chunk.height.at(5u, 7u).toFloat())
                ++wrongHeights;
            if (hint != i)
                ++wrongHints;
        }
    }
    test.check(wrongHeights == 0u, "every hint reads the height of the cell's own chunk");
    test.check(wrongHints == 0u, "and comes back pointing at that chunk");

    const lpl::core::i32 far = 1000 * static_cast<lpl::core::i32>(kChunkSize);
    lpl::core::u32 hint = 1u;
    test.check(world.streamer.groundAt(far, far, hint) ==
                   lpl::procgen::sampleWorldHeight(world.streamer.chunkParams(), far, far).toFloat(),
               "past the resident ring the world's height answers");
    test.check(hint == 1u, "and the hint is left as it was");
}
