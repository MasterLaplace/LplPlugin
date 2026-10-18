#include <cstdio>
#include <cstdlib>
#include <vector>

#include <lpl/engine/Config.hpp>
#include <lpl/engine/PropLibrary.hpp>
#include <lpl/engine/TerrainRenderer.hpp>
#include <lpl/engine/TerrainStreamer.hpp>
#include <lpl/engine/TerrainSurface.hpp>
#include <lpl/procgen/CaveWarren.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/samples/TerrainWorld.hpp>

static int failures = 0;
static int checks = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    ++checks;
    if (!ok)
        ++failures;
}

namespace {

constexpr lpl::core::u32 kWidth = 192u;
constexpr lpl::core::u32 kHeight = 120u;
constexpr lpl::core::u32 kChunkSize = 24u;

/**
 * @brief How far the fixture raises the sea so the origin window HAS water in it.
 *
 * @warning Not a tuning knob, a fixture requirement, and it came back from the dead once. The
 * walked world's landforms are about a hundred and ten cells across; this harness streams a
 * five-by-five block of twenty-four-cell chunks, which is a hundred and twenty. The whole
 * window therefore fits on ONE landform — and when that landform is above the waterline
 * there is no sea, no river and nothing for half of this file to assert about.
 *
 * Six metres puts the median cell at the waterline, so the window is partly drowned rather
 * than dry or submerged: the case the tightening exists for. Every block below that says
 * anything about water uses it, and the ones that measure it now REFUSE to pass on an empty
 * set — "every drowned cell is inside the bounds" is trivially true of no cells at all, and
 * that is exactly how this went unnoticed.
 */
constexpr lpl::core::f32 kSeaLift = 6.0f;

/** @brief A frame's worth of memory, and the only render target this test has. */
struct Frame {
    std::vector<lpl::core::u32> colour;
    std::vector<lpl::core::f32> depth;

    Frame()
        : colour(static_cast<std::size_t>(kWidth) * kHeight, 0u),
          depth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f)
    {
    }

    [[nodiscard]] lpl::render::RenderTarget target() noexcept
    {
        return lpl::render::RenderTarget{&colour[0], &depth[0], kWidth, kHeight};
    }
};

/**
 * @brief Fold of a frame, so "the picture changed" is a fact rather than an impression.
 */
[[nodiscard]] lpl::core::u32 foldFrame(const std::vector<lpl::core::u32> &pixels) noexcept
{
    lpl::core::u32 hash = 0x811C9DC5u;
    for (lpl::core::u32 pixel : pixels)
        for (lpl::core::u32 byte = 0u; byte < 4u; ++byte)
        {
            hash ^= (pixel >> (byte * 8u)) & 0xFFu;
            hash *= 0x01000193u;
        }
    return hash;
}

/**
 * @brief The biome palette the sample uses, reduced to what a fold needs.
 */
[[nodiscard]] lpl::core::u32 palette(lpl::procgen::BiomeId biome) noexcept
{
    return 0x00203040u + static_cast<lpl::core::u32>(biome) * 0x00101010u;
}

/**
 * @brief The walked plan, derived ONCE.
 *
 * procgen::endlessPlanFromRecipe calibrates the river threshold, and a calibration is a
 * bisection over a nine-chunk window — the most expensive thing in this file by an order
 * of magnitude. Six harnesses paying for it six times turned a two-second test into a
 * ninety-second one, which is how a test stops being run.
 */
[[nodiscard]] const lpl::procgen::EndlessPlan &walkedPlan()
{
    static const lpl::procgen::EndlessPlan plan =
        lpl::procgen::endlessPlanFromRecipe(lpl::procgen::parityWorldRecipe(), kChunkSize);
    return plan;
}

/** @brief A world configured the way the ring-0 client is, minus the display. */
struct Harness {
    lpl::procgen::EndlessPlan plan{};
    lpl::engine::TerrainStreamer streamer{};
    lpl::engine::TerrainSurface surface{};
    lpl::engine::PropLibrary props{};
    lpl::ecs::Registry registry{};
    lpl::engine::TerrainRenderer renderer{};
    lpl::render::OrbitCamera camera{};

    /**
     * @brief Streams the window around @p focusChunk rather than always around the origin.
     *
     * @warning Not a convenience. The walked world's landforms are about a hundred and ten cells
     * across and this window is a hundred and twenty, so it fits on ONE of them — and the
     * one at the origin has neither sea nor river in it. Every block below that measures
     * water was passing or failing on an empty set. The repository already knows this shape:
     * `testDeterminism` in test_procgen_chunking hunts for a chunk that carries water first,
     * because "comparing two empty masks proves only that nothing equals nothing".
     *
     * Raising the sea is how this test gets coverage of the sea path at all: the walked parity world
     * sits almost entirely above its own sea level, so at the recipe's level no chunk of the window
     * around the origin is wet and the tightening is never exercised on a PARTIAL chunk — which is the
     * case it exists for.
     */
    explicit Harness(lpl::core::u32 tessellation, bool reflection, lpl::core::f32 seaLift = 0.0f,
                     lpl::procgen::ChunkCoord focusChunk = {0, 0})
    {
        plan = walkedPlan();
        plan.rule.seaLevel += seaLift;
        plan.rivers.seaLevel += seaLift;

        lpl::procgen::StreamingParams stream;
        stream.generateRadius = 2u;
        stream.maxGeneratePerTick = 64u;
        stream.maxReleasePerTick = 4u;
        streamer.configure(plan.chunk, plan.rivers, stream, 64u, plan.rule);

        const lpl::engine::Config config = lpl::engine::Config::Builder{}
                                               .enablePerPixelSurface(true)
                                               .enableTerrainShadows(false)
                                               .enableWaterReflection(reflection)
                                               .waterTessellation(tessellation)
                                               .build();
        lpl::engine::TerrainSurfaceParams look;
        look.seaLevel = plan.rule.seaLevel;
        surface.configure(config, look, plan.chunk.worldSeed);

        lpl::engine::PropLibraryParams propParams;
        props.build(propParams, plan.chunk.worldSeed);

        focus = focusChunk;
        camera.setFocus(0.0f, 0.0f);
    }

    lpl::procgen::ChunkCoord focus{0, 0};

    /**
     * @brief Streams the chunks around @ref focus in, then reports how many hold sea.
     */
    lpl::core::u32 fill()
    {
        const lpl::core::f32 fx = static_cast<lpl::core::f32>(focus.x * static_cast<lpl::core::i32>(kChunkSize));
        const lpl::core::f32 fz = static_cast<lpl::core::f32>(focus.z * static_cast<lpl::core::i32>(kChunkSize));
        camera.setFocus(fx, fz);
        for (lpl::core::u32 tick = 0u; tick < 64u; ++tick)
            streamer.update(fx, fz, 0.0f, 0.0f, [](lpl::engine::TerrainChunk &) {});
        lpl::core::u32 wet = 0u;
        for (lpl::core::u32 i = 0u; i < streamer.size(); ++i)
            if (streamer.at(i).hasSea)
                ++wet;
        return wet;
    }

    [[nodiscard]] lpl::engine::TerrainDrawParams drawParams() const
    {
        lpl::engine::TerrainDrawParams params;
        params.chunkSize = kChunkSize;
        params.lodRings = 3u;
        params.seaLevel = plan.rule.seaLevel;
        params.riverSurfaceRise = plan.riverSurfaceRise;
        params.caveMouthDrop = plan.rule.caveMouthDrop;
        params.caveDrawRadius = caveDrawRadius;
        params.useFocusHeight = useFocusHeight;
        params.focusHeight = focusHeight;
        return params;
    }

    lpl::core::u32 caveDrawRadius{0u};
    bool useFocusHeight{false};
    lpl::core::f32 focusHeight{0.0f};

    /**
     * @brief Streams around a WORLD CELL rather than a chunk, and reports the cave there.
     *
     * A cave is sited on a landmark lattice that has nothing to do with the chunk grid,
     * so a fixture that streamed around a chunk coordinate would be hunting for one.
     */
    const lpl::procgen::CaveWarren *fillAtCell(lpl::core::i32 cellX, lpl::core::i32 cellZ)
    {
        const lpl::core::f32 fx = static_cast<lpl::core::f32>(cellX);
        const lpl::core::f32 fz = static_cast<lpl::core::f32>(cellZ);
        camera.setFocus(fx, fz);
        for (lpl::core::u32 tick = 0u; tick < 64u; ++tick)
            streamer.update(fx, fz, 0.0f, 0.0f, [](lpl::engine::TerrainChunk &) {});
        return streamer.warrenAt(cellX, cellZ);
    }

    lpl::core::u32 draw(Frame &frame, lpl::core::u32 tick)
    {
        return renderer.drawStreamed(frame.target(), camera, streamer, surface, props, registry, drawParams(), tick,
                                     palette,
                                     [this](lpl::core::i32 x, lpl::core::i32 z) { return streamer.groundAt(x, z); });
    }
};

/**
 * @brief The nearest chunk carrying what a block is about to measure.
 *
 * Spiral out from the origin until `carries` says yes. Without it every water assertion in
 * this file was taken on a window that happened to contain no water — and "every drowned
 * cell is inside its bounds" is trivially true of no cells at all.
 *
 * Each radius visits its own ring only: the interior was tested at a smaller radius.
 */
template <typename Carries> [[nodiscard]] lpl::procgen::ChunkCoord findChunk(lpl::core::f32 seaLift, Carries &&carries)
{
    lpl::procgen::EndlessPlan plan = walkedPlan();
    plan.rule.seaLevel += seaLift;
    plan.rivers.seaLevel += seaLift;
    for (lpl::core::i32 radius = 0; radius <= 14; ++radius)
        for (lpl::core::i32 cz = -radius; cz <= radius; ++cz)
            for (lpl::core::i32 cx = -radius; cx <= radius; ++cx)
            {
                if (cx > -radius && cx < radius && cz > -radius && cz < radius)
                    continue;
                const lpl::procgen::ChunkTerrain terrain = lpl::procgen::generateChunkTerrain(
                    plan.chunk, plan.rivers, {cx, cz}, plan.rule, [](lpl::core::i32, lpl::core::i32) {});
                if (carries(terrain))
                    return {cx, cz};
            }
    return {0, 0};
}

/**
 * @brief The sample world type-checks on the host.
 *
 * @details Constructed, not run: running it needs an engine::WorldContext, and none of the errors this
 *          catches need one. What is being exercised is the COMPILER — the include of
 *          `samples/TerrainWorld.hpp` is the whole test — and this is the smallest thing that keeps the
 *          include from being dropped as unused. The assertion is thin on purpose: what is being checked
 *          is that the TU COMPILED, and the compiler has already answered by the time it runs.
 */
void checkSampleWorldTypeChecksOnTheHost()
{
    std::printf("== the sample world ==\n");
    lpl::samples::TerrainWorld world{lpl::procgen::parityWorldRecipe()};
    std::printf("    constructed \"%s\"\n", world.name());
    check(world.name() != nullptr && world.name()[0] != '\0', "the sample world names itself");
}

/**
 * @brief The path compiles and draws something.
 *
 * @param seaFocus The nearest chunk whose window holds sea.
 */
void checkPathCompilesAndDrawsSomething(lpl::procgen::ChunkCoord seaFocus)
{
    lpl::core::u32 flatTriangles = 0u;
    lpl::core::u32 wetChunks = 0u;
    lpl::core::u32 riverChunks = 0u;
    Harness harness{0u, true, kSeaLift, seaFocus};
    wetChunks = harness.fill();
    for (lpl::core::u32 i = 0u; i < harness.streamer.size(); ++i)
        if (harness.streamer.at(i).hasRiver)
            ++riverChunks;

    Frame frame;
    flatTriangles = harness.draw(frame, 0u);
    std::printf("    %u chunks resident, %u hold sea, %u hold river; %u triangles\n", harness.streamer.size(),
                wetChunks, riverChunks, flatTriangles);
    check(harness.streamer.size() > 0u, "chunks streamed in");
    check(flatTriangles > 0u, "the frame drew geometry");
}

/**
 * @brief The sea's extent is the sea's extent.
 *
 * @details `lowest` says whether a chunk holds water; the bounds say WHERE. A chunk that reports sea
 *          must have a drowned cell inside its bounds, and every drowned cell must be inside them — the
 *          second half is the one that would fail silently, by clipping the ocean to a corner of itself.
 *
 *          That the window holds drowned cells at all is checked FIRST, or the two checks after it are
 *          true of nothing: a window with no water passes "every drowned cell is inside the bounds"
 *          without touching a single bound.
 *
 * @param seaFocus The nearest chunk whose window holds sea.
 */
void checkSeaExtentIsTheSeasExtent(lpl::procgen::ChunkCoord seaFocus)
{
    std::printf("== the sea's extent ==\n");
    Harness harness{0u, true, kSeaLift, seaFocus};
    harness.fill();
    lpl::core::u32 checkedChunks = 0u;
    lpl::core::u32 outside = 0u;
    lpl::core::u32 insideEmpty = 0u;
    for (lpl::core::u32 i = 0u; i < harness.streamer.size(); ++i)
    {
        const lpl::engine::TerrainChunk &chunk = harness.streamer.at(i);
        if (chunk.height.empty())
            continue;
        lpl::core::u32 drowned = 0u;
        lpl::core::u32 inBounds = 0u;
        for (lpl::core::u32 z = 0u; z < kChunkSize; ++z)
            for (lpl::core::u32 x = 0u; x < kChunkSize; ++x)
            {
                if (chunk.height.at(x, z).toFloat() >= harness.plan.rule.seaLevel)
                    continue;
                ++drowned;
                if (x >= chunk.seaMinX && x <= chunk.seaMaxX && z >= chunk.seaMinZ && z <= chunk.seaMaxZ)
                    ++inBounds;
            }
        if (drowned == 0u)
        {
            if (chunk.hasSea)
                ++insideEmpty;
            continue;
        }
        ++checkedChunks;
        if (!chunk.hasSea || inBounds != drowned)
            ++outside;
    }
    std::printf("    %u chunks with drowned cells, %u mis-bounded, %u claiming sea with none\n", checkedChunks, outside,
                insideEmpty);
    check(checkedChunks > 0u, "the fixture window actually contains drowned cells");
    check(outside == 0u, "every drowned cell is inside the reported bounds");
    check(insideEmpty == 0u, "no chunk reports sea it does not have");
}

/**
 * @brief The reflection pass is skipped when nothing reflects.
 *
 * @details The probe is a whole second render of the world, amortised over four frames. It used to run
 *          over dry land forever because nothing ever asked whether there was water on screen. The
 *          measurement is the probe's OWN validity: a pass that ran leaves a valid probe behind, and one
 *          that was skipped does not.
 *
 *          Whichever way the world came out, the two must AGREE. That is the invariant; "the probe never
 *          runs" would be satisfied by a probe that is simply broken.
 *
 * @param seaFocus The nearest chunk whose window holds sea.
 */
void checkReflectionPassIsSkippedWhenNothingReflects(lpl::procgen::ChunkCoord seaFocus)
{
    std::printf("== the reflection probe ==\n");
    Harness harness{0u, true, kSeaLift, seaFocus};
    const lpl::core::u32 wet = harness.fill();
    std::vector<lpl::core::u32> probeColour(static_cast<std::size_t>(64u) * 48u, 0u);
    std::vector<lpl::core::f32> probeDepth(static_cast<std::size_t>(64u) * 48u, 0.0f);
    harness.surface.attachProbe(&probeColour[0], &probeDepth[0], 64u, 48u);

    Frame frame;
    harness.draw(frame, 0u);
    std::printf("    %u wet chunks resident; probe %s\n", wet, harness.surface.probeValid() ? "ran" : "skipped");
    check(harness.surface.probeValid() == (wet > 0u), "the probe runs exactly when water is in sight");
}

/**
 * @brief A river is a surface, not a colour.
 *
 * @details River cells used to be Lake-coloured cells of GROUND: the blue on screen was terrain that
 *          happened to be blue, with no reflection and no depth. Now each one is a quad standing in its
 *          carved bed, so asking for it must produce triangles that asking for no river does not.
 *
 *          NO lift here, and that is the point: a river is water standing ABOVE the sea in a carved
 *          channel, so raising the sea six metres to guarantee an ocean — which is what the checks of
 *          the sea need — puts every river cell under it and the renderer rightly draws none. A fixture
 *          that drowns the thing it measures reports "no river water" and blames the code.
 *
 * @param riverFocus The nearest chunk whose window holds a river.
 */
void checkRiverIsASurfaceNotAColour(lpl::procgen::ChunkCoord riverFocus)
{
    std::printf("== river water ==\n");
    Harness dry{0u, false, 0.0f, riverFocus};
    dry.fill();
    Frame dryFrame;
    lpl::engine::TerrainDrawParams noRiver = dry.drawParams();
    noRiver.riverSurfaceRise = 0.0f;
    const lpl::core::u32 without = dry.renderer.drawStreamed(
        dryFrame.target(), dry.camera, dry.streamer, dry.surface, dry.props, dry.registry, noRiver, 0u, palette,
        [&dry](lpl::core::i32 x, lpl::core::i32 z) { return dry.streamer.groundAt(x, z); });

    Harness wet{0u, false, 0.0f, riverFocus};
    wet.fill();
    Frame wetFrame;
    const lpl::core::u32 with = wet.draw(wetFrame, 0u);
    std::printf("    %u triangles without river water, %u with\n", without, with);
    check(wet.streamer.at(0u).hasRiver || with > without, "the fixture window actually contains river");
    check(with > without, "asking for river water draws more than not asking");
    check(foldFrame(wetFrame.colour) != foldFrame(dryFrame.colour), "and it changes the picture");
}

/**
 * @brief The swell is geometry when the host pays for it.
 *
 * @details Two conditions gate the displaced mesh and BOTH are necessary: a world with no swell has
 *          nothing to displace, and a host that declined the cost gets the flat quad. Each is asserted
 *          on its own, because either alone passing would look like the feature working.
 *
 * @param seaFocus The nearest chunk whose window holds sea.
 */
void checkSwellIsGeometryWhenTheHostPaysForIt(lpl::procgen::ChunkCoord seaFocus)
{
    std::printf("== the displaced surface ==\n");
    Harness flat{0u, false, kSeaLift, seaFocus};
    flat.fill();
    Frame flatFrame;
    flat.surface.water().swellHeight = 0.9f;
    const lpl::core::u32 flatCount = flat.draw(flatFrame, 0u);

    Harness tess{6u, false, kSeaLift, seaFocus};
    tess.fill();
    Frame tessFrame;
    tess.surface.water().swellHeight = 0.9f;
    const lpl::core::u32 tessCount = tess.draw(tessFrame, 0u);

    Harness stillFlat{6u, false, kSeaLift, seaFocus};
    stillFlat.fill();
    Frame stillFrame;
    stillFlat.surface.water().swellHeight = 0.0f;
    const lpl::core::u32 stillCount = stillFlat.draw(stillFrame, 0u);

    std::printf("    tessellation 0: %u triangles, 6: %u, 6 with no swell: %u\n", flatCount, tessCount, stillCount);
    check(tessCount > flatCount, "a paid-for tessellation spends more triangles");
    check(stillCount == flatCount, "and a world with no swell spends none of them");
}

/**
 * @brief The frame is the same frame twice.
 *
 * @details Not a cross-target gate — this path is float and non-authoritative by contract — but a
 *          rendered frame must at least be a function of the world and the tick. A renderer that reads
 *          uninitialised memory or iterates a hash map passes everything above and fails this.
 *
 * @param seaFocus The nearest chunk whose window holds sea.
 */
void checkFrameIsTheSameFrameTwice(lpl::procgen::ChunkCoord seaFocus)
{
    std::printf("== determinism ==\n");
    Harness first{6u, true, kSeaLift, seaFocus};
    first.fill();
    Frame frameA;
    first.draw(frameA, 3u);

    Harness second{6u, true, kSeaLift, seaFocus};
    second.fill();
    Frame frameB;
    second.draw(frameB, 3u);

    const lpl::core::u32 a = foldFrame(frameA.colour);
    const lpl::core::u32 b = foldFrame(frameB.colour);
    std::printf("    fold 0x%08X against 0x%08X\n", a, b);
    check(a == b, "the same world at the same tick renders the same pixels");
}

/**
 * @brief The probe obeys the host, not only the world.
 *
 * @details Two independent reasons to skip the pass, and a test that only covered one would let the
 *          other rot: the world may have no water, and the host may have declined reflections outright.
 *
 * @param seaFocus The nearest chunk whose window holds sea.
 */
void checkProbeObeysTheHost(lpl::procgen::ChunkCoord seaFocus)
{
    std::printf("== the probe obeys the host ==\n");
    Harness harness{0u, false, kSeaLift, seaFocus};
    harness.fill();
    std::vector<lpl::core::u32> probeColour(static_cast<std::size_t>(64u) * 48u, 0u);
    std::vector<lpl::core::f32> probeDepth(static_cast<std::size_t>(64u) * 48u, 0.0f);
    harness.surface.attachProbe(&probeColour[0], &probeDepth[0], 64u, 48u);
    Frame frame;
    harness.draw(frame, 0u);
    check(!harness.surface.probeValid(), "a host that declined reflections gets no probe pass");
}

/**
 * @brief A partly drowned chunk is bounded tightly.
 *
 * @details The case the tightening exists for, and the one the recipe's own sea level cannot produce.
 *          With the sea raised, most chunks it reaches are PARTLY under it — so at least one must report
 *          bounds strictly inside its own extent. Without this the tightening could be returning the whole
 *          chunk every time and every other check would pass.
 *
 *          This used to lift the sea twelve metres to manufacture partly drowned chunks, because the walked
 *          world had none: it was entirely above its own sea level. It has a coastline now, and a lift that
 *          high drowns most of the chunks it reaches edge to edge; the sea is raised by @ref kSeaLift
 *          instead.
 */
void checkPartlyDrownedChunkIsBoundedTightly()
{
    std::printf("== a coast, not an ocean ==\n");
    Harness harness{0u, true, kSeaLift};
    const lpl::core::u32 wet = harness.fill();
    lpl::core::u32 partial = 0u;
    lpl::core::u32 full = 0u;
    for (lpl::core::u32 i = 0u; i < harness.streamer.size(); ++i)
    {
        const lpl::engine::TerrainChunk &chunk = harness.streamer.at(i);
        if (!chunk.hasSea)
            continue;
        const bool tight = chunk.seaMinX > 0u || chunk.seaMinZ > 0u || chunk.seaMaxX < kChunkSize - 1u ||
                           chunk.seaMaxZ < kChunkSize - 1u;
        if (tight)
            ++partial;
        else
            ++full;
    }
    std::printf("    %u wet chunks: %u bounded tightly, %u drowned edge to edge\n", wet, partial, full);
    check(wet > 1u, "the world's own sea reaches more than one chunk");
    check(partial > 0u, "and at least one is bounded tighter than its own extent");
}

/**
 * @brief And from OUTSIDE, which is the half a player meets first.
 *
 * @details Standing in the trench looking at the doorway. This is the view that was reported as broken
 *          and it had no coverage at all: every check before it puts the eye inside, where the geometry
 *          is unavoidably in front of it. Facing the doorway: the camera walks -sin(yaw), -cos(yaw),
 *          same convention as the body, so looking along the adit is atan2 of the negated step.
 *
 * @param harness The harness, streamed around the cave.
 * @param located The cave.
 */
void checkDoorwayShowsACaveFromOutside(Harness &harness, const lpl::procgen::CaveWarren &located)
{
    const lpl::core::i32 outX = located.apertureX[0] - located.adit.stepX * 5;
    const lpl::core::i32 outZ = located.apertureZ[0] - located.adit.stepZ * 5;
    harness.fillAtCell(outX, outZ);
    harness.camera.setFocus(static_cast<lpl::core::f32>(outX), static_cast<lpl::core::f32>(outZ));
    harness.camera.setYaw(lpl::math::Cordic::atan2(lpl::math::Fixed32::fromInt(-located.adit.stepX),
                                                   lpl::math::Fixed32::fromInt(-located.adit.stepZ))
                              .toFloat());
    harness.focusHeight = located.adit.floorY;
    harness.caveDrawRadius = 12u;
    Frame porch;
    const lpl::core::u32 porchTriangles = harness.draw(porch, 0u);
    harness.caveDrawRadius = 0u;
    Frame plain;
    const lpl::core::u32 plainTriangles = harness.draw(plain, 0u);
    lpl::core::u32 mouthPixels = 0u;
    for (lpl::core::u32 i = 0u; i < kWidth * kHeight; ++i)
        mouthPixels += porch.colour[i] != plain.colour[i] ? 1u : 0u;
    std::printf("    from the trench: %u vs %u triangles, %u pixels of cave visible\n", porchTriangles, plainTriangles,
                mouthPixels);
    std::printf("    (a doorway shows a cave through it: %s)\n", mouthPixels > 200u ? "yes" : "NO");
    check(mouthPixels > 200u, "the doorway is a hole you can see a cave through");
    harness.caveDrawRadius = 12u;
}

/**
 * @brief The lamp is a SPOT, not a glow.
 *
 * @details A bulb at the eye lights everything around you equally, which is a cave with no direction to
 *          walk in. The claim is that the beam has an AXIS, and the only way to assert that is to turn
 *          the eye and require the picture to change: a brightness measurement alone is satisfied by a
 *          glow. The eye turns a quarter: same eye, same world.
 *
 * @param harness The harness, with its eye in the cave.
 * @param insideX Column of the cell a few steps inside the doorway.
 * @param insideZ Row of that cell.
 * @param span    The rock above and below that cell.
 */
void checkLampIsASpotNotAGlow(Harness &harness, lpl::core::i32 insideX, lpl::core::i32 insideZ,
                              const lpl::procgen::VerticalSpan &span)
{
    harness.camera.setFocus(static_cast<lpl::core::f32>(insideX), static_cast<lpl::core::f32>(insideZ));
    harness.focusHeight = span.floor.toFloat();
    harness.camera.setYaw(0.0f);
    Frame ahead;
    (void) harness.draw(ahead, 0u);
    harness.camera.setYaw(1.57079633f);
    Frame aside;
    (void) harness.draw(aside, 0u);
    harness.camera.setYaw(0.0f);
    check(foldFrame(ahead.colour) != foldFrame(aside.colour), "turning the head moves the beam");

    if (const char *dump = std::getenv("LPL_DUMP_CAVE_PPM"); dump != nullptr)
    {
        std::FILE *out = std::fopen(dump, "wb");
        if (out != nullptr)
        {
            std::fprintf(out, "P6\n%u %u\n255\n", kWidth, kHeight);
            for (lpl::core::u32 i = 0u; i < kWidth * kHeight; ++i)
            {
                const lpl::core::u32 pixel = ahead.colour[i];
                const unsigned char rgb[3] = {static_cast<unsigned char>((pixel >> 16) & 0xFFu),
                                              static_cast<unsigned char>((pixel >> 8) & 0xFFu),
                                              static_cast<unsigned char>(pixel & 0xFFu)};
                std::fwrite(rgb, 1u, 3u, out);
            }
            std::fclose(out);
        }
    }
}

/**
 * @brief Underground.
 *
 * @details The render path had no coverage below the surface at all, and it is the half of the cave
 *          feature a signature cannot reach: the parity gate proves a body gets inside and stays where
 *          it should, and says nothing about whether anything is drawn once it is there. "The cave is
 *          unlit" and "I never got in" look identical from a dark screen, so each is measured
 *          separately here.
 *
 *          The doorway is that of a real cave in this world, found rather than named: which landmark
 *          cells carry one is a property of the terrain, and a constant here would quietly stop pointing
 *          at a cave the day the terrain moved. The eye stands a few steps INSIDE the doorway, so it is
 *          under the rock rather than in the trench looking at it.
 *
 *          FIRST PERSON, and this is the check that caught itself being useless. The harness camera
 *          defaults to an orbit, so it stands well back and above its focus — outside the hill, looking
 *          at it. Every cave face was drawn and every one lost the depth test to the terrain in front of
 *          it: 2194 triangles submitted and ZERO pixels different from the control. The "no sky"
 *          assertion passed anyway, off the colour beginCaveFrame clears to, which is a check satisfied
 *          by the thing it was meant to rule out.
 *
 *          And the control: the SAME eye, the same world, with the host budget at zero. Without it
 *          "triangles were drawn" is satisfied by the terrain the pass would have drawn anyway, and the
 *          cave geometry could be absent.
 *
 *          No sky. The frame begins with beginCaveFrame, so what a pixel the geometry misses shows is
 *          the cave's own dark — not the blue the surface path clears to. Counted rather than sampled at
 *          one pixel: a single probe would land on whatever happened to be there.
 *
 *          The control for the control. "No pixel is sky" could hold because the palette happens to be
 *          warm, so the SAME world is rendered from an eye on the surface — which must show sky, or the
 *          test is measuring a colour scheme rather than a roof. The eye decides now, so this is a move
 *          rather than a flag: put it above ground and the sky comes back on its own.
 */
void checkUnderground()
{
    std::printf("== inside a cave ==\n");
    Harness harness{0u, false};
    const lpl::procgen::EndlessPlan &plan = harness.plan;
    lpl::procgen::CaveWarren located;
    for (lpl::core::i32 lz = -6; lz <= 6 && !located.valid; ++lz)
        for (lpl::core::i32 lx = -6; lx <= 6; ++lx)
        {
            lpl::procgen::LandmarkSite site;
            if (!lpl::procgen::landmarkAt(plan.chunk, plan.rule.caveMouths, lpl::procgen::LandmarkKind::CaveMouth,
                                          plan.rule.seaLevel, lx, lz, site))
                continue;
            lpl::procgen::CaveWarren candidate =
                lpl::procgen::buildCaveWarren(plan.chunk, site, plan.rule.warren, plan.rule.caveMouthDrop);
            if (candidate.valid)
            {
                located = static_cast<lpl::procgen::CaveWarren &&>(candidate);
                break;
            }
        }
    check(located.valid, "the walked world has a cave to stand in");

    const lpl::core::i32 insideX = located.apertureX[0] + located.adit.stepX * 3;
    const lpl::core::i32 insideZ = located.apertureZ[0] + located.adit.stepZ * 3;
    const lpl::procgen::CaveWarren *resident = harness.fillAtCell(insideX, insideZ);
    check(resident != nullptr, "and the chunk that owns it is resident where a body would be");

    if (resident != nullptr)
    {
        const lpl::procgen::VerticalSpan span =
            harness.streamer.spanAt(insideX, insideZ, lpl::math::Fixed32::fromFloat(located.adit.floorY + 0.5f));
        check(span.enclosed, "the collider agrees there is rock overhead there");

        harness.camera.setFirstPerson(true);
        harness.camera.setPitch(0.0f);
        harness.camera.setYaw(0.0f);
        harness.camera.setEyeHeight(1.6f);
        harness.camera.setFocus(static_cast<lpl::core::f32>(insideX), static_cast<lpl::core::f32>(insideZ));
        harness.caveDrawRadius = 12u;
        harness.useFocusHeight = true;
        harness.focusHeight = span.floor.toFloat();

        Frame lit;
        const lpl::core::u32 caveTriangles = harness.draw(lit, 0u);

        harness.caveDrawRadius = 0u;
        Frame dark;
        const lpl::core::u32 bareTriangles = harness.draw(dark, 0u);

        lpl::core::u32 differing = 0u;
        for (lpl::core::u32 i = 0u; i < kWidth * kHeight; ++i)
            differing += lit.colour[i] != dark.colour[i] ? 1u : 0u;
        std::printf("    %u triangles with the cave drawn, %u without; %u pixels differ; folds %08X %08X\n",
                    caveTriangles, bareTriangles, differing, foldFrame(lit.colour), foldFrame(dark.colour));
        check(caveTriangles > bareTriangles, "asking for cave geometry draws more than not asking");
        check(foldFrame(lit.colour) != foldFrame(dark.colour), "and it changes the picture");

        lpl::core::u32 skyish = 0u;
        lpl::core::u32 rocky = 0u;
        for (lpl::core::u32 i = 0u; i < kWidth * kHeight; ++i)
        {
            const lpl::core::u32 pixel = lit.colour[i];
            const lpl::core::u32 red = (pixel >> 16) & 0xFFu;
            const lpl::core::u32 blue = pixel & 0xFFu;
            if (blue > red + 24u)
                ++skyish;
            if (red > 24u || blue > 24u)
                ++rocky;
        }
        const lpl::core::f32 keptHeight = harness.focusHeight;
        harness.useFocusHeight = false;
        harness.camera.setFirstPerson(false);
        Frame skyward;
        (void) harness.draw(skyward, 0u);
        harness.camera.setFirstPerson(true);
        harness.useFocusHeight = true;
        harness.focusHeight = keptHeight;
        lpl::core::u32 skyAbove = 0u;
        for (lpl::core::u32 i = 0u; i < kWidth * kHeight; ++i)
        {
            const lpl::core::u32 pixel = skyward.colour[i];
            if ((pixel & 0xFFu) > ((pixel >> 16) & 0xFFu) + 24u)
                ++skyAbove;
        }
        std::printf("    %u pixels bluer than the rock underground, %u on the surface path, %u lit of %u\n", skyish,
                    skyAbove, rocky, kWidth * kHeight);
        check(skyish == 0u, "standing under a hill, no pixel of the frame is sky");
        check(skyAbove > 0u, "and the same eye on the surface path does see sky");
        check(rocky > (kWidth * kHeight) / 20u, "and the lamp reaches enough of it to see by");

        checkDoorwayShowsACaveFromOutside(harness, located);
        checkLampIsASpotNotAGlow(harness, insideX, insideZ, span);
    }
}

/**
 * @brief The lamp at night.
 *
 * @details The same carried light, on the surface after dark. Two claims, and neither is provable from
 *          one frame: that night is dark at all, and that the lamp puts a DIRECTED pool of light on the
 *          ground rather than a uniform lift. So three frames — noon, midnight, and midnight turned a
 *          quarter — and the comparisons between them.
 *
 *          Day is whatever the configured hour is: noon-ish. Night is half a day on, so the sun is under
 *          the horizon and SunState::intensity is zero. The lamp has an AXIS on the surface too: a
 *          uniform night-time lift would pass "night is darker" and leave you unable to tell which way
 *          the path goes. The picture LPL_DUMP_NIGHT_PPM asks for has day above and night below, so the
 *          lamp is judged against the same view.
 */
void checkLampAtNight()
{
    std::printf("== the lamp at night ==\n");
    Harness harness{0u, false};
    harness.fill();
    harness.camera.setFirstPerson(true);
    harness.camera.setPitch(0.0f);
    harness.camera.setYaw(0.0f);

    harness.surface.advance(0.0f, 0.0f);
    Frame day;
    (void) harness.draw(day, 0u);

    harness.surface.advance(0.5f, 0.0f);
    Frame night;
    (void) harness.draw(night, 0u);
    harness.camera.setYaw(1.57079633f);
    Frame nightAside;
    (void) harness.draw(nightAside, 0u);

    const auto brightness = [](const std::vector<lpl::core::u32> &pixels) {
        lpl::core::u64 total = 0u;
        for (lpl::core::u32 pixel : pixels)
            total += ((pixel >> 16) & 0xFFu) + ((pixel >> 8) & 0xFFu) + (pixel & 0xFFu);
        return total / (pixels.size() * 3u);
    };
    const lpl::core::u64 dayLight = brightness(day.colour);
    const lpl::core::u64 nightLight = brightness(night.colour);
    std::printf("    mean brightness: %llu by day, %llu by night\n", static_cast<unsigned long long>(dayLight),
                static_cast<unsigned long long>(nightLight));
    check(nightLight < dayLight, "night is darker than day");
    check(foldFrame(night.colour) != foldFrame(nightAside.colour), "and turning at night moves the light with you");

    if (const char *dump = std::getenv("LPL_DUMP_NIGHT_PPM"); dump != nullptr)
    {
        std::FILE *out = std::fopen(dump, "wb");
        if (out != nullptr)
        {
            std::fprintf(out, "P6\n%u %u\n255\n", kWidth, kHeight * 2u);
            for (const std::vector<lpl::core::u32> *frame : {&day.colour, &night.colour})
                for (lpl::core::u32 i = 0u; i < kWidth * kHeight; ++i)
                {
                    const lpl::core::u32 pixel = (*frame)[i];
                    const unsigned char rgb[3] = {static_cast<unsigned char>((pixel >> 16) & 0xFFu),
                                                  static_cast<unsigned char>((pixel >> 8) & 0xFFu),
                                                  static_cast<unsigned char>(pixel & 0xFFu)};
                    std::fwrite(rgb, 1u, 3u, out);
                }
            std::fclose(out);
        }
    }
}

/**
 * @brief The showcase world stands on measured earth, not only on noise.
 *
 * @warning **The check that stops the whole relief chain being an inert field on a struct.** Everything
 *          before it is exercised by gates and tests of their own; this asks whether the ground the
 *          client walks on is actually changed by the survey. A world that carried one and ignored it
 *          would draw the same picture and pass every other assertion in this file.
 *
 * @warning The plan this file ALREADY derived, not a second one. endlessPlanFromRecipe calibrates a river
 *          threshold by bisecting over a nine-chunk window -- the most expensive thing here by an order
 *          of magnitude -- and a second call is both a second bill and a second answer to how this world
 *          is scaled. Building one cost this test ninety seconds and is exactly the mistake walkedPlan()
 *          exists to prevent.
 *
 * The origin is measured ground and somewhere far away is not. Both must hold, or the survey covers
 * everything or nothing.
 *
 * @warning The window sits at its TRUE projection cells -- 0 to side-1 -- and NOT centred on the origin,
 *          because a centred one spans negative columns and a closed world folds those to the far side of
 *          the planet. So the origin is the survey's north-west corner, and the sampling has to look where
 *          the ground actually is rather than around zero.
 *
 * @warning The ground must actually DIFFER from the world without it. A survey that is read and then
 *          contributes nothing is indistinguishable from no survey at all.
 *
 * And far outside the window the two worlds must AGREE again, or the survey is leaking into country it
 * does not cover. The window carries bathymetry, so both sides of sea level must appear. A world entirely
 * above water never exercises the reconciliation the projection exists to make.
 *
 * @warning **The bounded map must stand on the SAME survey.** Before this, pressing O to look at the map
 *          drew invented ground while the world underfoot was measured -- a map of somewhere else, shown
 *          in the one place a player compares the two. A bounded build with the survey injected must
 *          differ from one without it, at the same cells.
 */
void checkShowcaseWorldStandsOnMeasuredEarth()
{
    std::printf("-- the showcase world stands on measured earth, not only on noise\n");
    const lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    const lpl::procgen::EndlessPlan &plan = walkedPlan();

    static const lpl::math::ReliefField field = lpl::samples::makeRealReliefField(recipe, plan.chunk.noise.amplitude);
    check(field.valid(), "the checked-in survey is a usable field");

    lpl::math::ReliefMosaic mosaic{};
    check(mosaic.add(&field), "and it goes into a mosaic");

    const lpl::core::i32 side = static_cast<lpl::core::i32>(lpl::samples::kReliefBlobSide);
    lpl::math::Fixed32 measured{};
    check(mosaic.heightAt(0, 0, measured), "the origin is on measured ground");
    check(mosaic.heightAt(side / 2, side / 2, measured), "and so is the middle of the survey");
    check(!mosaic.heightAt(100000, 100000, measured), "and far away is not");

    lpl::procgen::ChunkParams params = plan.chunk;
    params.relief = &mosaic;
    params.reliefDetail = params.noise;
    params.reliefDetail.amplitude = params.noise.amplitude * 0.12f;
    params.reliefDetail.frequency = params.noise.frequency * 3.0f;
    params.reliefDetail.octaves = 3u;

    lpl::core::u32 differing = 0u;
    lpl::core::u32 sampled = 0u;
    for (lpl::core::i32 z = 2; z < side - 2; z += 4)
    {
        for (lpl::core::i32 x = 2; x < side - 2; x += 4)
        {
            ++sampled;
            if (lpl::procgen::sampleWorldHeight(params, x, z).raw() !=
                lpl::procgen::sampleWorldHeight(plan.chunk, x, z).raw())
                ++differing;
        }
    }
    check(differing == sampled, "the survey changes the ground it stands on");
    std::printf("     %u of %u sampled cells differ from the survey-less world\n", differing, sampled);

    check(lpl::procgen::sampleWorldHeight(params, 100000, 100000).raw() ==
              lpl::procgen::sampleWorldHeight(plan.chunk, 100000, 100000).raw(),
          "and leaves the country beyond it alone");

    lpl::core::u32 wet = 0u;
    lpl::core::u32 dry = 0u;
    const lpl::math::Fixed32 sea = lpl::math::Fixed32::fromFloat(recipe.biomes.seaLevel);
    for (lpl::core::i32 z = 0; z < side; z += 2)
        for (lpl::core::i32 x = 0; x < side; x += 2)
            (lpl::procgen::sampleWorldHeight(params, x, z) <= sea ? wet : dry)++;
    check(wet > 0u, "there is sea in the measured window");
    check(dry > 0u, "and land");

    lpl::procgen::WorldRecipe bounded = recipe;
    bounded.materializeGround = false;
    lpl::procgen::ReliefBlend blend{};
    blend.mosaic = &mosaic;
    blend.detail = bounded.terrain;
    blend.detail.amplitude = bounded.terrain.amplitude * 0.12f;
    blend.detail.frequency = bounded.terrain.frequency * 3.0f;
    blend.detail.octaves = 3u;
    blend.seed = bounded.seed;

    lpl::pmr::vector<lpl::ecs::EntityId> ids;
    const lpl::procgen::WorldSnapshot plain =
        lpl::procgen::buildSnapshot(bounded, nullptr, &ids, lpl::procgen::WalkabilityRule{}, nullptr);
    lpl::pmr::vector<lpl::ecs::EntityId> ids2;
    const lpl::procgen::WorldSnapshot surveyed =
        lpl::procgen::buildSnapshot(bounded, nullptr, &ids2, lpl::procgen::WalkabilityRule{}, &blend);

    check(plain.width == surveyed.width && plain.depth == surveyed.depth, "the two bounded builds are the same shape");
    lpl::core::u32 mapDiffer = 0u;
    lpl::core::u32 mapSampled = 0u;
    const lpl::core::u32 span =
        plain.width < static_cast<lpl::core::u32>(side) ? plain.width : static_cast<lpl::core::u32>(side);
    for (lpl::core::u32 z = 0u; z < span; z += 2u)
    {
        for (lpl::core::u32 x = 0u; x < span; x += 2u)
        {
            ++mapSampled;
            if (plain.height.at(x, z).raw() != surveyed.height.at(x, z).raw())
                ++mapDiffer;
        }
    }
    check(mapSampled > 0u, "the map overlaps the survey at all");
    check(mapDiffer > mapSampled / 2u, "the bounded map stands on the survey too");
    std::printf("     bounded map: %u of %u sampled cells differ from the survey-less build\n", mapDiffer, mapSampled);
    std::printf("     %u wet, %u dry cells across the measured window\n", wet, dry);
}

} // namespace

/**
 * @brief What one frame of a streamed world is actually made of.
 *
 * @details `engine::TerrainRenderer`, `engine::TerrainSurface` and `engine::TerrainStreamer` are
 *          templates instantiated by exactly one translation unit in the whole project —
 *          `libengine/src/client_app.cpp`, which only the KERNEL builds. So the entire terrain render
 *          path had no host target at all: a typo in it compiled fine on Linux and failed three build
 *          paths later, and a behavioural change in it could only be checked by booting QEMU and
 *          looking.
 *
 *          This is that target. It renders into memory — no display, no Vulkan — and asserts the things
 *          a screenshot cannot: that water is drawn where there is water and nowhere else, that the
 *          reflection pass is skipped when nothing can reflect, that a river is a surface rather than a
 *          colour, and that the swell is geometry when a host pays for it.
 *
 *          It also INCLUDES `samples/TerrainWorld.hpp`, and that one line is worth as much as the
 *          assertions. `TerrainWorld` is a plain class whose members are all defined inline, in its
 *          header and the `.inl` that header includes, so a host translation unit that merely includes
 *          it type-checks every one of them. Before that it had exactly one consumer in the whole
 *          project — the kernel's `client_app.cpp` — so a `Fixed32` compared against a `float`, or a
 *          member read off the wrong struct, compiled clean on Linux and failed twenty minutes later in
 *          a cross build. That happened three times in one afternoon.
 *
 *          The nearest window that HAS a sea, and the nearest that HAS a river, are found once, for the
 *          checks that need water. They are not the same place and neither is the origin.
 */
int main()
{
    checkSampleWorldTypeChecksOnTheHost();

    std::printf("== a streamed frame ==\n");
    const lpl::procgen::ChunkCoord seaFocus =
        findChunk(kSeaLift, [](const lpl::procgen::ChunkTerrain &t) { return t.hasSea; });
    const lpl::procgen::ChunkCoord riverFocus =
        findChunk(0.0f, [](const lpl::procgen::ChunkTerrain &t) { return t.hasRiver; });
    std::printf("    sea window at chunk (%d, %d), river window at (%d, %d)\n", seaFocus.x, seaFocus.z, riverFocus.x,
                riverFocus.z);

    checkPathCompilesAndDrawsSomething(seaFocus);
    checkSeaExtentIsTheSeasExtent(seaFocus);
    checkReflectionPassIsSkippedWhenNothingReflects(seaFocus);
    checkRiverIsASurfaceNotAColour(riverFocus);
    checkSwellIsGeometryWhenTheHostPaysForIt(seaFocus);
    checkFrameIsTheSameFrameTwice(seaFocus);
    checkProbeObeysTheHost(seaFocus);
    checkPartlyDrownedChunkIsBoundedTightly();
    checkUnderground();
    checkLampAtNight();
    checkShowcaseWorldStandsOnMeasuredEarth();

    std::printf("\n%s (%d failures, %d checks)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, checks);
    return failures == 0 ? 0 : 1;
}
