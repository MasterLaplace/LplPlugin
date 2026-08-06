/**
 * @file test_terrain_render.cpp
 * @brief What one frame of a streamed world is actually made of.
 *
 * `engine::TerrainRenderer`, `engine::TerrainSurface` and `engine::TerrainStreamer` are
 * templates instantiated by exactly one translation unit in the whole project —
 * `libengine/src/client_app.cpp`, which only the KERNEL builds. So the entire terrain
 * render path had no host target at all: a typo in it compiled fine on Linux and failed
 * three build paths later, and a behavioural change in it could only be checked by
 * booting QEMU and looking.
 *
 * This is that target. It renders into memory — no display, no Vulkan — and asserts the
 * things a screenshot cannot: that water is drawn where there is water and nowhere else,
 * that the reflection pass is skipped when nothing can reflect, that a river is a surface
 * rather than a colour, and that the swell is geometry when a host pays for it.
 *
 * It also INCLUDES `samples/TerrainWorld.hpp`, and that one line is worth as much as the
 * assertions. `TerrainWorld` is a plain class whose members are defined in its body, so a
 * host translation unit that merely includes it type-checks all nineteen hundred lines of
 * it. Before that it had exactly one consumer in the whole project — the kernel's
 * `client_app.cpp` — so a `Fixed32` compared against a `float`, or a member read off the
 * wrong struct, compiled clean on Linux and failed twenty minutes later in a cross build.
 * That happened three times in one afternoon.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

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

using namespace lpl;

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

constexpr core::u32 kWidth = 192u;
constexpr core::u32 kHeight = 120u;
constexpr core::u32 kChunkSize = 24u;

/**
 * @brief How far the fixture raises the sea so the origin window HAS water in it.
 *
 * ⚠ Not a tuning knob, a fixture requirement, and it came back from the dead once. The
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
constexpr core::f32 kSeaLift = 6.0f;

/// A frame's worth of memory, and the only render target this test has.
struct Frame {
    std::vector<core::u32> colour;
    std::vector<core::f32> depth;

    Frame()
        : colour(static_cast<std::size_t>(kWidth) * kHeight, 0u),
          depth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f)
    {
    }

    [[nodiscard]] render::RenderTarget target() noexcept
    {
        return render::RenderTarget{&colour[0], &depth[0], kWidth, kHeight};
    }
};

/// Fold of a frame, so "the picture changed" is a fact rather than an impression.
[[nodiscard]] core::u32 foldFrame(const std::vector<core::u32> &pixels) noexcept
{
    core::u32 hash = 0x811C9DC5u;
    for (core::u32 pixel : pixels)
        for (core::u32 byte = 0u; byte < 4u; ++byte)
        {
            hash ^= (pixel >> (byte * 8u)) & 0xFFu;
            hash *= 0x01000193u;
        }
    return hash;
}

/// The biome palette the sample uses, reduced to what a fold needs.
[[nodiscard]] core::u32 palette(procgen::BiomeId biome) noexcept
{
    return 0x00203040u + static_cast<core::u32>(biome) * 0x00101010u;
}

/**
 * @brief The walked plan, derived ONCE.
 *
 * procgen::endlessPlanFromRecipe calibrates the river threshold, and a calibration is a
 * bisection over a nine-chunk window — the most expensive thing in this file by an order
 * of magnitude. Six harnesses paying for it six times turned a two-second test into a
 * ninety-second one, which is how a test stops being run.
 */
[[nodiscard]] const procgen::EndlessPlan &walkedPlan()
{
    static const procgen::EndlessPlan plan = procgen::endlessPlanFromRecipe(procgen::parityWorldRecipe(), kChunkSize);
    return plan;
}

/// A world configured the way the ring-0 client is, minus the display.
struct Harness {
    procgen::EndlessPlan plan{};
    engine::TerrainStreamer streamer{};
    engine::TerrainSurface surface{};
    engine::PropLibrary props{};
    ecs::Registry registry{};
    engine::TerrainRenderer renderer{};
    render::OrbitCamera camera{};

    /**
     * @brief Streams the window around @p focusChunk rather than always around the origin.
     *
     * ⚠ Not a convenience. The walked world's landforms are about a hundred and ten cells
     * across and this window is a hundred and twenty, so it fits on ONE of them — and the
     * one at the origin has neither sea nor river in it. Every block below that measures
     * water was passing or failing on an empty set. The repository already knows this shape:
     * `testDeterminism` in test_procgen_chunking hunts for a chunk that carries water first,
     * because "comparing two empty masks proves only that nothing equals nothing".
     */
    explicit Harness(core::u32 tessellation, bool reflection, core::f32 seaLift = 0.0f,
                     procgen::ChunkCoord focusChunk = {0, 0})
    {
        plan = walkedPlan();
        // Raising the sea is how this test gets coverage of the sea path at all: the walked
        // parity world sits almost entirely above its own sea level, so at the recipe's
        // level exactly one chunk in twenty-five is wet and the tightening is never
        // exercised on a PARTIAL chunk — which is the case it exists for.
        plan.rule.seaLevel += seaLift;
        plan.rivers.seaLevel += seaLift;

        procgen::StreamingParams stream;
        stream.generateRadius = 2u;
        stream.maxGeneratePerTick = 64u;
        stream.maxReleasePerTick = 4u;
        streamer.configure(plan.chunk, plan.rivers, stream, 64u, plan.rule);

        const engine::Config config = engine::Config::Builder{}
                                          .enablePerPixelSurface(true)
                                          .enableTerrainShadows(false)
                                          .enableWaterReflection(reflection)
                                          .waterTessellation(tessellation)
                                          .build();
        engine::TerrainSurfaceParams look;
        look.seaLevel = plan.rule.seaLevel;
        surface.configure(config, look, plan.chunk.worldSeed);

        engine::PropLibraryParams propParams;
        props.build(propParams, plan.chunk.worldSeed);

        focus = focusChunk;
        camera.setFocus(0.0f, 0.0f);
    }

    /// Streams the chunks around the origin in, then reports how many hold water.
    procgen::ChunkCoord focus{0, 0};

    core::u32 fill()
    {
        const core::f32 fx = static_cast<core::f32>(focus.x * static_cast<core::i32>(kChunkSize));
        const core::f32 fz = static_cast<core::f32>(focus.z * static_cast<core::i32>(kChunkSize));
        camera.setFocus(fx, fz);
        for (core::u32 tick = 0u; tick < 64u; ++tick)
            streamer.update(fx, fz, 0.0f, 0.0f, [](engine::TerrainChunk &) {});
        core::u32 wet = 0u;
        for (core::u32 i = 0u; i < streamer.size(); ++i)
            if (streamer.at(i).hasSea)
                ++wet;
        return wet;
    }

    [[nodiscard]] engine::TerrainDrawParams drawParams() const
    {
        engine::TerrainDrawParams params;
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

    core::u32 caveDrawRadius{0u};
    bool useFocusHeight{false};
    core::f32 focusHeight{0.0f};

    /**
     * @brief Streams around a WORLD CELL rather than a chunk, and reports the cave there.
     *
     * A cave is sited on a landmark lattice that has nothing to do with the chunk grid,
     * so a fixture that streamed around a chunk coordinate would be hunting for one.
     */
    const procgen::CaveWarren *fillAtCell(core::i32 cellX, core::i32 cellZ)
    {
        const core::f32 fx = static_cast<core::f32>(cellX);
        const core::f32 fz = static_cast<core::f32>(cellZ);
        camera.setFocus(fx, fz);
        for (core::u32 tick = 0u; tick < 64u; ++tick)
            streamer.update(fx, fz, 0.0f, 0.0f, [](engine::TerrainChunk &) {});
        return streamer.warrenAt(cellX, cellZ);
    }

    core::u32 draw(Frame &frame, core::u32 tick)
    {
        return renderer.drawStreamed(frame.target(), camera, streamer, surface, props, registry, drawParams(), tick,
                                     palette, [this](core::i32 x, core::i32 z) { return streamer.groundAt(x, z); });
    }
};

/**
 * @brief The nearest chunk carrying what a block is about to measure.
 *
 * Spiral out from the origin until `carries` says yes. Without it every water assertion in
 * this file was taken on a window that happened to contain no water — and "every drowned
 * cell is inside its bounds" is trivially true of no cells at all.
 */
template <typename Carries> [[nodiscard]] procgen::ChunkCoord findChunk(core::f32 seaLift, Carries &&carries)
{
    procgen::EndlessPlan plan = walkedPlan();
    plan.rule.seaLevel += seaLift;
    plan.rivers.seaLevel += seaLift;
    for (core::i32 radius = 0; radius <= 14; ++radius)
        for (core::i32 cz = -radius; cz <= radius; ++cz)
            for (core::i32 cx = -radius; cx <= radius; ++cx)
            {
                if (cx > -radius && cx < radius && cz > -radius && cz < radius)
                    continue; // the ring only: the interior was tested at a smaller radius
                const procgen::ChunkTerrain terrain = procgen::generateChunkTerrain(
                    plan.chunk, plan.rivers, {cx, cz}, plan.rule, [](core::i32, core::i32) {});
                if (carries(terrain))
                    return {cx, cz};
            }
    return {0, 0};
}

} // namespace

int main()
{
    // ── The sample world type-checks on the host ─────────────────────────────
    //
    // Constructed, not run: running it needs an engine::WorldContext, and none of the
    // errors this catches need one. What is being exercised is the COMPILER — the include
    // above is the whole test — and this is the smallest thing that keeps the include from
    // being dropped as unused.
    std::printf("== the sample world ==\n");
    {
        samples::TerrainWorld world{procgen::parityWorldRecipe()};
        std::printf("    constructed \"%s\"\n", world.name());
        // A thin assertion on purpose: what is being checked here is that the TU COMPILED,
        // and the compiler has already answered by the time this line runs. Anything richer
        // would need an engine::WorldContext, and none of the three errors this catches did.
        check(world.name() != nullptr && world.name()[0] != '\0', "the sample world names itself");
    }

    // ── The path compiles and draws something ────────────────────────────────
    std::printf("== a streamed frame ==\n");
    // Found once, used by every block below: the nearest window that HAS a sea, and the
    // nearest that HAS a river. They are not the same place and neither is the origin.
    const procgen::ChunkCoord seaFocus = findChunk(kSeaLift, [](const procgen::ChunkTerrain &t) { return t.hasSea; });
    const procgen::ChunkCoord riverFocus = findChunk(0.0f, [](const procgen::ChunkTerrain &t) { return t.hasRiver; });
    std::printf("    sea window at chunk (%d, %d), river window at (%d, %d)\n", seaFocus.x, seaFocus.z, riverFocus.x,
                riverFocus.z);

    core::u32 flatTriangles = 0u;
    core::u32 wetChunks = 0u;
    core::u32 riverChunks = 0u;
    {
        Harness harness{0u, true, kSeaLift, seaFocus};
        wetChunks = harness.fill();
        for (core::u32 i = 0u; i < harness.streamer.size(); ++i)
            if (harness.streamer.at(i).hasRiver)
                ++riverChunks;

        Frame frame;
        flatTriangles = harness.draw(frame, 0u);
        std::printf("    %u chunks resident, %u hold sea, %u hold river; %u triangles\n", harness.streamer.size(),
                    wetChunks, riverChunks, flatTriangles);
        check(harness.streamer.size() > 0u, "chunks streamed in");
        check(flatTriangles > 0u, "the frame drew geometry");
    }

    // ── The sea's extent is the sea's extent ─────────────────────────────────
    //
    // `lowest` says whether a chunk holds water; the bounds say WHERE. A chunk that
    // reports sea must have a drowned cell inside its bounds, and every drowned cell
    // must be inside them — the second half is the one that would fail silently, by
    // clipping the ocean to a corner of itself.
    std::printf("== the sea's extent ==\n");
    {
        Harness harness{0u, true, kSeaLift, seaFocus};
        harness.fill();
        core::u32 checkedChunks = 0u;
        core::u32 outside = 0u;
        core::u32 insideEmpty = 0u;
        for (core::u32 i = 0u; i < harness.streamer.size(); ++i)
        {
            const engine::TerrainChunk &chunk = harness.streamer.at(i);
            if (chunk.height.empty())
                continue;
            core::u32 drowned = 0u;
            core::u32 inBounds = 0u;
            for (core::u32 z = 0u; z < kChunkSize; ++z)
                for (core::u32 x = 0u; x < kChunkSize; ++x)
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
        std::printf("    %u chunks with drowned cells, %u mis-bounded, %u claiming sea with none\n", checkedChunks,
                    outside, insideEmpty);
        // FIRST, or the two checks below are true of nothing: a window with no water passes
        // "every drowned cell is inside the bounds" without touching a single bound.
        check(checkedChunks > 0u, "the fixture window actually contains drowned cells");
        check(outside == 0u, "every drowned cell is inside the reported bounds");
        check(insideEmpty == 0u, "no chunk reports sea it does not have");
    }

    // ── The reflection pass is skipped when nothing reflects ─────────────────
    //
    // The probe is a whole second render of the world, amortised over four frames. It
    // used to run over dry land forever because nothing ever asked whether there was
    // water on screen. The measurement is the probe's OWN validity: a pass that ran
    // leaves a valid probe behind, and one that was skipped does not.
    //
    // The parity world is almost entirely above its sea level, which is what makes this
    // measurable at all — and is worth saying rather than assuming, because a world with
    // an ocean in every chunk would pass this check by never exercising it.
    std::printf("== the reflection probe ==\n");
    {
        Harness harness{0u, true, kSeaLift, seaFocus};
        const core::u32 wet = harness.fill();
        std::vector<core::u32> probeColour(static_cast<std::size_t>(64u) * 48u, 0u);
        std::vector<core::f32> probeDepth(static_cast<std::size_t>(64u) * 48u, 0.0f);
        harness.surface.attachProbe(&probeColour[0], &probeDepth[0], 64u, 48u);

        Frame frame;
        harness.draw(frame, 0u);
        std::printf("    %u wet chunks resident; probe %s\n", wet, harness.surface.probeValid() ? "ran" : "skipped");
        // Whichever way the world came out, the two must AGREE. That is the invariant;
        // "the probe never runs" would be satisfied by a probe that is simply broken.
        check(harness.surface.probeValid() == (wet > 0u), "the probe runs exactly when water is in sight");
    }

    // ── A river is a surface, not a colour ───────────────────────────────────
    //
    // River cells used to be Lake-coloured cells of GROUND: the blue on screen was
    // terrain that happened to be blue, with no reflection and no depth. Now each one is
    // a quad standing in its carved bed, so asking for it must produce triangles that
    // asking for no river does not.
    std::printf("== river water ==\n");
    {
        // NO lift here, and that is the point of the block: a river is water standing ABOVE
        // the sea in a carved channel, so raising the sea six metres to guarantee an ocean —
        // which is what every other block needs — puts every river cell under it and the
        // renderer rightly draws none. A fixture that drowns the thing it measures reports
        // "no river water" and blames the code.
        Harness dry{0u, false, 0.0f, riverFocus};
        dry.fill();
        Frame dryFrame;
        engine::TerrainDrawParams noRiver = dry.drawParams();
        noRiver.riverSurfaceRise = 0.0f;
        const core::u32 without = dry.renderer.drawStreamed(
            dryFrame.target(), dry.camera, dry.streamer, dry.surface, dry.props, dry.registry, noRiver, 0u, palette,
            [&dry](core::i32 x, core::i32 z) { return dry.streamer.groundAt(x, z); });

        Harness wet{0u, false, 0.0f, riverFocus};
        wet.fill();
        Frame wetFrame;
        const core::u32 with = wet.draw(wetFrame, 0u);
        std::printf("    %u triangles without river water, %u with\n", without, with);
        check(wet.streamer.at(0u).hasRiver || with > without, "the fixture window actually contains river");
        check(with > without, "asking for river water draws more than not asking");
        check(foldFrame(wetFrame.colour) != foldFrame(dryFrame.colour), "and it changes the picture");
    }

    // ── The swell is geometry when the host pays for it ──────────────────────
    //
    // Two conditions gate the displaced mesh and BOTH are necessary: a world with no
    // swell has nothing to displace, and a host that declined the cost gets the flat
    // quad. Each is asserted on its own, because either alone passing would look like
    // the feature working.
    std::printf("== the displaced surface ==\n");
    {
        Harness flat{0u, false, kSeaLift, seaFocus};
        flat.fill();
        Frame flatFrame;
        flat.surface.water().swellHeight = 0.9f;
        const core::u32 flatCount = flat.draw(flatFrame, 0u);

        Harness tess{6u, false, kSeaLift, seaFocus};
        tess.fill();
        Frame tessFrame;
        tess.surface.water().swellHeight = 0.9f;
        const core::u32 tessCount = tess.draw(tessFrame, 0u);

        Harness stillFlat{6u, false, kSeaLift, seaFocus};
        stillFlat.fill();
        Frame stillFrame;
        stillFlat.surface.water().swellHeight = 0.0f;
        const core::u32 stillCount = stillFlat.draw(stillFrame, 0u);

        std::printf("    tessellation 0: %u triangles, 6: %u, 6 with no swell: %u\n", flatCount, tessCount, stillCount);
        check(tessCount > flatCount, "a paid-for tessellation spends more triangles");
        check(stillCount == flatCount, "and a world with no swell spends none of them");
    }

    // ── The frame is the same frame twice ────────────────────────────────────
    //
    // Not a cross-target gate — this path is float and non-authoritative by contract —
    // but a rendered frame must at least be a function of the world and the tick. A
    // renderer that reads uninitialised memory or iterates a hash map passes everything
    // above and fails this.
    std::printf("== determinism ==\n");
    {
        Harness first{6u, true, kSeaLift, seaFocus};
        first.fill();
        Frame frameA;
        first.draw(frameA, 3u);

        Harness second{6u, true, kSeaLift, seaFocus};
        second.fill();
        Frame frameB;
        second.draw(frameB, 3u);

        const core::u32 a = foldFrame(frameA.colour);
        const core::u32 b = foldFrame(frameB.colour);
        std::printf("    fold 0x%08X against 0x%08X\n", a, b);
        check(a == b, "the same world at the same tick renders the same pixels");
    }

    // ── The probe obeys the host, not only the world ─────────────────────────
    //
    // Two independent reasons to skip the pass, and a test that only covered one would
    // let the other rot: the world may have no water, and the host may have declined
    // reflections outright.
    std::printf("== the probe obeys the host ==\n");
    {
        Harness harness{0u, false, kSeaLift, seaFocus};
        harness.fill();
        std::vector<core::u32> probeColour(static_cast<std::size_t>(64u) * 48u, 0u);
        std::vector<core::f32> probeDepth(static_cast<std::size_t>(64u) * 48u, 0.0f);
        harness.surface.attachProbe(&probeColour[0], &probeDepth[0], 64u, 48u);
        Frame frame;
        harness.draw(frame, 0u);
        check(!harness.surface.probeValid(), "a host that declined reflections gets no probe pass");
    }

    // ── A partly drowned chunk is bounded tightly ────────────────────────────
    //
    // The case the tightening exists for, and the one the recipe's own sea level cannot
    // produce. With the sea raised, most chunks are PARTLY under it — so at least one
    // must report bounds strictly inside its own extent. Without this the tightening
    // could be returning the whole chunk every time and every other check would pass.
    std::printf("== a coast, not an ocean ==\n");
    {
        // The world's OWN sea, not a raised one. This block used to lift the sea twelve
        // metres to manufacture partly drowned chunks, because the walked world had none:
        // it was entirely above its own sea level. It has a coastline now, so lifting the
        // sea drowns every chunk edge to edge and the tightening stops being exercised —
        // the check would pass by never testing anything.
        Harness harness{0u, true, kSeaLift};
        const core::u32 wet = harness.fill();
        core::u32 partial = 0u;
        core::u32 full = 0u;
        for (core::u32 i = 0u; i < harness.streamer.size(); ++i)
        {
            const engine::TerrainChunk &chunk = harness.streamer.at(i);
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

    // ── Underground ──────────────────────────────────────────────────────────
    //
    // The render path had no coverage below the surface at all, and it is the half of
    // the cave feature a signature cannot reach: the parity gate proves a body gets
    // inside and stays where it should, and says nothing about whether anything is
    // drawn once it is there. "The cave is unlit" and "I never got in" look identical
    // from a dark screen, so each is measured separately here.
    std::printf("== inside a cave ==\n");
    {
        Harness harness{0u, false};
        // The doorway of a real cave in this world, found rather than named: which
        // landmark cells carry one is a property of the terrain, and a constant here
        // would quietly stop pointing at a cave the day the terrain moved.
        const procgen::EndlessPlan &plan = harness.plan;
        procgen::CaveWarren located;
        for (core::i32 lz = -6; lz <= 6 && !located.valid; ++lz)
            for (core::i32 lx = -6; lx <= 6; ++lx)
            {
                procgen::LandmarkSite site;
                if (!procgen::landmarkAt(plan.chunk, plan.rule.caveMouths, procgen::LandmarkKind::CaveMouth,
                                         plan.rule.seaLevel, lx, lz, site))
                    continue;
                procgen::CaveWarren candidate =
                    procgen::buildCaveWarren(plan.chunk, site, plan.rule.warren, plan.rule.caveMouthDrop);
                if (candidate.valid)
                {
                    located = static_cast<procgen::CaveWarren &&>(candidate);
                    break;
                }
            }
        check(located.valid, "the walked world has a cave to stand in");

        // A cell a few steps INSIDE the doorway, so the eye is under the rock rather
        // than in the trench looking at it.
        const core::i32 insideX = located.apertureX[0] + located.adit.stepX * 3;
        const core::i32 insideZ = located.apertureZ[0] + located.adit.stepZ * 3;
        const procgen::CaveWarren *resident = harness.fillAtCell(insideX, insideZ);
        check(resident != nullptr, "and the chunk that owns it is resident where a body would be");

        if (resident != nullptr)
        {
            const procgen::VerticalSpan span =
                harness.streamer.spanAt(insideX, insideZ, math::Fixed32::fromFloat(located.adit.floorY + 0.5f));
            check(span.enclosed, "the collider agrees there is rock overhead there");

            // FIRST PERSON, and this is the check that caught itself being useless. The
            // harness camera defaults to an orbit, so it stands well back and above its
            // focus — outside the hill, looking at it. Every cave face was drawn and
            // every one lost the depth test to the terrain in front of it: 2194
            // triangles submitted and ZERO pixels different from the control. The
            // "no sky" assertion passed anyway, off the colour beginCaveFrame clears
            // to, which is a check satisfied by the thing it was meant to rule out.
            harness.camera.setFirstPerson(true);
            harness.camera.setPitch(0.0f);
            harness.camera.setYaw(0.0f);
            harness.camera.setEyeHeight(1.6f);
            harness.camera.setFocus(static_cast<core::f32>(insideX), static_cast<core::f32>(insideZ));
            harness.caveDrawRadius = 12u;
            harness.useFocusHeight = true;
            harness.focusHeight = span.floor.toFloat();

            Frame lit;
            const core::u32 caveTriangles = harness.draw(lit, 0u);

            // And the control: the SAME eye, the same world, with the host budget at
            // zero. Without it "triangles were drawn" is satisfied by the terrain the
            // pass would have drawn anyway, and the cave geometry could be absent.
            harness.caveDrawRadius = 0u;
            Frame dark;
            const core::u32 bareTriangles = harness.draw(dark, 0u);

            core::u32 differing = 0u;
            for (core::u32 i = 0u; i < kWidth * kHeight; ++i)
                differing += lit.colour[i] != dark.colour[i] ? 1u : 0u;
            std::printf("    %u triangles with the cave drawn, %u without; %u pixels differ; folds %08X %08X\n",
                        caveTriangles, bareTriangles, differing, foldFrame(lit.colour), foldFrame(dark.colour));
            check(caveTriangles > bareTriangles, "asking for cave geometry draws more than not asking");
            check(foldFrame(lit.colour) != foldFrame(dark.colour), "and it changes the picture");

            // No sky. The frame begins with beginCaveFrame, so what a pixel the geometry
            // misses shows is the cave's own dark — not the blue the surface path clears
            // to. Counted rather than sampled at one pixel: a single probe would land on
            // whatever happened to be there.
            core::u32 skyish = 0u;
            core::u32 rocky = 0u;
            for (core::u32 i = 0u; i < kWidth * kHeight; ++i)
            {
                const core::u32 pixel = lit.colour[i];
                const core::u32 red = (pixel >> 16) & 0xFFu;
                const core::u32 blue = pixel & 0xFFu;
                if (blue > red + 24u)
                    ++skyish;
                if (red > 24u || blue > 24u)
                    ++rocky;
            }
            // The control for the control. "No pixel is sky" could hold because the
            // palette happens to be warm, so the SAME world is rendered from an eye on
            // the surface — which must show sky, or the test is measuring a colour
            // scheme rather than a roof. The eye decides now, so this is a move rather
            // than a flag: put it above ground and the sky comes back on its own.
            const core::f32 keptHeight = harness.focusHeight;
            harness.useFocusHeight = false;
            harness.camera.setFirstPerson(false);
            Frame skyward;
            (void) harness.draw(skyward, 0u);
            harness.camera.setFirstPerson(true);
            harness.useFocusHeight = true;
            harness.focusHeight = keptHeight;
            core::u32 skyAbove = 0u;
            for (core::u32 i = 0u; i < kWidth * kHeight; ++i)
            {
                const core::u32 pixel = skyward.colour[i];
                if ((pixel & 0xFFu) > ((pixel >> 16) & 0xFFu) + 24u)
                    ++skyAbove;
            }
            std::printf("    %u pixels bluer than the rock underground, %u on the surface path, %u lit of %u\n", skyish,
                        skyAbove, rocky, kWidth * kHeight);
            check(skyish == 0u, "standing under a hill, no pixel of the frame is sky");
            check(skyAbove > 0u, "and the same eye on the surface path does see sky");
            check(rocky > (kWidth * kHeight) / 20u, "and the lamp reaches enough of it to see by");

            // ── And from OUTSIDE, which is the half a player meets first ─────
            //
            // Standing in the trench looking at the doorway. This is the view that was
            // reported as broken and it had no coverage at all: every check above puts
            // the eye inside, where the geometry is unavoidably in front of it.
            const core::i32 outX = located.apertureX[0] - located.adit.stepX * 5;
            const core::i32 outZ = located.apertureZ[0] - located.adit.stepZ * 5;
            harness.fillAtCell(outX, outZ);
            harness.camera.setFocus(static_cast<core::f32>(outX), static_cast<core::f32>(outZ));
            // Facing the doorway: the camera walks -sin(yaw), -cos(yaw), same convention
            // as the body, so looking along the adit is atan2 of the negated step.
            harness.camera.setYaw(math::Cordic::atan2(math::Fixed32::fromInt(-located.adit.stepX),
                                                      math::Fixed32::fromInt(-located.adit.stepZ))
                                      .toFloat());
            harness.focusHeight = located.adit.floorY;
            harness.caveDrawRadius = 12u;
            Frame porch;
            const core::u32 porchTriangles = harness.draw(porch, 0u);
            harness.caveDrawRadius = 0u;
            Frame plain;
            const core::u32 plainTriangles = harness.draw(plain, 0u);
            core::u32 mouthPixels = 0u;
            for (core::u32 i = 0u; i < kWidth * kHeight; ++i)
                mouthPixels += porch.colour[i] != plain.colour[i] ? 1u : 0u;
            std::printf("    from the trench: %u vs %u triangles, %u pixels of cave visible\n", porchTriangles,
                        plainTriangles, mouthPixels);
            std::printf("    (a doorway shows a cave through it: %s)\n", mouthPixels > 200u ? "yes" : "NO");
            check(mouthPixels > 200u, "the doorway is a hole you can see a cave through");
            harness.caveDrawRadius = 12u;

            // ── The lamp is a SPOT, not a glow ───────────────────────────────
            //
            // A bulb at the eye lights everything around you equally, which is a cave
            // with no direction to walk in. The claim is that the beam has an AXIS, and
            // the only way to assert that is to turn the eye and require the picture to
            // change: a brightness measurement alone is satisfied by a glow.
            harness.camera.setFocus(static_cast<core::f32>(insideX), static_cast<core::f32>(insideZ));
            harness.focusHeight = span.floor.toFloat();
            harness.camera.setYaw(0.0f);
            Frame ahead;
            (void) harness.draw(ahead, 0u);
            harness.camera.setYaw(1.57079633f); // a quarter turn: same eye, same world
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
                    for (core::u32 i = 0u; i < kWidth * kHeight; ++i)
                    {
                        const core::u32 pixel = ahead.colour[i];
                        const unsigned char rgb[3] = {static_cast<unsigned char>((pixel >> 16) & 0xFFu),
                                                      static_cast<unsigned char>((pixel >> 8) & 0xFFu),
                                                      static_cast<unsigned char>(pixel & 0xFFu)};
                        std::fwrite(rgb, 1u, 3u, out);
                    }
                    std::fclose(out);
                }
            }
        }
    }

    // ── The lamp at night ────────────────────────────────────────────────────
    //
    // The same carried light, on the surface after dark. Two claims, and neither is
    // provable from one frame: that night is dark at all, and that the lamp puts a
    // DIRECTED pool of light on the ground rather than a uniform lift. So three frames —
    // noon, midnight, and midnight turned a quarter — and the comparisons between them.
    std::printf("== the lamp at night ==\n");
    {
        Harness harness{0u, false};
        harness.fill();
        harness.camera.setFirstPerson(true);
        harness.camera.setPitch(0.0f);
        harness.camera.setYaw(0.0f);

        harness.surface.advance(0.0f, 0.0f); // whatever the configured hour is: noon-ish
        Frame day;
        (void) harness.draw(day, 0u);

        // Half a day on, so the sun is under the horizon and SunState::intensity is zero.
        harness.surface.advance(0.5f, 0.0f);
        Frame night;
        (void) harness.draw(night, 0u);
        harness.camera.setYaw(1.57079633f);
        Frame nightAside;
        (void) harness.draw(nightAside, 0u);

        const auto brightness = [](const std::vector<core::u32> &pixels) {
            core::u64 total = 0u;
            for (core::u32 pixel : pixels)
                total += ((pixel >> 16) & 0xFFu) + ((pixel >> 8) & 0xFFu) + (pixel & 0xFFu);
            return total / (pixels.size() * 3u);
        };
        const core::u64 dayLight = brightness(day.colour);
        const core::u64 nightLight = brightness(night.colour);
        std::printf("    mean brightness: %llu by day, %llu by night\n", static_cast<unsigned long long>(dayLight),
                    static_cast<unsigned long long>(nightLight));
        check(nightLight < dayLight, "night is darker than day");
        // The lamp has an AXIS on the surface too. A uniform night-time lift would pass
        // "night is darker" and leave you unable to tell which way the path goes.
        check(foldFrame(night.colour) != foldFrame(nightAside.colour),
              "and turning at night moves the light with you");

        if (const char *dump = std::getenv("LPL_DUMP_NIGHT_PPM"); dump != nullptr)
        {
            std::FILE *out = std::fopen(dump, "wb");
            if (out != nullptr)
            {
                // Day above, night below, so the lamp is judged against the same view.
                std::fprintf(out, "P6\n%u %u\n255\n", kWidth, kHeight * 2u);
                for (const std::vector<core::u32> *frame : {&day.colour, &night.colour})
                    for (core::u32 i = 0u; i < kWidth * kHeight; ++i)
                    {
                        const core::u32 pixel = (*frame)[i];
                        const unsigned char rgb[3] = {static_cast<unsigned char>((pixel >> 16) & 0xFFu),
                                                      static_cast<unsigned char>((pixel >> 8) & 0xFFu),
                                                      static_cast<unsigned char>(pixel & 0xFFu)};
                        std::fwrite(rgb, 1u, 3u, out);
                    }
                std::fclose(out);
            }
        }
    }

    std::printf("\n%s (%d failures, %d checks)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, checks);
    return failures == 0 ? 0 : 1;
}
