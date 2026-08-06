/**
 * @file TerrainSurface.hpp
 * @brief The engine's surface layer: sky, light, grain, shadows, water, reflection.
 *
 * Every one of these was written inside a sample, and none of them is about that
 * sample: a game that wants water wants Fresnel and a rippled normal; a game that
 * wants shadows wants the terrain to shade itself and its props; a game that wants
 * a procedural surface wants grain with a mip chain and a level chosen by distance.
 * What differs between games is which colour a biome is and where the ground is —
 * two callbacks — not any of the arithmetic.
 *
 * Driven by @ref Config, so a host chooses the whole surface behaviour through
 * @ref HostProfile rather than by editing a world: per-pixel or flat, physically
 * based or Lambert, shadows and their per-tick budget, the water reflection probe,
 * the sky's evaluation block.
 *
 * What is deliberately NOT here: the terrain's geometry (render::HeightfieldPatch),
 * where props are scattered (procgen), and what a biome looks like (the game). This
 * owns the surface's LOOK, not the world's content.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_TERRAIN_SURFACE_HPP
#    define LPL_ENGINE_TERRAIN_SURFACE_HPP

#    include <lpl/engine/Config.hpp>
#    include <lpl/render/HeightfieldPatch.hpp>
#    include <lpl/render/Irradiance.hpp>
#    include <lpl/render/Lighting.hpp>
#    include <lpl/render/MipTexture.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/render/Pbr.hpp>
#    include <lpl/render/Reflection.hpp>
#    include <lpl/render/Sky.hpp>
#    include <lpl/render/SkyDome.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>
#    include <lpl/render/Water.hpp>

#    include <lpl/procgen/Grid.hpp>

namespace lpl::engine {

/**
 * @struct TerrainSurfaceParams
 * @brief The few numbers that are a WORLD's, not a host's.
 *
 * Sea level and the fog density describe the place; whether the fog is computed
 * per pixel describes the machine. The split matters: a .lplscene can carry these
 * because they are content, and it must not carry the others.
 */
struct TerrainSurfaceParams {
    core::f32 seaLevel{-1.0f};
    core::f32 fogDensity{0.010f}; ///< Reciprocal is roughly the distance haze wins.
    core::f32 ambient{0.28f};     ///< What a surface facing away from the sun receives.
    core::f32 grainTiles{0.25f};  ///< Texture tiles per world cell.
    core::u32 shadowSteps{24u};   ///< Cells a shadow ray marches; also its longest shadow.
};

/**
 * @class TerrainSurface
 * @brief Sky, sun, grain, shadow masks, water and the reflection probe.
 */
class TerrainSurface {
public:
    /** @brief Reads the host's presentation choices and builds the grain textures. */
    void configure(const Config &config, const TerrainSurfaceParams &params, core::u32 seed);

    /**
     * @brief Adopts the look a cartridge asked for: sky, water, time of day.
     *
     * Separate from @ref configure because the two answer different questions and
     * arrive from different places. configure() reads the HOST (what this machine
     * can afford); this reads the DOCUMENT (what the world looks like). Folding
     * them into one call would have made the caller choose which of the two owns
     * the sea level, and the answer is that they own different fields of it.
     */
    void applyLook(const render::SkyParams &sky, const render::WaterParams &water, core::f32 dayFraction) noexcept
    {
        _skyParams = sky;
        // The phase is where the ripples HAPPEN to be — state, not content. Keeping
        // ours means a cartridge load does not jerk the swell back to zero.
        const core::f32 phase = _water.phase;
        _water = water;
        _water.phase = phase;
        _dayFraction = dayFraction;
        _sun = render::sunAt(_dayFraction);
        // The integrated sky is now stale by construction; force one re-projection.
        _irradianceElevation = -1.0e9f;
    }

    /** @brief Allocates the reflection probe's target once, out of the render path. */
    void attachProbe(core::u32 *colour, core::f32 *depth, core::u32 width, core::u32 height);

    /**
     * @brief Advances the time of day and the water's ripples, in SECONDS.
     *
     * Both arguments are per-second rates multiplied by the caller's dt, not per-tick
     * constants — and the difference is the whole point. A per-tick constant makes the
     * speed of the sea a function of how many fixed steps the loop happens to run, and
     * this loop runs about seventeen of them per rendered frame: the swell came out
     * seventeen times too fast, the same defect the creatures had when they teleported
     * across the map. There is no default for the ripple any more, because a default is
     * exactly how a caller comes to forget that this quantity has a unit.
     *
     * @param dayFraction How far the day advanced, as a fraction of one.
     * @param ripplePhase How far the wave phase advanced.
     */
    void advance(core::f32 dayFraction, core::f32 ripplePhase) noexcept;

    [[nodiscard]] const render::SunState &sun() const noexcept { return _sun; }
    [[nodiscard]] const render::SkyParams &skyParams() const noexcept { return _skyParams; }
    [[nodiscard]] render::WaterParams &water() noexcept { return _water; }
    [[nodiscard]] bool perPixel() const noexcept { return _perPixel; }
    [[nodiscard]] bool physicallyBased() const noexcept { return _physicallyBased; }
    [[nodiscard]] bool shadowsEnabled() const noexcept { return _shadows; }
    [[nodiscard]] core::u32 shadowChunksPerTick() const noexcept { return _shadowChunksPerTick; }
    [[nodiscard]] core::u32 haze() const noexcept { return _haze; }
    /** @brief Whether the last frame began underground. */
    [[nodiscard]] bool underground() const noexcept { return _underground; }
    /**
     * @brief How fast distance fades, this frame.
     *
     * One accessor rather than two call sites choosing: the four places that apply
     * aerial perspective all want the same answer, and one of them reading the surface
     * density underground is a lit corridor stretching into a black cave.
     */
    [[nodiscard]] core::f32 fogDensity() const noexcept { return _underground ? _caveFog : _params.fogDensity; }
    [[nodiscard]] core::u32 waterTessellation() const noexcept { return _waterTessellation; }

    /**
     * @brief Whether the reflection probe holds a picture the shader may sample.
     *
     * Exposed so a test can assert that the pass RAN or did not. Its cost is a whole
     * second render of the world, and a saving that nothing can observe is a saving
     * nobody can keep: the day someone drops the water-in-sight test, this is what
     * notices.
     */
    [[nodiscard]] bool probeValid() const noexcept { return _probe.valid; }
    [[nodiscard]] const TerrainSurfaceParams &params() const noexcept { return _params; }
    [[nodiscard]] const render::IrradianceProbe &irradiance() const noexcept { return _irradiance; }

    /** @brief Paints the sky and remembers the haze the terrain fades into. */
    void beginFrame(const render::RenderTarget &rt, const render::CameraBasis &basis) noexcept;

    /**
     * @brief Paints the dark a cave has instead of a sky, and fades everything into it.
     *
     * Not a colour swap: the haze is what @ref shadeSurface fades distance into, so
     * setting it to the same near-black the frame was cleared with turns the existing
     * aerial-perspective term into the only lighting a cave needs. A separate cave
     * lighting model would have been a second answer to how far you can see, and the
     * two would have disagreed at the mouth.
     *
     * The sun keeps moving and the day keeps advancing while this runs. That is
     * deliberate: the surface is still out there, and a walker who spends ten minutes
     * underground should come out to a different hour.
     *
     * @param rt      Target to clear.
     * @param tint    The dark; also what distance fades into.
     * @param density Reciprocal is roughly how far a lamp reaches.
     */
    void beginCaveFrame(const render::RenderTarget &rt, core::u32 tint, core::f32 density) noexcept;

    /**
     * @struct LampState
     * @brief The light a walker carries, as the surface shading needs it.
     *
     * Set once a frame rather than passed per pixel: a lamp does not move between two
     * pixels of the same frame, and threading nine more arguments through a shader
     * called a hundred thousand times a frame to say so would be nine arguments.
     *
     * ⚠ The cone here is HORIZONTAL — about the view's heading in the ground plane, with
     * no vertical term. The shading callback is given a world x and z and no y (see
     * @ref shadeSurface), so a true 3D cone would mean threading a height through four
     * call sites of a shared signature. The approximation is also the friendlier one for
     * a surface: the pool of light stays on the ground when you look up, where a real
     * torch would swing it into the sky. The cave path, which has full positions, uses a
     * proper 3D cone — see engine::TerrainRenderer::drawWarrens.
     */
    struct LampState {
        bool on{false};
        core::f32 x{0.0f}; ///< Where the lamp is, world.
        core::f32 z{0.0f};
        core::f32 headingX{0.0f}; ///< The beam's axis, normalised in the ground plane.
        core::f32 headingZ{-1.0f};
        core::f32 coneInner{0.70f};
        core::f32 coneOuter{0.10f};
        core::f32 reach{20.0f};
        core::u32 tint{0x00C8B79Eu}; ///< What ground under the beam looks like.
    };

    /**
     * @brief Hands the surface the lamp for this frame.
     *
     * Whether it is ON is the caller's call and not a threshold invented here: darkness
     * is either being under rock or being at night, and render::SunState::intensity is
     * already zero at night — a second definition of "dark" would be a second thing to
     * keep in step with the sky.
     *
     * @param lamp The lamp, or one with @c on false to light nothing.
     */
    void setLamp(const LampState &lamp) noexcept { _lamp = lamp; }

    /**
     * @brief Surface colour for one point: grain, light, haze.
     *
     * The grain is sampled at WORLD coordinates rather than per-vertex UVs. A cell
     * has no natural UV in a streamed world — there is no corner to measure from —
     * and world coordinates make two patches agree along their border for free.
     */
    [[nodiscard]] core::u32 shade(core::f32 worldX, core::f32 worldZ, core::u32 base, core::f32 lit, core::f32 distance,
                                  bool rocky) const noexcept;

    /**
     * @brief Physically based colour for one point: GGX, sky irradiance, ACES.
     *
     * What it buys on terrain is not a highlight — grass has none — but the
     * RESPONSE: a slope facing away from the sun keeps the sky's blue instead of
     * going grey. The environment term is a constant, which is the honest version
     * of image-based lighting with no image: the sky evaluated straight up. A real
     * irradiance map integrates the dome, and that is the next step, not this one.
     */
    [[nodiscard]] core::u32 shadePhysical(core::f32 worldX, core::f32 worldZ, core::u32 base, core::f32 nx,
                                          core::f32 nz, core::f32 occlusion, core::f32 distance, bool rocky,
                                          const render::CameraBasis &basis) const noexcept;

    /** @brief The shader to hand @c render::drawHeightfieldPatch, whichever path is on. */
    [[nodiscard]] core::u32 shadeSurface(core::f32 worldX, core::f32 worldZ, core::u32 base, core::f32 lit,
                                         core::f32 nx, core::f32 nz, core::f32 occlusion,
                                         const render::CameraBasis &basis) const noexcept;

    /**
     * @brief Draws a water surface: reflection, Fresnel, glint, and the probe.
     *
     * @param latticePitch World-space spacing of the displacement grid. See
     *                      @ref drawWaterWith for why it may not be derived from the quad.
     * @param bedDepthAt (worldX, worldZ) -> how far the bed is below the surface.
     *                   Zero at the shoreline, which is what fades a beach instead
     *                   of ending it on a hard line.
     */
    template <typename BedDepthAt>
    core::u32 drawWater(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                        const render::CameraBasis &basis, const core::f32 *quad, core::f32 latticePitch,
                        BedDepthAt &&bedDepthAt) const
    {
        return drawWaterWith(rt, mvp, basis, quad, _water, latticePitch, bedDepthAt);
    }

    /**
     * @brief The same water, shaded with a surface the CALLER chose.
     *
     * The open sea takes @ref water; a river takes its own, because its swell runs with
     * its current rather than with the wind and its body is shallower. Two entry points
     * over one implementation rather than two implementations: a river drawn by a copy
     * of this arithmetic would be a second place for Fresnel to be got wrong.
     *
     * @param water     The surface to shade with.
     * @param latticePitch World-space spacing of the displacement grid, in cells.
     * @param bedDepthAt (worldX, worldZ) -> how far the bed is below the surface.
     */
    template <typename BedDepthAt>
    core::u32 drawWaterWith(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                            const render::CameraBasis &basis, const core::f32 *quad, const render::WaterParams &water,
                            core::f32 latticePitch, BedDepthAt &&bedDepthAt) const
    {
        const auto shade = [&](core::f32 wx, core::f32 wy, core::f32 wz) {
            core::u32 colour = render::waterColour(wx, wy, wz, basis.eye, _sun, _skyParams, water, bedDepthAt(wx, wz));
            // The probe carries what the sky cannot: the terrain standing over
            // the water. Where it has nothing to say — off its edge, behind its
            // near plane — the sky mirror already answered.
            core::u32 reflected = 0u;
            if (_reflection && render::sampleProbe(_probe, _probePixels, wx, wy, wz, reflected))
                colour = render::mixColours(colour, reflected, 0.45f);
            return render::applyAerialPerspective(
                colour, _haze, render::approximateLength(wx - basis.eye.x, wz - basis.eye.z), _params.fogDensity);
        };

        // A flat quad is a mirror that ripples: its silhouette against the horizon is a
        // straight line however good the shading is. The swell becomes GEOMETRY only when
        // there are vertices to move, which is what the divisions buy.
        //
        // Two conditions, and both are necessary: a world with no swell has nothing to
        // displace, and a host that declined the cost gets the quad. Either alone would
        // spend triangles on a flat surface.
        const core::u32 divisions = (water.swellHeight > 0.0f && _waterTessellation > 1u) ? _waterTessellation : 1u;
        if (divisions == 1u)
            return render::fillPolygonShadedClipped(rt, mvp, quad, 4u, shade);

        // The grid is built from the quad's own corners, which the callers lay out as
        // (x0,z0) (x1,z0) (x1,z1) (x0,z1) at one height. A quad that is not that — one
        // corner lifted, a river cell on a slope — falls back rather than being
        // subdivided wrongly: a tessellator that assumed the layout would silently
        // flatten anything else.
        const core::f32 y = quad[1];
        if (quad[4] != y || quad[7] != y || quad[10] != y)
            return render::fillPolygonShadedClipped(rt, mvp, quad, 4u, shade);

        const core::f32 x0 = quad[0];
        const core::f32 z0 = quad[2];
        const core::f32 x1 = quad[6];
        const core::f32 z1 = quad[8];
        if (latticePitch <= 0.0f || x1 <= x0 || z1 <= z0)
            return render::fillPolygonShadedClipped(rt, mvp, quad, 4u, shade);

        // Displacement is damped to nothing as the bed rises to meet the surface. Two
        // reasons, and the second is the one that would have shown: a crest lifted over a
        // beach pokes through the sand it is supposed to lap against, and the waterline
        // then crawls up and down the shore every frame.
        const core::f32 fade = 1.0f / (2.0f * water.swellHeight);
        const auto surfaceY = [&](core::f32 wx, core::f32 wz) {
            const core::f32 depth = bedDepthAt(wx, wz);
            core::f32 damp = depth * fade;
            damp = damp < 0.0f ? 0.0f : (damp > 1.0f ? 1.0f : damp);
            return y + render::waterHeight(wx, wz, water) * damp;
        };

        // ⚠ The grid lines are snapped to an ABSOLUTE WORLD LATTICE, not divided out of this
        // quad's own extent — and that is the difference between a watertight sheet and a
        // cracked one. Quads differ in size, because each is tightened to the cells its own
        // chunk has under water: dividing each by the same count puts their vertices at
        // different world positions, so two neighbours interpolate the same wave function
        // along two different piecewise-linear paths and the surface splits along every
        // chunk border. It is the same discipline the terrain follows for the same reason —
        // sample at absolute coordinates and neighbours agree for free.
        //
        // Both quads also put a vertex exactly ON their shared border, because the walk is
        // clamped to the quad, so the lattice never has to line up with the border itself.
        const auto latticeFloor = [latticePitch](core::f32 v) {
            const core::f32 scaled = v / latticePitch;
            const core::f32 truncated = static_cast<core::f32>(static_cast<core::i32>(scaled));
            return (scaled < truncated ? truncated - 1.0f : truncated) * latticePitch;
        };

        core::u32 triangles = 0u;
        for (core::f32 az = latticeFloor(z0); az < z1; az += latticePitch)
        {
            const core::f32 lowZ = az < z0 ? z0 : az;
            const core::f32 highZ = az + latticePitch > z1 ? z1 : az + latticePitch;
            if (highZ <= lowZ)
                continue;
            for (core::f32 ax = latticeFloor(x0); ax < x1; ax += latticePitch)
            {
                const core::f32 lowX = ax < x0 ? x0 : ax;
                const core::f32 highX = ax + latticePitch > x1 ? x1 : ax + latticePitch;
                if (highX <= lowX)
                    continue;
                // Corners are evaluated from WORLD coordinates, so the cell shares its
                // edge values with its neighbours — inside this quad and across the seam
                // into the next chunk's — and the sheet cannot tear.
                const core::f32 cell[12] = {lowX,  surfaceY(lowX, lowZ),   lowZ,  highX, surfaceY(highX, lowZ), lowZ,
                                            highX, surfaceY(highX, highZ), highZ, lowX,  surfaceY(lowX, highZ), highZ};
                triangles += render::fillPolygonShadedClipped(rt, mvp, cell, 4u, shade);
            }
        }
        return triangles;
    }

    /**
     * @brief Refreshes the reflection probe by rendering the world mirrored.
     *
     * @param drawWorld (target, mirrorMvp, mirroredBasis) -> void. The caller draws
     *                  whatever belongs in a reflection, which is its business: at
     *                  probe resolution a tree is two pixels, and the pass exists to
     *                  put the MOUNTAIN in the water.
     *
     * Amortised by @p frame: a reflection one frame stale is invisible on water that
     * is rippling anyway, and the pass is a whole extra render of the world.
     *
     * @param waterInSight Whether anything on screen will sample the probe. Required
     *                     rather than defaulted: this pass is the single most expensive
     *                     thing the surface layer does — a whole second render of the
     *                     world — and it was running over dry land for every frame of
     *                     every world without water in it, because nothing ever asked.
     *                     A default of true would keep that, silently, for any caller
     *                     that did not know to pass it.
     */
    template <typename DrawWorld>
    void refreshProbe(core::u32 frame, const render::CameraBasis &basis, bool waterInSight, DrawWorld &&drawWorld)
    {
        if (!_reflection || _probeColour == nullptr || _probeDepth == nullptr || !waterInSight)
        {
            _probe.valid = false;
            return;
        }
        if ((frame & 3u) != 0u && _probe.valid)
            return;

        const render::CameraBasis mirrored = render::mirrorBasisAboutPlane(basis, _params.seaLevel);
        const core::f32 aspect = static_cast<core::f32>(_probe.width) / static_cast<core::f32>(_probe.height);
        _probe.planeY = _params.seaLevel;
        _probe.mirrorMvp =
            render::mirrorViewProjection(mirrored, aspect, render::CameraLens{1.04719755f, 0.4f, 600.0f});

        const render::RenderTarget target{_probeColour, _probeDepth, _probe.width, _probe.height};
        render::drawSky(target, mirrored, _sun, _skyParams, 0.5773502692f, 3u);
        drawWorld(target, _probe.mirrorMvp, mirrored);

        // Into a texture allocated once: allocating here measurably stalled a
        // world's streaming, because a probe pass runs every few frames.
        _probe.valid = render::copyTargetToTexture(target, _probePixels);
    }

    /**
     * @brief Fills a patch's shadow mask: the terrain over itself, then its props.
     *
     * @param heightAt   (worldX, worldZ) -> f32, reading the WORLD and not the patch:
     *                   a ridge one patch over still casts here, and a shadow that
     *                   stopped at a border would be the most visible seam in the
     *                   scene.
     * @param forEachProp (emit) with emit(cellX, cellZ, height, spread) for every
     *                    prop that should cast. A prop is a vertical stick of known
     *                    height, so where its shadow lands is arithmetic rather than
     *                    a ray: walk the ground away from the sun, covering
     *                    height / sunElevation cells.
     * @param mask       Written in place; 0 lit, 255 fully shadowed.
     */
    template <typename HeightAt, typename ForEachProp>
    void fillShadowMask(const render::HeightfieldPatchParams &patch, HeightAt &&heightAt, ForEachProp &&forEachProp,
                        procgen::Grid<core::u8> &mask) const
    {
        if (!_shadows || mask.empty())
            return;

        for (core::u32 z = 0u; z < patch.size; ++z)
            for (core::u32 x = 0u; x < patch.size; ++x)
            {
                const core::i32 worldX = patch.originX + static_cast<core::i32>(x);
                const core::i32 worldZ = patch.originZ + static_cast<core::i32>(z);
                const core::f32 occlusion =
                    render::terrainShadow([&heightAt](core::i32 sx, core::i32 sz) { return heightAt(sx, sz); }, worldX,
                                          worldZ, heightAt(worldX, worldZ), _sun, _params.shadowSteps);
                mask.at(x, z) = static_cast<core::u8>(occlusion * 255.0f);
            }

        if (_sun.y <= 0.12f)
            return; // sun at or below the horizon: everything is shadowed anyway

        const core::f32 reach = 1.0f / _sun.y;
        forEachProp([&](core::i32 cellX, core::i32 cellZ, core::f32 height, core::f32 spread) {
            const core::u32 steps = static_cast<core::u32>(height * reach);
            for (core::u32 step = 0u; step <= steps && step < _params.shadowSteps; ++step)
            {
                const core::f32 t = static_cast<core::f32>(step);
                const core::i32 shadowX = cellX - static_cast<core::i32>(_sun.x * t * reach);
                const core::i32 shadowZ = cellZ - static_cast<core::i32>(_sun.z * t * reach);
                const core::i32 localX = shadowX - patch.originX;
                const core::i32 localZ = shadowZ - patch.originZ;
                if (localX < 0 || localZ < 0 || localX >= static_cast<core::i32>(patch.size) ||
                    localZ >= static_cast<core::i32>(patch.size))
                    continue;

                // Fades along its length: a crown thins towards the top, so the far
                // end of its shadow is dappled rather than a hard bar.
                const core::f32 along = steps == 0u ? 0.0f : t / static_cast<core::f32>(steps);
                const core::f32 strength = (0.85f - 0.45f * along) * (spread > 0.6f ? 1.0f : 0.7f);
                core::u8 &cell = mask.at(static_cast<core::u32>(localX), static_cast<core::u32>(localZ));
                const core::u32 added = static_cast<core::u32>(cell) + static_cast<core::u32>(strength * 255.0f);
                cell = static_cast<core::u8>(added > 255u ? 255u : added);
            }
        });
    }

private:
    TerrainSurfaceParams _params{};
    render::SunState _sun{render::sunAt(0.32f)};
    render::SkyParams _skyParams{};
    /**
     * @brief The sky, integrated: refreshed when the sun has actually moved.
     *
     * Three hundred and eighty-four sky evaluations is nothing next to a frame, but
     * it is not nothing next to zero, and the sky only changes when the sun does.
     * The threshold is what makes this a once-in-a-while cost instead of a per-frame
     * one, without a flag for the caller to forget to set.
     */
    render::IrradianceProbe _irradiance{};
    core::f32 _irradianceElevation{-1.0e9f};
    render::WaterParams _water{};
    render::MipTexture _grassGrain{};
    render::MipTexture _rockGrain{};
    render::ReflectionProbe _probe{};
    render::Texture _probePixels{};
    core::u32 *_probeColour{nullptr};
    core::f32 *_probeDepth{nullptr};
    core::f32 _dayFraction{0.32f};
    core::u32 _haze{0u};
    core::f32 _caveFog{0.10f};
    LampState _lamp{};

    /** @brief Brightens a shaded colour by the carried lamp. */
    [[nodiscard]] core::u32 applyLamp(core::u32 colour, core::f32 worldX, core::f32 worldZ) const noexcept;
    bool _underground{false};
    core::u32 _skyBlock{1u};
    core::u32 _waterTessellation{0u};
    core::u32 _shadowChunksPerTick{1u};
    bool _perPixel{true};
    bool _physicallyBased{false};
    bool _shadows{true};
    bool _reflection{true};
};

} // namespace lpl::engine

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/engine/TerrainSurface.inl>

#endif // LPL_ENGINE_TERRAIN_SURFACE_HPP
