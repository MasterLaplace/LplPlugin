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

    /** @brief Advances the time of day and the water's ripples. */
    void advance(core::f32 dayStep, core::f32 rippleStep = 0.035f) noexcept;

    [[nodiscard]] const render::SunState &sun() const noexcept { return _sun; }
    [[nodiscard]] const render::SkyParams &skyParams() const noexcept { return _skyParams; }
    [[nodiscard]] render::WaterParams &water() noexcept { return _water; }
    [[nodiscard]] bool perPixel() const noexcept { return _perPixel; }
    [[nodiscard]] bool physicallyBased() const noexcept { return _physicallyBased; }
    [[nodiscard]] bool shadowsEnabled() const noexcept { return _shadows; }
    [[nodiscard]] core::u32 shadowChunksPerTick() const noexcept { return _shadowChunksPerTick; }
    [[nodiscard]] core::u32 haze() const noexcept { return _haze; }
    [[nodiscard]] const TerrainSurfaceParams &params() const noexcept { return _params; }
    [[nodiscard]] const render::IrradianceProbe &irradiance() const noexcept { return _irradiance; }

    /** @brief Paints the sky and remembers the haze the terrain fades into. */
    void beginFrame(const render::RenderTarget &rt, const render::CameraBasis &basis) noexcept;

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
     * @param bedDepthAt (worldX, worldZ) -> how far the bed is below the surface.
     *                   Zero at the shoreline, which is what fades a beach instead
     *                   of ending it on a hard line.
     */
    template <typename BedDepthAt>
    core::u32 drawWater(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                        const render::CameraBasis &basis, const core::f32 *quad, BedDepthAt &&bedDepthAt) const
    {
        return render::fillPolygonShadedClipped(rt, mvp, quad, 4u, [&](core::f32 wx, core::f32 wy, core::f32 wz) {
            core::u32 colour = render::waterColour(wx, wy, wz, basis.eye, _sun, _skyParams, _water, bedDepthAt(wx, wz));
            // The probe carries what the sky cannot: the terrain standing over
            // the water. Where it has nothing to say — off its edge, behind its
            // near plane — the sky mirror already answered.
            core::u32 reflected = 0u;
            if (_reflection && render::sampleProbe(_probe, _probePixels, wx, wy, wz, reflected))
                colour = render::mixColours(colour, reflected, 0.45f);
            return render::applyAerialPerspective(
                colour, _haze, render::approximateLength(wx - basis.eye.x, wz - basis.eye.z), _params.fogDensity);
        });
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
     */
    template <typename DrawWorld>
    void refreshProbe(core::u32 frame, const render::CameraBasis &basis, DrawWorld &&drawWorld)
    {
        if (!_reflection || _probeColour == nullptr || _probeDepth == nullptr)
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
    core::u32 _skyBlock{1u};
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
