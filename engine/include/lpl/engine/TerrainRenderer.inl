/**
 * @file TerrainRenderer.inl
 * @brief Out-of-line definitions for engine::TerrainRenderer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_TERRAIN_RENDERER_INL
#    define LPL_ENGINE_TERRAIN_RENDERER_INL

namespace lpl::engine {

template <typename Palette, typename GroundAt>
core::u32 TerrainRenderer::drawStreamed(const render::RenderTarget &rt, const render::OrbitCamera &camera,
                                        TerrainStreamer &streamer, TerrainSurface &surface, const PropLibrary &props,
                                        const ecology::Herd &herd, const TerrainDrawParams &params, core::u32 frame,
                                        Palette &&palette, GroundAt &&groundAt)
{
    _triangles = 0u;
    if (streamer.empty())
        return 0u;

    // Never below the water: the terrain goes on under a lake, and an eye that
    // followed it there looks up at the underside of the ground sheet.
    const core::f32 groundAtFocus = render::OrbitCamera::clamp(
        groundAt(static_cast<core::i32>(camera.focusX()), static_cast<core::i32>(camera.focusZ())), params.seaLevel,
        1.0e6f);
    render::CameraBasis basis{};
    const auto mvp = camera.viewProjection(groundAtFocus,
                                           static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height),
                                           render::CameraLens{params.fovRadians, params.nearPlane, params.farPlane},
                                           basis);

    surface.beginFrame(rt, basis);
    const core::u32 haze = surface.haze();

    const core::i32 span = static_cast<core::i32>(params.chunkSize);
    const core::i32 focusChunkX = procgen::floorDivChunk(static_cast<core::i32>(camera.focusX()), span);
    const core::i32 focusChunkZ = procgen::floorDivChunk(static_cast<core::i32>(camera.focusZ()), span);

    render::ChunkedViewParams viewParams;
    viewParams.chunkSize = params.chunkSize;
    viewParams.lodRings = params.lodRings;
    viewParams.centreY = params.chunkCentreY;
    viewParams.halfHeight = params.chunkHalfHeight;
    viewParams.ambient = params.ambient;
    viewParams.skirtDrop = params.skirtDrop;

    props.beginFrame();
    _view.select(mvp, basis.eye, rt.width, rt.height, viewParams, focusChunkX, focusChunkZ, streamer.size(),
                 [&streamer](core::u32 index, core::i32 &outX, core::i32 &outZ) {
                     outX = streamer.at(index).coord.x;
                     outZ = streamer.at(index).coord.z;
                 });

    // ONCE per frame, after the visible set is known and before the ground is drawn.
    // It was called twice for a while, which rendered the mirrored world for nothing
    // every fourth frame.
    refreshProbe(streamer, surface, basis, frame, params, palette);

    _view.draw(rt, mvp, viewParams, surface.sun(),
               [&](const render::RenderTarget &target, const math::Mat4<core::f32> &matrix,
                   const render::SunState &sun, const render::VisibleChunkRef &ref,
                   render::HeightfieldPatchParams patch, const render::ChunkedViewParams &view) -> core::u32 {
                   const TerrainChunk &chunk = streamer.at(ref.index);
                   if (chunk.height.empty())
                       return 0u;

                   const core::i32 originX = chunk.coord.x * static_cast<core::i32>(view.chunkSize);
                   const core::i32 originZ = chunk.coord.z * static_cast<core::i32>(view.chunkSize);
                   patch.originX = originX;
                   patch.originZ = originZ;

                   const auto heightAt = [&chunk](core::u32 x, core::u32 z) {
                       return chunk.height.at(x, z).toFloat();
                   };
                   const auto shadeAt = [&chunk](core::u32 x, core::u32 z) {
                       return chunk.shade.empty() ? 0.0f
                                                  : static_cast<core::f32>(chunk.shade.at(x, z)) * (1.0f / 255.0f);
                   };

                   core::u32 triangles = render::drawHeightfieldPatch(
                       target, matrix, patch, sun, heightAt, shadeAt,
                       [&chunk, &palette](core::u32 x, core::u32 z) { return palette(chunk.biomes.at(x, z)); },
                       [&](core::f32 wx, core::f32 wz, core::u32 base, core::f32 lit, core::f32 nx, core::f32 nz,
                           core::f32 occlusion) {
                           return surface.shadeSurface(wx, wz, base, lit, nx, nz, occlusion, basis);
                       },
                       surface.perPixel());

                   triangles += render::drawPatchSkirts(target, matrix, patch, view.skirtDrop, heightAt,
                                                        [&chunk, &palette](core::u32 cx, core::u32 cz) {
                                                            return render::modulate(palette(chunk.biomes.at(cx, cz)),
                                                                                    0.45f);
                                                        });

                   // The sea, per chunk: one flat quad rather than a world-sized plane,
                   // because an endless world has no width to span — and only where
                   // there IS water, or a chunk whose bed never drops below sea level
                   // pays for a per-pixel pass to draw nothing.
                   if (chunk.lowest < params.seaLevel)
                   {
                       const core::f32 sx0 = static_cast<core::f32>(originX);
                       const core::f32 sx1 = sx0 + static_cast<core::f32>(view.chunkSize);
                       const core::f32 sz0 = static_cast<core::f32>(originZ);
                       const core::f32 sz1 = sz0 + static_cast<core::f32>(view.chunkSize);
                       const core::f32 sea[12] = {sx0, params.seaLevel, sz0, sx1, params.seaLevel, sz0,
                                                  sx1, params.seaLevel, sz1, sx0, params.seaLevel, sz1};
                       triangles += surface.drawWater(target, matrix, basis, sea, [&](core::f32 wx, core::f32 wz) {
                           return params.seaLevel - groundAt(static_cast<core::i32>(wx), static_cast<core::i32>(wz));
                       });
                   }

                   // Props AFTER the ground of their own chunk: a boulder then tests
                   // against a depth buffer that already holds the hill it stands on.
                   if (ref.ring <= 1)
                       triangles += props.drawRocks(
                           target, matrix, basis, originX, originZ, view.chunkSize, params.seaLevel,
                           math::Vec3<core::f32>(sun.x, sun.y, sun.z), params.ambient, heightAt);

                   if (ref.ring <= 2)
                       for (core::u32 i = 0u; i < chunk.plants.size(); ++i)
                       {
                           if (!chunk.plants[i].standing)
                               continue; // grazed: the herd ate it, and the map says so
                           const core::u32 lx = static_cast<core::u32>(chunk.plants[i].cellX - originX);
                           const core::u32 lz = static_cast<core::u32>(chunk.plants[i].cellZ - originZ);
                           props.queuePlant(
                               chunk.plants[i].cellX, chunk.plants[i].cellZ, chunk.height.at(lx, lz).toFloat(),
                               chunk.shade.empty()
                                   ? 1.0f
                                   : 1.0f - 0.55f * (static_cast<core::f32>(chunk.shade.at(lx, lz)) / 255.0f));
                       }
                   return triangles;
               });

    _triangles = _view.stats().triangles;
    _triangles += props.flushPlants(rt, mvp, basis, haze);
    _triangles += drawHerd(rt, mvp, herd, params, groundAt);
    return _triangles;
}

template <typename Palette, typename HeightAt, typename ColourAt, typename GroundAt>
core::u32 TerrainRenderer::drawBounded(const render::RenderTarget &rt, const render::OrbitCamera &camera,
                                       TerrainSurface &surface, const PropLibrary &props, const ecology::Herd &herd,
                                       core::u32 gridWidth, core::u32 gridDepth, const ecology::PlantCell *plants,
                                       core::u32 plantCount, const TerrainDrawParams &params, Palette &&palette,
                                       HeightAt &&heightAt, ColourAt &&colourAt, GroundAt &&groundAt)
{
    (void) palette;
    _triangles = 0u;

    render::CameraBasis basis{};
    const auto mvp = camera.viewProjection(0.0f, static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height),
                                           render::CameraLens{params.fovRadians, params.nearPlane, 400.0f}, basis);

    surface.beginFrame(rt, basis);
    const core::u32 haze = surface.haze();

    const core::f32 halfX = static_cast<core::f32>(gridWidth) * 0.5f;
    const core::f32 halfZ = static_cast<core::f32>(gridDepth) * 0.5f;

    // The bounded grid is ONE patch, centred on the origin: the same module call the
    // streamed world makes per chunk, which is the point of having it.
    render::HeightfieldPatchParams patch;
    patch.size = gridWidth < gridDepth ? gridWidth : gridDepth;
    patch.stride = 1u;
    patch.originX = -static_cast<core::i32>(halfX);
    patch.originZ = -static_cast<core::i32>(halfZ);
    patch.ambient = params.ambient;

    _triangles += render::drawHeightfieldPatch(
        rt, mvp, patch, surface.sun(), heightAt, [](core::u32, core::u32) { return 0.0f; }, colourAt,
        [&](core::f32 wx, core::f32 wz, core::u32 base, core::f32 lit, core::f32 nx, core::f32 nz,
            core::f32 occlusion) { return surface.shadeSurface(wx, wz, base, lit, nx, nz, occlusion, basis); },
        surface.perPixel());

    const core::f32 sea[12] = {-halfX, params.seaLevel, -halfZ, halfX,  params.seaLevel, -halfZ,
                               halfX,  params.seaLevel, halfZ,  -halfX, params.seaLevel, halfZ};
    _triangles += surface.drawWater(rt, mvp, basis, sea, [&](core::f32 wx, core::f32 wz) {
        return params.seaLevel - groundAt(static_cast<core::i32>(wx), static_cast<core::i32>(wz));
    });

    props.beginFrame();
    for (core::u32 i = 0u; i < plantCount; ++i)
    {
        if (!plants[i].standing)
            continue;
        const core::u32 cellX = static_cast<core::u32>(plants[i].cellX);
        const core::u32 cellZ = static_cast<core::u32>(plants[i].cellZ);
        props.queuePlant(plants[i].cellX - static_cast<core::i32>(halfX),
                         plants[i].cellZ - static_cast<core::i32>(halfZ), heightAt(cellX, cellZ), 1.0f);
    }
    _triangles += props.flushPlants(rt, mvp, basis, haze);
    _triangles += drawHerd(rt, mvp, herd, params, groundAt);
    return _triangles;
}

template <typename GroundAt>
core::u32 TerrainRenderer::drawHerd(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                    const ecology::Herd &herd, const TerrainDrawParams &params,
                                    GroundAt &&groundAt) const
{
    core::u32 triangles = 0u;
    for (core::u32 i = 0u; i < herd.size(); ++i)
    {
        const ecology::HerdMember &member = herd.at(i);
        const core::f32 wx = member.body.x.toFloat();
        const core::f32 wz = member.body.z.toFloat();
        // The ground that is DRAWN, not the noise: the two differ by exactly what
        // erosion moved, and the herd hung in the air above the ridges it lowered.
        const core::f32 ground = groundAt(member.body.x.toInt(), member.body.z.toInt());
        const core::f32 size = params.bodyScale * member.genome.size.toFloat();

        const ai::PersonalityTraits traits = ai::personalityOf(member.id, member.species);
        core::u32 tint = member.species == 1u ? params.hunterTint : params.grazerTint;
        if (traits.aggression > math::Fixed32::fromFloat(0.75f))
            tint |= 0x00200000u;

        // A billboard rather than a box: at this scale a creature is a handful of
        // pixels, and six lit faces cost six times as much to say the same.
        const core::f32 body[12] = {wx - size, ground,               wz, wx + size, ground,               wz,
                                    wx + size, ground + size * 2.4f, wz, wx - size, ground + size * 2.4f, wz};
        triangles += render::fillPolygonClipped(rt, mvp, body, 4u, tint);
    }
    return triangles;
}

template <typename Palette>
void TerrainRenderer::refreshProbe(TerrainStreamer &streamer, TerrainSurface &surface,
                                   const render::CameraBasis &basis, core::u32 frame,
                                   const TerrainDrawParams &params, Palette &&palette)
{
    surface.refreshProbe(frame, basis,
                         [&](const render::RenderTarget &target, const math::Mat4<core::f32> &mirrorMvp,
                             const render::CameraBasis &) {
                             // Terrain only, flat-shaded, at a coarse stride: at probe
                             // resolution a tree is two pixels, and the pass exists to
                             // put the MOUNTAIN in the water.
                             for (core::u32 v = 0u; v < _view.visible().size(); ++v)
                             {
                                 const TerrainChunk &chunk = streamer.at(_view.visible()[v].index);
                                 if (chunk.height.empty())
                                     continue;

                                 render::HeightfieldPatchParams patch;
                                 patch.size = params.chunkSize;
                                 patch.stride = 4u;
                                 patch.originX = chunk.coord.x * static_cast<core::i32>(params.chunkSize);
                                 patch.originZ = chunk.coord.z * static_cast<core::i32>(params.chunkSize);
                                 patch.ambient = params.ambient;

                                 render::drawHeightfieldPatch(
                                     target, mirrorMvp, patch, surface.sun(),
                                     [&chunk](core::u32 x, core::u32 z) { return chunk.height.at(x, z).toFloat(); },
                                     [](core::u32, core::u32) { return 0.0f; },
                                     [&chunk, &palette](core::u32 x, core::u32 z) {
                                         return palette(chunk.biomes.at(x, z));
                                     },
                                     [](core::f32, core::f32, core::u32 base, core::f32 lit, core::f32, core::f32,
                                        core::f32) { return render::modulate(base, lit); },
                                     false);
                             }
                         });
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_RENDERER_INL
