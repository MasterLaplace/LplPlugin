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

inline void TerrainRenderer::selectChunks(const math::Mat4<core::f32> &mvp, const render::CameraBasis &basis,
                                          core::u32 targetWidth, core::u32 targetHeight,
                                          const render::ChunkedViewParams &view, const TerrainDrawParams &params,
                                          core::i32 focusChunkX, core::i32 focusChunkZ, TerrainStreamer &streamer)
{
    const core::u32 count = streamer.size();
    const core::f32 span = static_cast<core::f32>(view.chunkSize);

    if (!params.useSpatialCull || count < params.spatialCullThreshold)
    {
        _view.select(mvp, basis.eye, targetWidth, targetHeight, view, focusChunkX, focusChunkZ, count,
                     [&streamer](core::u32 index, core::i32 &outX, core::i32 &outZ) {
                         outX = streamer.at(index).coord.x;
                         outZ = streamer.at(index).coord.z;
                     });
        return;
    }

    // The tree's bounds are the resident set's own extent, recomputed each frame:
    // a streamed world slides, and bounds pinned at the origin would spend the whole
    // key range on space that holds nothing.
    core::i32 minChunkX = streamer.at(0u).coord.x;
    core::i32 maxChunkX = minChunkX;
    core::i32 minChunkZ = streamer.at(0u).coord.z;
    core::i32 maxChunkZ = minChunkZ;
    for (core::u32 i = 1u; i < count; ++i)
    {
        const auto &coord = streamer.at(i).coord;
        minChunkX = coord.x < minChunkX ? coord.x : minChunkX;
        maxChunkX = coord.x > maxChunkX ? coord.x : maxChunkX;
        minChunkZ = coord.z < minChunkZ ? coord.z : minChunkZ;
        maxChunkZ = coord.z > maxChunkZ ? coord.z : maxChunkZ;
    }

    const auto worldFixed = [](core::f32 v) { return math::Fixed32::fromFloat(v); };
    const core::f32 lowY = params.chunkCentreY - params.chunkHalfHeight;
    const core::f32 highY = params.chunkCentreY + params.chunkHalfHeight;
    _chunkIndex.clear();
    _chunkIndex.setWorldBounds(math::AABB<math::Fixed32>{
        math::Vec3<math::Fixed32>{worldFixed(static_cast<core::f32>(minChunkX) * span),     worldFixed(lowY),
                                  worldFixed(static_cast<core::f32>(minChunkZ) * span)    },
        math::Vec3<math::Fixed32>{worldFixed(static_cast<core::f32>(maxChunkX + 1) * span), worldFixed(highY),
                                  worldFixed(static_cast<core::f32>(maxChunkZ + 1) * span)}
    });

    for (core::u32 i = 0u; i < count; ++i)
    {
        const auto &coord = streamer.at(i).coord;
        const core::f32 x0 = static_cast<core::f32>(coord.x) * span;
        const core::f32 z0 = static_cast<core::f32>(coord.z) * span;
        _chunkIndex.insert(
            i, math::AABB<math::Fixed32>{
                   math::Vec3<math::Fixed32>{worldFixed(x0),        worldFixed(lowY),  worldFixed(z0)       },
                   math::Vec3<math::Fixed32>{worldFixed(x0 + span), worldFixed(highY), worldFixed(z0 + span)}
        });
    }
    _chunkIndex.rebuild();

    // The node test is the SAME projection the chunks use, applied to a node's box.
    // A view volume is not an AABB, so querying its bounding box would hand back most
    // of the world whenever the camera is pitched; rejecting a node instead prunes a
    // whole subtree, which is the only thing a hierarchy is worth having for.
    core::u32 nodesVisited = 0u;
    core::u32 nodesPruned = 0u;
    _view.beginSelect();
    _chunkIndex.queryVisible(
        [&](const math::AABB<math::Fixed32> &bound) {
            const core::f32 cx = (bound.min.x.toFloat() + bound.max.x.toFloat()) * 0.5f;
            const core::f32 cy = (bound.min.y.toFloat() + bound.max.y.toFloat()) * 0.5f;
            const core::f32 cz = (bound.min.z.toFloat() + bound.max.z.toFloat()) * 0.5f;
            const core::f32 hx = (bound.max.x.toFloat() - bound.min.x.toFloat()) * 0.5f;
            const core::f32 hy = (bound.max.y.toFloat() - bound.min.y.toFloat()) * 0.5f;
            const core::f32 hz = (bound.max.z.toFloat() - bound.min.z.toFloat()) * 0.5f;
            return !render::boxOutsideFrustum(mvp, cx, cy, cz, hx, hy, hz, targetWidth, targetHeight);
        },
        [&](core::u32 index) {
            const auto &coord = streamer.at(index).coord;
            _view.consider(mvp, basis.eye, targetWidth, targetHeight, view, focusChunkX, focusChunkZ, index, coord.x,
                           coord.z);
        },
        &nodesVisited, &nodesPruned);
    _view.endSelect();
    _view.noteHierarchy(nodesVisited, nodesPruned);
}

template <typename Palette, typename GroundAt>
core::u32 TerrainRenderer::drawStreamed(const render::RenderTarget &rt, const render::OrbitCamera &camera,
                                        TerrainStreamer &streamer, TerrainSurface &surface, const PropLibrary &props,
                                        const ecs::Registry &registry, const TerrainDrawParams &params, core::u32 frame,
                                        Palette &&palette, GroundAt &&groundAt)
{
    _triangles = 0u;
    if (streamer.empty())
        return 0u;

    // Never below the water: the terrain goes on under a lake, and an eye that
    // followed it there looks up at the underside of the ground sheet.
    const core::f32 groundAtFocus = params.useFocusHeight ?
                                        params.focusHeight :
                                        render::OrbitCamera::clamp(groundAt(static_cast<core::i32>(camera.focusX()),
                                                                            static_cast<core::i32>(camera.focusZ())),
                                                                   params.seaLevel, 1.0e6f);
    render::CameraBasis basis{};
    const auto mvp =
        camera.viewProjection(groundAtFocus, static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height),
                              render::CameraLens{params.fovRadians, params.nearPlane, params.farPlane}, basis);

    const core::u64 skyBegan = now();
    surface.beginFrame(rt, basis);
    _skyCycles += now() - skyBegan;
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
    selectChunks(mvp, basis, rt.width, rt.height, viewParams, params, focusChunkX, focusChunkZ, streamer);

    // ONCE per frame, after the visible set is known and before the ground is drawn.
    // It was called twice for a while, which rendered the mirrored world for nothing
    // every fourth frame.
    refreshProbe(streamer, surface, basis, frame, params, palette);

    _view.draw(
        rt, mvp, viewParams, surface.sun(),
        [&](const render::RenderTarget &target, const math::Mat4<core::f32> &matrix, const render::SunState &sun,
            const render::VisibleChunkRef &ref, render::HeightfieldPatchParams patch,
            const render::ChunkedViewParams &view) -> core::u32 {
            const TerrainChunk &chunk = streamer.at(ref.index);
            if (chunk.height.empty())
                return 0u;

            const core::i32 originX = chunk.coord.x * static_cast<core::i32>(view.chunkSize);
            const core::i32 originZ = chunk.coord.z * static_cast<core::i32>(view.chunkSize);
            patch.originX = originX;
            patch.originZ = originZ;

            // Index == size is legitimate and expected: it is the shared edge with
            // the next chunk, and drawing it is what closes the one-cell gap that
            // used to run around every chunk. The neighbour answers, through the
            // same world height function both chunks were generated from — so the
            // two agree on that column exactly, which is the whole reason a
            // world-absolute sampler was chosen in the first place.
            const core::u32 patchSize = view.chunkSize;
            const auto heightAt = [&chunk, patchSize, originX, originZ, &groundAt](core::u32 x, core::u32 z) {
                if (x < patchSize && z < patchSize)
                    return chunk.height.at(x, z).toFloat();
                return groundAt(originX + static_cast<core::i32>(x), originZ + static_cast<core::i32>(z));
            };
            // Shade and colour are CLAMPED rather than fetched from the neighbour.
            // A seam column lit or tinted by its own chunk is invisible; a seam
            // column that is not drawn at all is a slit to the horizon. The two
            // are not the same kind of wrong.
            const auto shadeAt = [&chunk, patchSize](core::u32 x, core::u32 z) {
                if (chunk.shade.empty())
                    return 0.0f;
                const core::u32 cx = x < patchSize ? x : patchSize - 1u;
                const core::u32 cz = z < patchSize ? z : patchSize - 1u;
                return static_cast<core::f32>(chunk.shade.at(cx, cz)) * (1.0f / 255.0f);
            };

            const core::u64 groundBegan = now();
            core::u32 triangles = render::drawHeightfieldPatch(
                target, matrix, patch, sun, heightAt, shadeAt,
                [&chunk, &palette, patchSize](core::u32 x, core::u32 z) {
                    return palette(
                        chunk.biomes.at(x < patchSize ? x : patchSize - 1u, z < patchSize ? z : patchSize - 1u));
                },
                [&](core::f32 wx, core::f32 wz, core::u32 base, core::f32 lit, core::f32 nx, core::f32 nz,
                    core::f32 occlusion) { return surface.shadeSurface(wx, wz, base, lit, nx, nz, occlusion, basis); },
                surface.perPixel());

            triangles += render::drawPatchSkirts(target, matrix, patch, view.skirtDrop, heightAt,
                                                 [&chunk, &palette, patchSize](core::u32 cx, core::u32 cz) {
                                                     return render::modulate(
                                                         palette(chunk.biomes.at(cx < patchSize ? cx : patchSize - 1u,
                                                                                 cz < patchSize ? cz : patchSize - 1u)),
                                                         0.45f);
                                                 });

            _groundCycles += now() - groundBegan;

            const core::u64 waterBegan = now();
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

            _waterCycles += now() - waterBegan;

            const core::u64 propBegan = now();
            // Props AFTER the ground of their own chunk: a boulder then tests
            // against a depth buffer that already holds the hill it stands on.
            if (ref.ring <= 1)
                triangles += props.drawRocks(target, matrix, basis, originX, originZ, view.chunkSize, params.seaLevel,
                                             math::Vec3<core::f32>(sun.x, sun.y, sun.z), params.ambient, heightAt);

            if (ref.ring <= 2)
                for (core::u32 i = 0u; i < chunk.plants.size(); ++i)
                {
                    if (!chunk.plants[i].standing)
                        continue; // grazed: the herd ate it, and the map says so
                    const core::u32 lx = static_cast<core::u32>(chunk.plants[i].cellX - originX);
                    const core::u32 lz = static_cast<core::u32>(chunk.plants[i].cellZ - originZ);
                    props.queuePlant(chunk.plants[i].cellX, chunk.plants[i].cellZ, chunk.height.at(lx, lz).toFloat(),
                                     chunk.shade.empty() ?
                                         1.0f :
                                         1.0f - 0.55f * (static_cast<core::f32>(chunk.shade.at(lx, lz)) / 255.0f));
                }
            _propCycles += now() - propBegan;
            return triangles;
        });

    _triangles = _view.stats().triangles;
    const core::u64 flushBegan = now();
    _triangles += props.flushPlants(rt, mvp, basis, haze);
    _propCycles += now() - flushBegan;

    const core::u64 herdBegan = now();
    _triangles += drawHerd(rt, mvp, registry, params, groundAt);
    _herdCycles += now() - herdBegan;
    return _triangles;
}

template <typename Palette, typename HeightAt, typename ColourAt, typename GroundAt>
core::u32 TerrainRenderer::drawBounded(const render::RenderTarget &rt, const render::OrbitCamera &camera,
                                       TerrainSurface &surface, const PropLibrary &props, const ecs::Registry &registry,
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
    _triangles += drawHerd(rt, mvp, registry, params, groundAt);
    return _triangles;
}

template <typename GroundAt>
core::u32 TerrainRenderer::drawHerd(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                    const ecs::Registry &registry, const TerrainDrawParams &params,
                                    GroundAt &&groundAt) const
{
    // Draws what is IN THE WORLD, not what a container remembers. Everything a
    // body needs to be drawn — where it is, what kind it is, how big it is — is a
    // component now, so this pass no longer takes an ecology::Herd at all: the
    // renderer and the simulation stopped having to agree about a second list.
    //
    // Read the side the WRITER wrote. A creature's components are written on the
    // back buffer, by the spawn and then by the locomotion system, and the buffer
    // swap is a phase callback the Engine installs after Physics — so a world whose
    // systems are all PrePhysics never publishes a front buffer at all. Reading the
    // front here drew every animal at the origin with a size of zero, which looks
    // exactly like no animals. (Props are the other way round: procgen writes them
    // on the front buffer and nothing moves them, which is why drawProps reads it.)
    core::u32 triangles = 0u;
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(ecs::ComponentId::Position) ||
            !part->archetype().has(ecs::ComponentId::Creature))
            continue;
        const bool hasGenome = part->archetype().has(ecs::ComponentId::Genome);

        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            const auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Position));
            const auto *creature = static_cast<const core::u32 *>(chunk->writeComponent(ecs::ComponentId::Creature));
            const auto *genome =
                hasGenome ? static_cast<const ecology::Genome *>(chunk->writeComponent(ecs::ComponentId::Genome)) :
                            nullptr;
            if (positions == nullptr || creature == nullptr)
                continue;

            for (core::u32 i = 0u; i < chunk->count(); ++i)
            {
                const core::u32 species = creature[i * 2u];
                const core::u32 id = creature[i * 2u + 1u];
                const core::f32 wx = positions[i].x.toFloat();
                const core::f32 wz = positions[i].z.toFloat();
                // The ground that is DRAWN, not the noise: the two differ by exactly
                // what erosion moved, and the herd hung in the air above the ridges
                // it lowered.
                const core::f32 ground = groundAt(positions[i].x.toInt(), positions[i].z.toInt());
                const core::f32 scale = genome != nullptr ? genome[i].size.toFloat() : 1.0f;
                const core::f32 size = params.bodyScale * scale;

                const ai::PersonalityTraits traits = ai::personalityOf(id, species);
                core::u32 tint = species == 1u ? params.hunterTint : params.grazerTint;
                if (traits.aggression > math::Fixed32::fromFloat(0.75f))
                    tint |= 0x00200000u;

                // A billboard rather than a box: at this scale a creature is a
                // handful of pixels, and six lit faces cost six times as much to
                // say the same.
                const core::f32 body[12] = {wx - size,
                                            ground,
                                            wz,
                                            wx + size,
                                            ground,
                                            wz,
                                            wx + size,
                                            ground + size * 2.4f,
                                            wz,
                                            wx - size,
                                            ground + size * 2.4f,
                                            wz};
                triangles += render::fillPolygonClipped(rt, mvp, body, 4u, tint);
            }
        }
    }
    return triangles;
}

template <typename Palette>
void TerrainRenderer::refreshProbe(TerrainStreamer &streamer, TerrainSurface &surface, const render::CameraBasis &basis,
                                   core::u32 frame, const TerrainDrawParams &params, Palette &&palette)
{
    surface.refreshProbe(
        frame, basis,
        [&](const render::RenderTarget &target, const math::Mat4<core::f32> &mirrorMvp, const render::CameraBasis &) {
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
                    [&chunk, &palette](core::u32 x, core::u32 z) { return palette(chunk.biomes.at(x, z)); },
                    [](core::f32, core::f32, core::u32 base, core::f32 lit, core::f32, core::f32, core::f32) {
                        return render::modulate(base, lit);
                    },
                    false);
            }
        });
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_RENDERER_INL
