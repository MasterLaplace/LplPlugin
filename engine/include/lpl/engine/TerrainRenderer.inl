#ifndef LPL_ENGINE_TERRAIN_RENDERER_INL
#define LPL_ENGINE_TERRAIN_RENDERER_INL

namespace lpl::engine {

namespace detail {

/**
 * @brief Surface speed of a channel per square root of its drop, in cells a second.
 *
 * The Chézy–Manning coefficient, folded into one number because everything else in it
 * (roughness, hydraulic radius) is constant for a one-cell channel.
 */
inline constexpr core::f32 kRiverCurrent = 2.4f;

/**
 * @brief How much of the wind a free surface carries, as a fraction.
 *
 * Three per cent, which is the figure for wind drift on open water. It is what makes a
 * LAKE move at all — a lake has no slope, so it has no current, and the wind is the only
 * thing left. That is the case the world's prevailing wind was modelled for.
 */
inline constexpr core::f32 kWindOnWater = 0.03f;

} // namespace detail

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

    const core::f32 groundAtFocus = params.useFocusHeight ?
                                        params.focusHeight :
                                        render::OrbitCamera::clamp(groundAt(static_cast<core::i32>(camera.focusX()),
                                                                            static_cast<core::i32>(camera.focusZ())),
                                                                   params.seaLevel, 1.0e6f);
    render::CameraBasis basis{};
    const auto mvp =
        camera.viewProjection(groundAtFocus, static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height),
                              render::CameraLens{params.fovRadians, params.nearPlane, params.farPlane}, basis);

    const bool underground = streamer
                                 .spanAt(static_cast<core::i32>(basis.eye.x), static_cast<core::i32>(basis.eye.z),
                                         math::Fixed32::fromFloat(basis.eye.y))
                                 .enclosed;

    {
        TerrainSurface::LampState lamp;
        lamp.on = underground || surface.sun().intensity <= 0.03f;
        lamp.x = basis.eye.x;
        lamp.z = basis.eye.z;
        core::f32 headingX = basis.forward.x;
        core::f32 headingZ = basis.forward.z;
        const core::f32 flat = render::approximateLength(headingX, headingZ);
        if (flat > 0.0001f)
        {
            headingX /= flat;
            headingZ /= flat;
        }
        lamp.headingX = headingX;
        lamp.headingZ = headingZ;
        lamp.coneInner = params.lampConeInner;
        lamp.coneOuter = params.lampConeOuter;
        lamp.reach = params.lampReach;
        lamp.tint = params.lampLitTint;
        surface.setLamp(lamp);
    }

    const core::u64 skyBegan = now();
    if (underground)
        surface.beginCaveFrame(rt, params.caveDarkTint, params.caveFogDensity);
    else
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
    viewParams.cameraX = basis.eye.x;
    viewParams.cameraZ = basis.eye.z;
    viewParams.curvatureFactor = effectiveCurvature(surface.params(), basis.eye.y);

    props.beginFrame();
    selectChunks(mvp, basis, rt.width, rt.height, viewParams, params, focusChunkX, focusChunkZ, streamer);

    if (!underground)
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

            const core::u32 patchSize = view.chunkSize;
            const auto heightAt = [&chunk, patchSize, originX, originZ, &groundAt](core::u32 x, core::u32 z) {
                if (x < patchSize && z < patchSize)
                    return chunk.height.at(x, z).toFloat();
                return groundAt(originX + static_cast<core::i32>(x), originZ + static_cast<core::i32>(z));
            };
            const auto shadeAt = [&chunk, patchSize](core::u32 x, core::u32 z) {
                if (chunk.shade.empty())
                    return 0.0f;
                const core::u32 cx = x < patchSize ? x : patchSize - 1u;
                const core::u32 cz = z < patchSize ? z : patchSize - 1u;
                return static_cast<core::f32>(chunk.shade.at(cx, cz)) * (1.0f / 255.0f);
            };

            core::i32 holeX[procgen::kMaxApertureCells * 2u];
            core::i32 holeZ[procgen::kMaxApertureCells * 2u];
            core::u32 holes = 0u;
            if (params.caveDrawRadius != 0u)
                streamer.forEachResidentWarren([&](const procgen::CaveWarren &warren) {
                    for (core::u32 a = 0u; a < warren.apertureCount && holes < procgen::kMaxApertureCells * 2u; ++a)
                    {
                        const core::i32 lx = warren.apertureX[a] - originX;
                        const core::i32 lz = warren.apertureZ[a] - originZ;
                        if (lx < 0 || lz < 0 || static_cast<core::u32>(lx) >= patchSize ||
                            static_cast<core::u32>(lz) >= patchSize)
                            continue;
                        holeX[holes] = lx;
                        holeZ[holes] = lz;
                        ++holes;
                    }
                });

            const core::u64 groundBegan = now();
            core::u32 triangles = render::drawHeightfieldPatch(
                target, matrix, patch, sun, heightAt, shadeAt,
                [&chunk, &palette, patchSize](core::u32 x, core::u32 z) {
                    return palette(
                        chunk.biomes.at(x < patchSize ? x : patchSize - 1u, z < patchSize ? z : patchSize - 1u));
                },
                [&](core::f32 wx, core::f32 wz, core::u32 base, core::f32 lit, core::f32 nx, core::f32 nz,
                    core::f32 occlusion) { return surface.shadeSurface(wx, wz, base, lit, nx, nz, occlusion, basis); },
                surface.perPixel(),
                [&holeX, &holeZ, holes, stride = patch.stride](core::u32 x, core::u32 z) {
                    const core::i32 lowX = static_cast<core::i32>(x);
                    const core::i32 lowZ = static_cast<core::i32>(z);
                    const core::i32 highX = lowX + static_cast<core::i32>(stride);
                    const core::i32 highZ = lowZ + static_cast<core::i32>(stride);
                    for (core::u32 i = 0u; i < holes; ++i)
                        if (holeX[i] >= lowX && holeX[i] <= highX && holeZ[i] >= lowZ && holeZ[i] <= highZ)
                            return true;
                    return false;
                });

            if (!underground)
                triangles += render::drawPatchSkirts(
                    target, matrix, patch, view.skirtDrop, heightAt,
                    [&chunk, &palette, patchSize](core::u32 cx, core::u32 cz) {
                        return render::modulate(palette(chunk.biomes.at(cx < patchSize ? cx : patchSize - 1u,
                                                                        cz < patchSize ? cz : patchSize - 1u)),
                                                0.45f);
                    });

            _groundCycles += now() - groundBegan;

            const core::u64 waterBegan = now();
            if (chunk.hasSea)
            {
                const core::u32 size = view.chunkSize;
                const core::u32 lowX = chunk.seaMinX > 0u ? chunk.seaMinX - 1u : 0u;
                const core::u32 lowZ = chunk.seaMinZ > 0u ? chunk.seaMinZ - 1u : 0u;
                const core::u32 highX = chunk.seaMaxX + 2u > size ? size : chunk.seaMaxX + 2u;
                const core::u32 highZ = chunk.seaMaxZ + 2u > size ? size : chunk.seaMaxZ + 2u;

                const core::f32 sx0 = static_cast<core::f32>(originX + static_cast<core::i32>(lowX));
                const core::f32 sx1 = static_cast<core::f32>(originX + static_cast<core::i32>(highX));
                const core::f32 sz0 = static_cast<core::f32>(originZ + static_cast<core::i32>(lowZ));
                const core::f32 sz1 = static_cast<core::f32>(originZ + static_cast<core::i32>(highZ));
                const core::f32 sea[12] = {sx0, params.seaLevel, sz0, sx1, params.seaLevel, sz0,
                                           sx1, params.seaLevel, sz1, sx0, params.seaLevel, sz1};
                const core::f32 pitch =
                    static_cast<core::f32>(view.chunkSize) /
                    static_cast<core::f32>(surface.waterTessellation() < 1u ? 1u : surface.waterTessellation());
                triangles += surface.drawWater(target, matrix, basis, sea, pitch, [&](core::f32 wx, core::f32 wz) {
                    return params.seaLevel - groundAt(static_cast<core::i32>(wx), static_cast<core::i32>(wz));
                });
            }

            if (params.riverSurfaceRise > 0.0f && chunk.hasRiver && ref.ring <= 1 && !chunk.flow.empty())
            {
                render::WaterParams river = surface.water();
                river.swellHeight = 0.0f;
                const core::u32 size = view.chunkSize;

                core::f32 chunkDriftX = 0.0f;
                core::f32 chunkDriftZ = 0.0f;
                core::f32 chunkDrop = 0.0f;
                core::u32 flowCells = 0u;
                for (core::u32 fz = 0u; fz < size; ++fz)
                    for (core::u32 fx = 0u; fx < size; ++fx)
                    {
                        if (chunk.rivers.at(fx, fz) == 0u)
                            continue;
                        const core::u8 dir = chunk.flow.at(fx, fz);
                        if (dir == procgen::kNoFlow || dir >= 8u)
                            continue;
                        chunkDriftX += static_cast<core::f32>(procgen::kNeighbor8X[dir]);
                        chunkDriftZ += static_cast<core::f32>(procgen::kNeighbor8Z[dir]);
                        ++flowCells;
                    }
                if (flowCells != 0u)
                {
                    chunkDrop = static_cast<core::f32>(flowCells);
                    river.setDrift(chunkDriftX, chunkDriftZ);
                }
                else
                {
                    river.setDrift(surface.water().driftX, surface.water().driftZ);
                }
                river.flowSpeed = 1.0f + detail::kRiverCurrent * 0.25f * (chunkDrop > 0.0f ? 1.0f : 0.0f);
                for (core::u32 rz = 0u; rz < size; ++rz)
                    for (core::u32 rx = 0u; rx < size; ++rx)
                    {
                        if (chunk.rivers.at(rx, rz) == 0u)
                            continue;
                        const core::i32 worldCellX = originX + static_cast<core::i32>(rx);
                        const core::i32 worldCellZ = originZ + static_cast<core::i32>(rz);

                        const auto cornerLevel = [&](core::i32 cornerX, core::i32 cornerZ) {
                            core::f32 lowest = groundAt(cornerX, cornerZ);
                            const core::f32 west = groundAt(cornerX - 1, cornerZ);
                            const core::f32 north = groundAt(cornerX, cornerZ - 1);
                            const core::f32 corner = groundAt(cornerX - 1, cornerZ - 1);
                            lowest = west < lowest ? west : lowest;
                            lowest = north < lowest ? north : lowest;
                            lowest = corner < lowest ? corner : lowest;
                            return lowest + params.riverSurfaceRise;
                        };

                        const core::f32 y00 = cornerLevel(worldCellX, worldCellZ);
                        const core::f32 y10 = cornerLevel(worldCellX + 1, worldCellZ);
                        const core::f32 y11 = cornerLevel(worldCellX + 1, worldCellZ + 1);
                        const core::f32 y01 = cornerLevel(worldCellX, worldCellZ + 1);

                        core::f32 low = y00;
                        core::f32 high = y00;
                        for (const core::f32 corner : {y10, y11, y01})
                        {
                            low = corner < low ? corner : low;
                            high = corner > high ? corner : high;
                        }

                        if (high <= params.seaLevel)
                            continue;

                        if (high - low > params.riverSurfaceRise * 2.0f)
                            continue;

                        const core::f32 wx0 = static_cast<core::f32>(worldCellX);
                        const core::f32 wz0 = static_cast<core::f32>(worldCellZ);
                        const core::f32 cell[12] = {wx0,        y00, wz0,        wx0 + 1.0f, y10, wz0,
                                                    wx0 + 1.0f, y11, wz0 + 1.0f, wx0,        y01, wz0 + 1.0f};
                        const core::f32 bed00 = groundAt(worldCellX, worldCellZ);
                        const core::f32 bed10 = groundAt(worldCellX + 1, worldCellZ);
                        const core::f32 bed11 = groundAt(worldCellX + 1, worldCellZ + 1);
                        const core::f32 bed01 = groundAt(worldCellX, worldCellZ + 1);
                        triangles += surface.drawWaterWith(
                            target, matrix, basis, cell, river, 1.0f, [&](core::f32 sx, core::f32 sz) {
                                core::f32 u = sx - wx0;
                                core::f32 v = sz - wz0;
                                u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
                                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                                const core::f32 bedHere = bed00 * (1.0f - u) * (1.0f - v) + bed10 * u * (1.0f - v) +
                                                          bed01 * (1.0f - u) * v + bed11 * u * v;
                                const core::f32 topHere = y00 * (1.0f - u) * (1.0f - v) + y10 * u * (1.0f - v) +
                                                          y01 * (1.0f - u) * v + y11 * u * v;
                                return topHere - bedHere;
                            });
                    }
            }

            _waterCycles += now() - waterBegan;

            const core::u64 propBegan = now();

            if (ref.ring <= 2 && !chunk.buildings.empty())
            {
                const math::Vec3<core::f32> sunDirection{sun.x, sun.y, sun.z};
                for (core::u32 b = 0u; b < chunk.buildings.size(); ++b)
                {
                    const procgen::LandmarkBuilding &building = chunk.buildings[b];
                    const core::f32 wallTop = building.baseY + building.height * 0.76f;
                    triangles += render::drawBox(
                        target, matrix, building.minX, building.baseY, building.minZ, building.maxX, wallTop,
                        building.maxZ,
                        render::modulate(params.buildingTint,
                                         0.86f + 0.07f * static_cast<core::f32>(building.district & 3u)),
                        sunDirection, params.ambient);
                    triangles += render::drawBox(target, matrix, building.minX - 0.25f, wallTop, building.minZ - 0.25f,
                                                 building.maxX + 0.25f, building.baseY + building.height,
                                                 building.maxZ + 0.25f, params.roofTint, sunDirection, params.ambient);
                }
            }

            if (ref.ring <= 2 && !chunk.caveMouths.empty())
            {
                for (core::u32 m = 0u; m < chunk.caveMouths.size(); ++m)
                {
                    const procgen::LandmarkSite &site = chunk.caveMouths[m];
                    bool real = false;
                    for (core::u32 w = 0u; w < chunk.warrens.size() && !real; ++w)
                        real = chunk.warrens[w].site.cellX == site.cellX && chunk.warrens[w].site.cellZ == site.cellZ;
                    if (real)
                        continue;
                    const core::f32 fx = static_cast<core::f32>(procgen::kNeighbor8X[site.facing & 7u]);
                    const core::f32 fz = static_cast<core::f32>(procgen::kNeighbor8Z[site.facing & 7u]);
                    const core::f32 reach = static_cast<core::f32>(site.radius);
                    const core::f32 backX = static_cast<core::f32>(site.cellX) - fx * reach * 0.8f;
                    const core::f32 backZ = static_cast<core::f32>(site.cellZ) - fz * reach * 0.8f;
                    const core::f32 px = -fz;
                    const core::f32 pz = fx;
                    const core::f32 half = reach * 0.45f;
                    const core::f32 floorY = site.height - params.caveMouthDrop;
                    const core::f32 topY = floorY + params.mouthHeight;

                    const core::f32 opening[12] = {
                        backX - px * half, floorY, backZ - pz * half, backX + px * half, floorY, backZ + pz * half,
                        backX + px * half, topY,   backZ + pz * half, backX - px * half, topY,   backZ - pz * half};
                    triangles += render::fillPolygonClipped(target, matrix, opening, 4u, params.caveMouthTint);
                }
            }

            if (ref.ring <= 1)
                triangles += props.drawRocks(target, matrix, basis, originX, originZ, view.chunkSize, params.seaLevel,
                                             math::Vec3<core::f32>(sun.x, sun.y, sun.z), params.ambient, heightAt);

            if (ref.ring <= 2)
                for (core::u32 i = 0u; i < chunk.plants.size(); ++i)
                {
                    if (!chunk.plants[i].standing)
                        continue;
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
    _triangles += drawWarrens(rt, mvp, streamer, basis, params, surface.sun());
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

    render::HeightfieldPatchParams patch;
    patch.size = gridWidth < gridDepth ? gridWidth : gridDepth;
    patch.stride = 1u;
    patch.originX = -static_cast<core::i32>(halfX);
    patch.originZ = -static_cast<core::i32>(halfZ);
    patch.ambient = params.ambient;
    patch.cameraX = basis.eye.x;
    patch.cameraZ = basis.eye.z;
    patch.curvatureFactor = effectiveCurvature(surface.params(), basis.eye.y);

    _triangles += render::drawHeightfieldPatch(
        rt, mvp, patch, surface.sun(), heightAt, [](core::u32, core::u32) { return 0.0f; }, colourAt,
        [&](core::f32 wx, core::f32 wz, core::u32 base, core::f32 lit, core::f32 nx, core::f32 nz,
            core::f32 occlusion) { return surface.shadeSurface(wx, wz, base, lit, nx, nz, occlusion, basis); },
        surface.perPixel());

    if (params.boundedHasSea)
    {
        const core::f32 sea[12] = {-halfX, params.seaLevel, -halfZ, halfX,  params.seaLevel, -halfZ,
                                   halfX,  params.seaLevel, halfZ,  -halfX, params.seaLevel, halfZ};
        const core::f32 pitch =
            static_cast<core::f32>(params.chunkSize) /
            static_cast<core::f32>(surface.waterTessellation() < 1u ? 1u : surface.waterTessellation());
        _triangles += surface.drawWater(rt, mvp, basis, sea, pitch, [&](core::f32 wx, core::f32 wz) {
            return params.seaLevel - groundAt(static_cast<core::i32>(wx), static_cast<core::i32>(wz));
        });
    }

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
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->latestComponent(ecs::ComponentId::Position));
            const auto *creature = static_cast<const core::u32 *>(chunk->latestComponent(ecs::ComponentId::Creature));
            const auto *genome =
                hasGenome ? static_cast<const ecology::Genome *>(chunk->latestComponent(ecs::ComponentId::Genome)) :
                            nullptr;
            if (positions == nullptr || creature == nullptr)
                continue;

            for (core::u32 i = 0u; i < chunk->count(); ++i)
            {
                const core::u32 species = creature[i * 2u];
                const core::u32 id = creature[i * 2u + 1u];
                const core::f32 wx = positions[i].x.toFloat();
                const core::f32 wz = positions[i].z.toFloat();
                const core::f32 ground = groundAt(positions[i].x.toInt(), positions[i].z.toInt());
                const core::f32 scale = genome != nullptr ? genome[i].size.toFloat() : 1.0f;
                const core::f32 size = params.bodyScale * scale;

                const ai::PersonalityTraits traits = ai::personalityOf(id, species);
                core::u32 tint = species == 1u ? params.hunterTint : params.grazerTint;
                if (traits.aggression > math::Fixed32::fromFloat(0.75f))
                    tint |= 0x00200000u;

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
    bool waterInSight = false;
    for (core::u32 v = 0u; v < _view.visible().size() && !waterInSight; ++v)
        waterInSight = streamer.at(_view.visible()[v].index).hasSea;

    surface.refreshProbe(
        frame, basis, waterInSight,
        [&](const render::RenderTarget &target, const math::Mat4<core::f32> &mirrorMvp, const render::CameraBasis &) {
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
                patch.cameraX = basis.eye.x;
                patch.cameraZ = basis.eye.z;
                patch.curvatureFactor = effectiveCurvature(surface.params(), basis.eye.y);

                const core::u32 patchSize = params.chunkSize;
                const auto probeHeight = [&chunk, &streamer, patchSize, &patch](core::u32 x, core::u32 z) {
                    if (x < patchSize && z < patchSize)
                        return chunk.height.at(x, z).toFloat();
                    return streamer.groundAt(patch.originX + static_cast<core::i32>(x),
                                             patch.originZ + static_cast<core::i32>(z));
                };
                const auto probeColour = [&chunk, &palette, patchSize](core::u32 x, core::u32 z) {
                    return palette(
                        chunk.biomes.at(x < patchSize ? x : patchSize - 1u, z < patchSize ? z : patchSize - 1u));
                };

                render::drawHeightfieldPatch(
                    target, mirrorMvp, patch, surface.sun(), probeHeight, [](core::u32, core::u32) { return 0.0f; },
                    probeColour,
                    [](core::f32, core::f32, core::u32 base, core::f32 lit, core::f32, core::f32, core::f32) {
                        return render::modulate(base, lit);
                    },
                    false);
            }
        });
}

inline core::u32 TerrainRenderer::drawWarrens(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                              const TerrainStreamer &streamer, const render::CameraBasis &basis,
                                              const TerrainDrawParams &params, const render::SunState &sun) const
{
    if (params.caveDrawRadius == 0u)
        return 0u;

    core::u32 triangles = 0u;
    const core::i32 reach = static_cast<core::i32>(params.caveDrawRadius);
    constexpr core::i32 kDoorwaySightCells = 72;
    const core::i32 eyeX = static_cast<core::i32>(basis.eye.x);
    const core::i32 eyeZ = static_cast<core::i32>(basis.eye.z);

    streamer.forEachResidentWarren([&](const procgen::CaveWarren &warren) {
        if (warren.volume.empty() || warren.levelHeight <= 0.0f)
            return;

        core::i32 lowX = 0;
        core::i32 lowZ = 0;
        core::i32 highX = 0;
        core::i32 highZ = 0;
        bool any = false;
        const auto include = [&](core::i32 cellX, core::i32 cellZ, core::i32 span) {
            const core::i32 x0 = cellX - span - warren.originX;
            const core::i32 z0 = cellZ - span - warren.originZ;
            const core::i32 x1 = cellX + span + 1 - warren.originX;
            const core::i32 z1 = cellZ + span + 1 - warren.originZ;
            if (x1 <= 0 || z1 <= 0 || x0 >= static_cast<core::i32>(warren.volume.width) ||
                z0 >= static_cast<core::i32>(warren.volume.depth))
                return;
            lowX = !any || x0 < lowX ? x0 : lowX;
            lowZ = !any || z0 < lowZ ? z0 : lowZ;
            highX = !any || x1 > highX ? x1 : highX;
            highZ = !any || z1 > highZ ? z1 : highZ;
            any = true;
        };

        const core::i32 eyeLocalX = eyeX - warren.originX;
        const core::i32 eyeLocalZ = eyeZ - warren.originZ;
        const bool eyeInside = eyeLocalX >= 0 && eyeLocalZ >= 0 &&
                               eyeLocalX < static_cast<core::i32>(warren.volume.width) &&
                               eyeLocalZ < static_cast<core::i32>(warren.volume.depth);
        if (eyeInside)
            include(eyeX, eyeZ, reach);
        for (core::u32 a = 0u; a < warren.apertureCount && !eyeInside; ++a)
        {
            const core::i32 dx = warren.apertureX[a] - eyeX;
            const core::i32 dz = warren.apertureZ[a] - eyeZ;
            if (dx * dx + dz * dz > kDoorwaySightCells * kDoorwaySightCells)
                continue;
            include(warren.apertureX[a], warren.apertureZ[a], reach);
        }
        if (!any)
            return;

        procgen::VoxelWindow window;
        window.minX = lowX < 0 ? 0u : static_cast<core::u32>(lowX);
        window.minZ = lowZ < 0 ? 0u : static_cast<core::u32>(lowZ);
        window.maxX = static_cast<core::u32>(highX);
        window.maxZ = static_cast<core::u32>(highZ);

        const core::f32 originX = static_cast<core::f32>(warren.originX);
        const core::f32 originZ = static_cast<core::f32>(warren.originZ);
        const core::f32 levelHeight = warren.levelHeight;

        procgen::forEachVoxelFace(
            warren.volume, window, [](core::i32, core::i32, core::i32) { return true; },
            [&](const core::f32 quad[12], core::f32 nx, core::f32 ny, core::f32 nz, core::u8, core::u32 x, core::u32,
                core::u32 z) {
                if (warren.covered.at(x, z) == 0u)
                    return;

                core::f32 world[12];
                for (core::u32 v = 0u; v < 4u; ++v)
                {
                    world[v * 3u] = originX + quad[v * 3u];
                    world[v * 3u + 1u] = warren.baseY + quad[v * 3u + 1u] * levelHeight;
                    world[v * 3u + 2u] = originZ + quad[v * 3u + 2u];
                }

                const core::f32 midX = (world[0] + world[6]) * 0.5f;
                const core::f32 midY = (world[1] + world[7]) * 0.5f;
                const core::f32 midZ = (world[2] + world[8]) * 0.5f;
                core::f32 toEyeX = basis.eye.x - midX;
                core::f32 toEyeY = basis.eye.y - midY;
                core::f32 toEyeZ = basis.eye.z - midZ;
                const core::f32 distance = render::approximateLength(toEyeX, toEyeZ);
                const core::f32 length = distance + (toEyeY < 0.0f ? -toEyeY : toEyeY) + 0.001f;
                toEyeX /= length;
                toEyeY /= length;
                toEyeZ /= length;
                core::f32 facing = nx * toEyeX + ny * toEyeY + nz * toEyeZ;
                facing = facing < 0.0f ? 0.0f : facing;

                const core::f32 offAxis =
                    -(toEyeX * basis.forward.x + toEyeY * basis.forward.y + toEyeZ * basis.forward.z);
                const core::f32 band = params.lampConeInner - params.lampConeOuter;
                core::f32 cone = band > 0.0001f ? (offAxis - params.lampConeOuter) / band : 1.0f;
                cone = cone < 0.0f ? 0.0f : (cone > 1.0f ? 1.0f : cone);
                cone *= cone;

                core::f32 reach = params.lampReach > 0.0001f ? 1.0f - distance / params.lampReach : 0.0f;
                reach = reach < 0.0f ? 0.0f : reach;

                const core::f32 beam = facing * cone * reach;
                const core::f32 lit = params.lampAmbient + (1.0f - params.lampAmbient) * beam;
                const core::u32 stone = render::mixColours(params.caveRockTint, params.lampLitTint, beam);
                const core::u32 shaded = render::applyAerialPerspective(
                    render::modulate(stone, lit), params.caveDarkTint, distance, params.caveFogDensity);
                triangles += render::fillPolygonClipped(rt, mvp, world, 4u, shaded);
            });
    });

    (void) sun;
    return triangles;
}

inline void TerrainRenderer::resetPhaseCounters() noexcept
{
    _skyCycles = 0u;
    _groundCycles = 0u;
    _waterCycles = 0u;
    _propCycles = 0u;
    _herdCycles = 0u;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_RENDERER_INL
