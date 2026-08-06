/**
 * @file MapMesh.hpp
 * @brief Turning what this module generated into triangles you can look at.
 *
 * The geometric counterpart of @ref MapShading: one says what colour a cell is,
 * this says where its corners are. Together they are the whole of "how do I look
 * at a generated world", and both of them lived inside a viewer's `main.cpp` —
 * six hundred lines of it — where nothing could test them, nothing could reuse
 * them, and the editor was about to grow a second set.
 *
 * That is not a tidiness complaint. Code in an app has **no test target**: a
 * mesher that emits an inside-out wall, forgets a boundary face or shears a
 * building into a staircase can only be caught by a human looking at a picture,
 * and every one of those three happened here. Moved into a module they become
 * assertable — a vertex count is arithmetic, a fold is a fold.
 *
 * What each of these knows, which is the part worth keeping:
 *  - **only boundary faces.** Rock is not drawn, and drawing the inside of a
 *    solid makes every wall two coincident surfaces whose z-fighting reads as
 *    noise on the roofs rather than as the "too many quads" it is.
 *  - **cap the walls**, so a plan read from above reads as a plan.
 *  - **one datum per footprint**, not per column: a plot on a slope whose columns
 *    each start at their own ground shears into a staircase, and along a ridge into
 *    a long wall standing free of the hillside.
 *  - **the underground follows the terrain** at a fixed depth rather than lying on
 *    a flat plane the surface never touches.
 *  - **outside the map counts as rock**, so a cave is walled in rather than open to
 *    nothing at the border.
 *
 * Non-authoritative throughout: these are floats, they are geometry for an eye, and
 * nothing here may feed a simulation. @c lpl::pmr::sqrt rather than @c std::sqrt for
 * the normals — the module is libm-free by contract, and the viewer's copy of this
 * code reached for @c std::sqrt and @c std::log where the module already had
 * @c pmr::sqrt and @ref fixedLog2.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_MAP_MESH_HPP
#    define LPL_PROCGEN_MAP_MESH_HPP

#    include <lpl/procgen/Extrusion.hpp>
#    include <lpl/procgen/Liminal.hpp>
#    include <lpl/procgen/MapShading.hpp>
#    include <lpl/std/cmath.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/**
 * @struct MapVertex
 * @brief Position, normal and a per-vertex colour.
 *
 * Deliberately not @c render::Vertex, which carries texture coordinates and no
 * colour: an instrument mesh is flat-shaded from a palette and has no material.
 * Keeping them apart means neither has a field the other ignores.
 */
struct MapVertex {
    float x, y, z;
    float nx, ny, nz;
    Rgb colour;
};

/// A triangle list. Two triangles per quad, no index buffer: these are rebuilt
/// per regeneration, not streamed.
using MapMesh = lpl::pmr::vector<MapVertex>;

/**
 * @brief Appends one quad as two triangles, with a shared flat normal.
 *
 * This exact lambda was written out FIVE times in the viewer, once per mesher,
 * and each copy was a chance to wind a face the wrong way round.
 *
 * Corner order is a-b-c-d around the face; the triangles are (a,b,c) and (a,c,d),
 * so the winding of the caller's corners IS the facing.
 */
inline void appendQuad(MapMesh &mesh, float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                       float cz, float dx, float dy, float dz, float nx, float ny, float nz, const Rgb &colour)
{
    const MapVertex a{ax, ay, az, nx, ny, nz, colour};
    const MapVertex b{bx, by, bz, nx, ny, nz, colour};
    const MapVertex c{cx, cy, cz, nx, ny, nz, colour};
    const MapVertex d{dx, dy, dz, nx, ny, nz, colour};
    mesh.push_back(a);
    mesh.push_back(b);
    mesh.push_back(c);
    mesh.push_back(a);
    mesh.push_back(c);
    mesh.push_back(d);
}

/** @brief Appends one triangle with a flat normal. The @ref appendQuad of odd faces. */
inline void appendTriangle(MapMesh &mesh, float ax, float ay, float az, float bx, float by, float bz, float cx,
                           float cy, float cz, float nx, float ny, float nz, const Rgb &colour)
{
    mesh.push_back(MapVertex{ax, ay, az, nx, ny, nz, colour});
    mesh.push_back(MapVertex{bx, by, bz, nx, ny, nz, colour});
    mesh.push_back(MapVertex{cx, cy, cz, nx, ny, nz, colour});
}

/**
 * @struct MapSurfaceStyle
 * @brief What the surface shows on top of its shading.
 *
 * The three overlays are drawn INTO the surface rather than over it, because each
 * is a property of the ground and a separate overlay would z-fight with it.
 */
struct MapSurfaceStyle {
    MapShading shading{MapShading::Biome}; ///< Which quantity colours a cell.
    core::u32 climateAxis{0u};             ///< Which axis, when shading by climate.
    bool rivers{true};                     ///< Paint river cells blue.
    bool settlement{true};                 ///< Paint roads, plazas and plots.
    bool roads{true};                      ///< Paint the long-distance network, last.
};

/**
 * @brief The terrain surface, one quad per cell, centred on the origin.
 *
 * Per-cell normals from the central difference, which is what makes relief read at
 * all: flat shading a heightfield is very nearly invisible.
 */
[[nodiscard]] inline MapMesh buildSurfaceMesh(const WorldAtlas &atlas, const MapSurfaceStyle &style)
{
    MapMesh mesh;
    if (atlas.height.empty())
        return mesh;

    const core::u32 width = atlas.height.width();
    const core::u32 depth = atlas.height.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;

    lpl::pmr::vector<MapVertex> grid(static_cast<core::usize>(width) * depth);
    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            const core::i32 ix = static_cast<core::i32>(x);
            const core::i32 iz = static_cast<core::i32>(z);
            const float left = atlas.height.clamped(ix - 1, iz).toFloat();
            const float right = atlas.height.clamped(ix + 1, iz).toFloat();
            const float back = atlas.height.clamped(ix, iz - 1).toFloat();
            const float front = atlas.height.clamped(ix, iz + 1).toFloat();

            float nx = left - right;
            float ny = 2.0f;
            float nz = back - front;
            const float length = lpl::pmr::sqrt(nx * nx + ny * ny + nz * nz);
            if (length > 0.0f)
            {
                nx /= length;
                ny /= length;
                nz /= length;
            }

            Rgb colour = surfaceColour(atlas, style.shading, style.climateAxis, x, z);
            // Rivers, drawn as part of the surface rather than over it: a river is
            // a property of the ground, and an overlay would z-fight with it.
            if (style.rivers && !atlas.rivers.empty() && atlas.rivers.at(x, z) != 0u)
                colour = {0.16f, 0.42f, 0.72f};
            if (style.settlement && !atlas.settlement.empty())
            {
                switch (atlas.settlement.at(x, z))
                {
                case SettlementCell::Road: colour = {0.35f, 0.32f, 0.30f}; break;
                case SettlementCell::Plaza: colour = {0.55f, 0.50f, 0.44f}; break;
                case SettlementCell::Plot: colour = {0.82f, 0.55f, 0.22f}; break;
                default: break;
                }
            }
            // The highway network, on top of the town's own streets: it is the
            // long-distance layer, grown by the grammar and steered by the field,
            // where the settlement's roads are district borders. Drawn last so a
            // road crossing a town reads as passing through it.
            if (style.roads && !atlas.roads.empty() && atlas.roads.at(x, z) != 0u)
                colour = {0.24f, 0.22f, 0.20f};

            grid[atlas.height.index(x, z)] = MapVertex{static_cast<float>(x) - halfW,
                                                       atlas.height.at(x, z).toFloat(),
                                                       static_cast<float>(z) - halfD,
                                                       nx,
                                                       ny,
                                                       nz,
                                                       colour};
        }
    }

    mesh.reserve(static_cast<core::usize>(width - 1u) * (depth - 1u) * 6u);
    for (core::u32 z = 0u; z + 1u < depth; ++z)
    {
        for (core::u32 x = 0u; x + 1u < width; ++x)
        {
            const MapVertex &a = grid[atlas.height.index(x, z)];
            const MapVertex &b = grid[atlas.height.index(x + 1u, z)];
            const MapVertex &c = grid[atlas.height.index(x + 1u, z + 1u)];
            const MapVertex &d = grid[atlas.height.index(x, z + 1u)];
            mesh.push_back(a);
            mesh.push_back(b);
            mesh.push_back(c);
            mesh.push_back(a);
            mesh.push_back(c);
            mesh.push_back(d);
        }
    }
    return mesh;
}

/**
 * @brief The flat underground as a real volume: a floor, and walls around it.
 *
 * A sheet of coloured quads is not a cave. What makes a dungeon legible is the
 * WALLS — the boundary between the carved space and the rock — because that is the
 * only thing that shows a corridor as a corridor and a chamber as a chamber. Each
 * open cell contributes a floor quad, and each of its solid neighbours a vertical
 * face: the mesh is the surface of the void, not a picture of it.
 *
 * @param atlas      The world; its heightfield puts the cave under the ground.
 * @param depthBelow World units between the ground and the cave floor.
 */
[[nodiscard]] inline MapMesh buildDungeonMesh(const WorldAtlas &atlas, float depthBelow)
{
    MapMesh mesh;
    if (atlas.dungeon.empty())
        return mesh;

    const core::u32 width = atlas.dungeon.width();
    const core::u32 depth = atlas.dungeon.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;
    const float wallHeight = 1.6f;

    const Rgb floorColour{0.52f, 0.44f, 0.36f};
    const Rgb wallColour{0.30f, 0.26f, 0.24f};
    const Rgb capColour{0.66f, 0.44f, 0.20f};

    const auto solid = [&atlas](core::i32 x, core::i32 z) {
        // Outside the map counts as rock, so the cave is walled in rather than open
        // to nothing at the border.
        if (!atlas.dungeon.contains(x, z))
            return true;
        return atlas.dungeon.at(static_cast<core::u32>(x), static_cast<core::u32>(z)) == DungeonCell::Wall;
    };

    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            const core::i32 ix = static_cast<core::i32>(x);
            const core::i32 iz = static_cast<core::i32>(z);
            if (solid(ix, iz))
                continue;

            // The cave hangs a fixed distance below the ground above it, so it follows
            // the terrain instead of lying on a flat plane the surface never touches.
            const float ground = atlas.height.empty() ? 0.0f : atlas.height.clamped(ix, iz).toFloat();
            const float floorY = ground - depthBelow;
            const float ceilY = floorY + wallHeight;

            const float x0 = static_cast<float>(x) - halfW;
            const float x1 = x0 + 1.0f;
            const float z0 = static_cast<float>(z) - halfD;
            const float z1 = z0 + 1.0f;

            appendQuad(mesh, x0, floorY, z0, x1, floorY, z0, x1, floorY, z1, x0, floorY, z1, 0.0f, 1.0f, 0.0f,
                       floorColour);

            // A wall wherever the rock begins.
            if (solid(ix + 1, iz))
                appendQuad(mesh, x1, floorY, z0, x1, ceilY, z0, x1, ceilY, z1, x1, floorY, z1, 1.0f, 0.0f, 0.0f,
                           wallColour);
            if (solid(ix - 1, iz))
                appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x0, ceilY, z1, x0, floorY, z1, -1.0f, 0.0f, 0.0f,
                           wallColour);
            if (solid(ix, iz + 1))
                appendQuad(mesh, x0, floorY, z1, x0, ceilY, z1, x1, ceilY, z1, x1, floorY, z1, 0.0f, 0.0f, 1.0f,
                           wallColour);
            if (solid(ix, iz - 1))
                appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x1, ceilY, z0, x1, floorY, z0, 0.0f, 0.0f, -1.0f,
                           wallColour);

            // Cap the top of the walls, so from above the plan reads as a plan.
            if (solid(ix + 1, iz) || solid(ix - 1, iz) || solid(ix, iz + 1) || solid(ix, iz - 1))
                appendQuad(mesh, x0, ceilY, z0, x1, ceilY, z0, x1, ceilY, z1, x0, ceilY, z1, 0.0f, 1.0f, 0.0f,
                           capColour);
        }
    }
    return mesh;
}

/**
 * @brief Raises the settlement: one box per footprint, sunk to its lowest ground.
 *
 * Painting plots onto the ground says where a town is. It does not put a town
 * there — a settlement is read by its silhouette, and a silhouette needs height.
 * Each footprint becomes one box whose height comes from its own area and district,
 * hashed, so a quarter has a character rather than a uniform skyline.
 */
[[nodiscard]] inline MapMesh buildTownMesh(const WorldAtlas &atlas)
{
    MapMesh mesh;
    if (atlas.height.empty())
        return mesh;

    const float halfW = static_cast<float>(atlas.height.width()) * 0.5f;
    const float halfD = static_cast<float>(atlas.height.depth()) * 0.5f;

    const auto box = [&mesh](float x0, float x1, float y0, float y1, float z0, float z1, const Rgb &wall,
                             const Rgb &roof) {
        appendQuad(mesh, x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1, 0.0f, 1.0f, 0.0f, roof);
        appendQuad(mesh, x0, y0, z1, x0, y1, z1, x1, y1, z1, x1, y0, z1, 0.0f, 0.0f, 1.0f, wall);
        appendQuad(mesh, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z0, 0.0f, 0.0f, -1.0f, wall);
        appendQuad(mesh, x1, y0, z1, x1, y1, z1, x1, y1, z0, x1, y0, z0, 1.0f, 0.0f, 0.0f, wall);
        appendQuad(mesh, x0, y0, z0, x0, y1, z0, x0, y1, z1, x0, y0, z1, -1.0f, 0.0f, 0.0f, wall);
    };

    for (core::usize p = 0u; p < atlas.plots.size(); ++p)
    {
        const BuildingPlot &plot = atlas.plots[p];

        // Sink the base to the lowest ground the footprint covers: a box placed at the
        // centre height would hang off the downhill corner.
        float lowest = 1.0e9f;
        for (core::u32 z = plot.z; z < plot.z + plot.depth; ++z)
            for (core::u32 x = plot.x; x < plot.x + plot.width; ++x)
            {
                const float h = atlas.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
                if (h < lowest)
                    lowest = h;
            }
        if (lowest > 1.0e8f)
            continue;

        // Height from the footprint's area and its district, hashed: a big plot on a
        // busy district gets a tall building, and the variation is reproducible.
        const core::u32 area = plot.width * plot.depth;
        const core::u32 h = ValueNoise2D::hash2(static_cast<core::i32>(plot.x), static_cast<core::i32>(plot.z),
                                                static_cast<core::u32>(plot.district) + 1u);
        const float storeys = 1.0f + static_cast<float>(area) * 0.28f + static_cast<float>(h & 0x7u) * 0.55f;

        const float x0 = static_cast<float>(plot.x) - halfW + 0.12f;
        const float x1 = static_cast<float>(plot.x + plot.width) - halfW - 0.12f;
        const float z0 = static_cast<float>(plot.z) - halfD + 0.12f;
        const float z1 = static_cast<float>(plot.z + plot.depth) - halfD - 0.12f;
        const float y0 = lowest - 0.4f;
        const float y1 = lowest + storeys;

        const float tint = static_cast<float>((h >> 8) & 0x3Fu) / 63.0f;
        const Rgb wall{0.58f + 0.22f * tint, 0.48f + 0.16f * tint, 0.38f + 0.10f * tint};
        const Rgb roof{0.34f + 0.10f * tint, 0.22f + 0.06f * tint, 0.20f};
        box(x0, x1, y0, y1, z0, z1, wall, roof);
    }
    return mesh;
}

/**
 * @brief Meshes a voxel volume into world coordinates, exposed faces only.
 *
 * The core of it, with no opinion about where the volume SITS. Extracted because a
 * streamed village needs exactly this walk and cannot supply a @ref WorldAtlas — an atlas
 * is a bounded map's product, and an endless world has no such thing. Writing the walk a
 * second time would have been the sixth copy of "emit a quad per exposed face" in this
 * repository, and every copy is a chance to wind one the wrong way round.
 *
 * @param mesh        Appended to.
 * @param volume      What to mesh.
 * @param originX     World x of the volume's (0, *, 0) corner.
 * @param originZ     World z of it.
 * @param groundAt    (localX, localZ) -> the ground this COLUMN stands on. A building
 *                    passes one level for its whole footprint: a plot on a slope with each
 *                    column at its own ground shears into a staircase.
 * @param baseLift    World units between that ground and level 0.
 * @param palette     Colour per material id; index 0 is never drawn.
 * @param paletteSize Entries in @p palette.
 */
template <typename GroundAt>
void appendVoxelFaces(MapMesh &mesh, const VoxelVolume &volume, float originX, float originZ, GroundAt &&groundAt,
                      float baseLift, const Rgb *palette, core::u32 paletteSize)
{
    if (volume.empty() || palette == nullptr || paletteSize == 0u)
        return;

    // The walk itself is @ref forEachVoxelFace's, and this is now only its sink: place
    // the face in world units and give it a colour. A second consumer arrived that has
    // no vertex buffer to append to, and rewriting the walk for it would have been the
    // sixth copy of the six faces — with the winding to get right a sixth time.
    forEachVoxelFace(
        volume, VoxelWindow{},
        // Outside the array is EMPTY for a building: it stands in the air, so its outer
        // faces are the ones a viewer sees. A cave answers the opposite way.
        [](core::i32, core::i32, core::i32) { return false; },
        [&](const float quad[12], float nx, float ny, float nz, core::u8 material, core::u32 x, core::u32 /*y*/,
            core::u32 z) {
            const Rgb colour = palette[material < paletteSize ? material : 0u];
            const float lift = groundAt(x, z) + baseLift;
            float world[12];
            for (core::u32 v = 0u; v < 4u; ++v)
            {
                world[v * 3u] = originX + quad[v * 3u];
                world[v * 3u + 1u] = lift + quad[v * 3u + 1u];
                world[v * 3u + 2u] = originZ + quad[v * 3u + 2u];
            }
            appendQuad(mesh, world[0], world[1], world[2], world[3], world[4], world[5], world[6], world[7], world[8],
                       world[9], world[10], world[11], nx, ny, nz, colour);
        });
}

/**
 * @brief Meshes a voxel volume, emitting only the faces that border empty space.
 *
 * The generic mesher the grammar's products need: the town raised by @ref buildTown
 * and the fences and lamps @ref decoratePath leaves along the roads are both a
 * @ref VoxelVolume, so both arrive here.
 *
 * @param volume     What to mesh.
 * @param atlas      The terrain, so the volume sits on the ground it was planned on.
 * @param baseLift   World units between the terrain and level 0 of the volume.
 * @param palette    Colour per material id; index 0 is never drawn.
 * @param paletteSize Entries in @p palette.
 * @param datum      One ground height per cell — the FOOTPRINT's datum, not the
 *                   column's. Null falls back to the column, which is right for
 *                   roadside decoration and wrong for a building.
 * @param datumCount Entries in @p datum.
 */
[[nodiscard]] inline MapMesh buildVoxelMesh(const VoxelVolume &volume, const WorldAtlas &atlas, float baseLift,
                                            const Rgb *palette, core::u32 paletteSize, const float *datum = nullptr,
                                            core::usize datumCount = 0u)
{
    MapMesh mesh;
    if (volume.empty() || atlas.height.empty())
        return mesh;

    const float halfW = static_cast<float>(atlas.height.width()) * 0.5f;
    const float halfD = static_cast<float>(atlas.height.depth()) * 0.5f;

    // The ground under the whole FOOTPRINT, not under this column. The difference is
    // visible from across the map: a plot on a slope with each column at its own ground
    // level shears into a staircase and, along a ridge, into a long wall standing free of
    // the hillside.
    appendVoxelFaces(
        mesh, volume, -halfW, -halfD,
        [&](core::u32 x, core::u32 z) {
            const core::usize index = static_cast<core::usize>(z) * volume.width + x;
            return datum != nullptr && index < datumCount ?
                       datum[index] :
                       atlas.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
        },
        baseLift, palette, paletteSize);
    return mesh;
}

/**
 * @brief Meshes the layered cave system: every floor, and the shafts joining them.
 *
 * A flat underground can be drawn as one plan because it IS one plan. A system is a
 * stack, so what has to read is the *vertical* relationship — which layer sits under
 * which, where a shaft drops from one to the next, and which shafts come out on the
 * surface. Entrances are amber for that reason: an entrance is the difference
 * between a cave and a sealed void, and it is the one property a flat generator
 * could not express at all.
 */
[[nodiscard]] inline MapMesh buildCaveSystemMesh(const WorldAtlas &atlas, float topDepth, float layerSpacing)
{
    MapMesh mesh;
    const CaveSystem &system = atlas.caveSystem;
    if (system.layerCount == 0u || atlas.height.empty())
        return mesh;

    const float halfW = static_cast<float>(atlas.height.width()) * 0.5f;
    const float halfD = static_cast<float>(atlas.height.depth()) * 0.5f;

    for (core::u32 layer = 0u; layer < system.layerCount; ++layer)
    {
        const DungeonMap &plan = system.layer[layer];
        if (plan.empty())
            continue;

        // Deeper layers darken. Reading depth off a colour is crude and it works:
        // three identically lit floors stacked in a wireframe are unreadable.
        const float shade = 1.0f - 0.22f * static_cast<float>(layer);
        const Rgb floorColour{0.52f * shade, 0.44f * shade, 0.36f * shade};
        const Rgb wallColour{0.30f * shade, 0.26f * shade, 0.24f * shade};

        const auto rock = [&plan](core::i32 x, core::i32 z) {
            if (!plan.contains(x, z))
                return true;
            return !isWalkable(plan.at(static_cast<core::u32>(x), static_cast<core::u32>(z)));
        };

        for (core::u32 z = 0u; z < plan.depth(); ++z)
            for (core::u32 x = 0u; x < plan.width(); ++x)
            {
                const core::i32 ix = static_cast<core::i32>(x);
                const core::i32 iz = static_cast<core::i32>(z);
                if (rock(ix, iz))
                    continue;

                const float ground = atlas.height.clamped(ix, iz).toFloat();
                const float floorY = ground - topDepth - static_cast<float>(layer) * layerSpacing;
                const float ceilY = floorY + 1.6f;
                const float x0 = static_cast<float>(x) - halfW;
                const float x1 = x0 + 1.0f;
                const float z0 = static_cast<float>(z) - halfD;
                const float z1 = z0 + 1.0f;

                appendQuad(mesh, x0, floorY, z0, x1, floorY, z0, x1, floorY, z1, x0, floorY, z1, 0.0f, 1.0f, 0.0f,
                           floorColour);
                if (rock(ix + 1, iz))
                    appendQuad(mesh, x1, floorY, z0, x1, ceilY, z0, x1, ceilY, z1, x1, floorY, z1, 1.0f, 0.0f, 0.0f,
                               wallColour);
                if (rock(ix - 1, iz))
                    appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x0, ceilY, z1, x0, floorY, z1, -1.0f, 0.0f, 0.0f,
                               wallColour);
                if (rock(ix, iz + 1))
                    appendQuad(mesh, x0, floorY, z1, x0, ceilY, z1, x1, ceilY, z1, x1, floorY, z1, 0.0f, 0.0f, 1.0f,
                               wallColour);
                if (rock(ix, iz - 1))
                    appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x1, ceilY, z0, x1, floorY, z0, 0.0f, 0.0f, -1.0f,
                               wallColour);
            }
    }

    // The shafts, as square columns joining the two floors they connect. A surface
    // shaft runs all the way up to the ground and is drawn in amber.
    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const CaveShaft &shaft = system.shafts[i];
        const float ground =
            atlas.height.clamped(static_cast<core::i32>(shaft.x), static_cast<core::i32>(shaft.z)).toFloat();
        const float upper =
            shaft.surface ? ground + 0.4f : ground - topDepth - static_cast<float>(shaft.upperLayer) * layerSpacing;
        const float lower =
            ground - topDepth - static_cast<float>(shaft.surface ? shaft.upperLayer : shaft.lowerLayer) * layerSpacing;
        const Rgb colour = shaft.surface ? Rgb{0.95f, 0.62f, 0.14f} : Rgb{0.40f, 0.34f, 0.30f};

        const float x0 = static_cast<float>(shaft.x) - halfW + 0.25f;
        const float x1 = x0 + 0.5f;
        const float z0 = static_cast<float>(shaft.z) - halfD + 0.25f;
        const float z1 = z0 + 0.5f;
        appendQuad(mesh, x0, lower, z0, x0, upper, z0, x1, upper, z0, x1, lower, z0, 0.0f, 0.0f, -1.0f, colour);
        appendQuad(mesh, x1, lower, z1, x1, upper, z1, x0, upper, z1, x0, lower, z1, 0.0f, 0.0f, 1.0f, colour);
        appendQuad(mesh, x1, lower, z0, x1, upper, z0, x1, upper, z1, x1, lower, z1, 1.0f, 0.0f, 0.0f, colour);
        appendQuad(mesh, x0, lower, z1, x0, upper, z1, x0, upper, z0, x0, lower, z0, -1.0f, 0.0f, 0.0f, colour);
    }
    return mesh;
}

/** @brief The palette a liminal zone is drawn in. */
[[nodiscard]] inline Rgb liminalZoneColour(LiminalZone zone) noexcept
{
    switch (zone)
    {
    case LiminalZone::Corridor: return {0.76f, 0.72f, 0.55f};
    case LiminalZone::Office: return {0.82f, 0.79f, 0.62f};
    case LiminalZone::Hall: return {0.70f, 0.68f, 0.58f};
    case LiminalZone::Pool: return {0.55f, 0.72f, 0.74f};
    case LiminalZone::Count: break;
    }
    return {0.7f, 0.7f, 0.7f};
}

/**
 * @brief Meshes a liminal sector as walls on a flat floor, tinted by zone.
 *
 * Flat on purpose: the whole effect depends on the ceiling being uniform and the
 * light being even, and a liminal space that follows terrain stops being one.
 */
[[nodiscard]] inline MapMesh buildLiminalMesh(const LiminalSpace &space)
{
    MapMesh mesh;
    if (space.map.empty())
        return mesh;

    const core::u32 width = space.map.width();
    const core::u32 depth = space.map.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;
    const float wallHeight = 2.6f;

    const auto rock = [&space](core::i32 x, core::i32 z) {
        if (!space.map.contains(x, z))
            return true;
        return !isWalkable(space.map.at(static_cast<core::u32>(x), static_cast<core::u32>(z)));
    };

    for (core::u32 z = 0u; z < depth; ++z)
        for (core::u32 x = 0u; x < width; ++x)
        {
            const core::i32 ix = static_cast<core::i32>(x);
            const core::i32 iz = static_cast<core::i32>(z);
            if (rock(ix, iz))
                continue;
            const Rgb floorColour = liminalZoneColour(space.zones.at(x, z));
            const Rgb wallColour{floorColour.r * 0.78f, floorColour.g * 0.78f, floorColour.b * 0.70f};

            const float x0 = static_cast<float>(x) - halfW;
            const float x1 = x0 + 1.0f;
            const float z0 = static_cast<float>(z) - halfD;
            const float z1 = z0 + 1.0f;

            appendQuad(mesh, x0, 0.0f, z0, x1, 0.0f, z0, x1, 0.0f, z1, x0, 0.0f, z1, 0.0f, 1.0f, 0.0f, floorColour);
            if (rock(ix + 1, iz))
                appendQuad(mesh, x1, 0.0f, z0, x1, wallHeight, z0, x1, wallHeight, z1, x1, 0.0f, z1, 1.0f, 0.0f, 0.0f,
                           wallColour);
            if (rock(ix - 1, iz))
                appendQuad(mesh, x0, 0.0f, z0, x0, wallHeight, z0, x0, wallHeight, z1, x0, 0.0f, z1, -1.0f, 0.0f, 0.0f,
                           wallColour);
            if (rock(ix, iz + 1))
                appendQuad(mesh, x0, 0.0f, z1, x0, wallHeight, z1, x1, wallHeight, z1, x1, 0.0f, z1, 0.0f, 0.0f, 1.0f,
                           wallColour);
            if (rock(ix, iz - 1))
                appendQuad(mesh, x0, 0.0f, z0, x0, wallHeight, z0, x1, wallHeight, z0, x1, 0.0f, z0, 0.0f, 0.0f, -1.0f,
                           wallColour);
        }
    return mesh;
}

/** @brief The canopy colour of a plant standing in @p biome. */
[[nodiscard]] inline Rgb canopyColour(BiomeId biome) noexcept
{
    switch (biome)
    {
    case BiomeId::Taiga: return {0.13f, 0.33f, 0.25f};
    case BiomeId::Rainforest: return {0.09f, 0.42f, 0.17f};
    case BiomeId::Savanna: return {0.50f, 0.47f, 0.21f};
    case BiomeId::Desert: return {0.30f, 0.50f, 0.28f};
    case BiomeId::Marsh: return {0.29f, 0.46f, 0.27f};
    default: break;
    }
    return {0.16f, 0.42f, 0.20f};
}

/**
 * @brief One plant: a crossed trunk and a four-sided canopy, tinted by its biome.
 *
 * The last mesher to leave the viewer, and the one that reads an ENTITY's position
 * rather than a grid — which is why it takes a world position and not a cell.
 *
 * Two crossed faces are enough trunk at this scale, and the canopy is a pyramid with
 * a distinct normal per face: sharing one normal makes it catch the light evenly and
 * read as a flat green blob instead of as volume.
 *
 * @param mesh  Destination.
 * @param atlas The world, for the biome the plant stands in.
 * @param cx    World X of the plant's centre.
 * @param cy    World Y (the ground it stands on).
 * @param cz    World Z.
 * @param half  Half-extent: the plant's size.
 */
inline void appendPlant(MapMesh &mesh, const WorldAtlas &atlas, float cx, float cy, float cz, float half)
{
    const float trunkHalf = half * 0.22f;
    const float trunkTop = cy + half * 1.1f;
    const float base = cy - half;

    // Tint from the biome it stands in, so a conifer stand and a jungle differ.
    Rgb canopy = canopyColour(BiomeId::Count);
    if (!atlas.biomes.empty() && !atlas.height.empty())
    {
        const core::i32 gx = static_cast<core::i32>(cx + static_cast<float>(atlas.height.width()) * 0.5f);
        const core::i32 gz = static_cast<core::i32>(cz + static_cast<float>(atlas.height.depth()) * 0.5f);
        if (atlas.biomes.contains(gx, gz))
            canopy = canopyColour(atlas.biomes.at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)));
    }

    const Rgb bark{0.33f, 0.23f, 0.15f};
    appendQuad(mesh, cx - trunkHalf, base, cz, cx - trunkHalf, trunkTop, cz, cx + trunkHalf, trunkTop, cz,
               cx + trunkHalf, base, cz, 0.0f, 0.0f, 1.0f, bark);
    appendQuad(mesh, cx, base, cz - trunkHalf, cx, trunkTop, cz - trunkHalf, cx, trunkTop, cz + trunkHalf, cx, base,
               cz + trunkHalf, 1.0f, 0.0f, 0.0f, bark);

    const float spread = half * 1.15f;
    const float apex = trunkTop + half * 2.1f;
    appendTriangle(mesh, cx, apex, cz, cx - spread, trunkTop, cz - spread, cx + spread, trunkTop, cz - spread, 0.0f,
                   0.45f, -0.89f, canopy);
    appendTriangle(mesh, cx, apex, cz, cx + spread, trunkTop, cz - spread, cx + spread, trunkTop, cz + spread, 0.89f,
                   0.45f, 0.0f, canopy);
    appendTriangle(mesh, cx, apex, cz, cx + spread, trunkTop, cz + spread, cx - spread, trunkTop, cz + spread, 0.0f,
                   0.45f, 0.89f, canopy);
    appendTriangle(mesh, cx, apex, cz, cx - spread, trunkTop, cz + spread, cx - spread, trunkTop, cz - spread, -0.89f,
                   0.45f, 0.0f, canopy);
}

/**
 * @brief FNV-1a fold of a mesh's geometry, for a test that must not need an eye.
 *
 * Folds the raw bit patterns of every float: a mesher that winds a face the other
 * way, drops a boundary quad or shifts a datum changes this, and that is the whole
 * point of moving these out of an app. Non-authoritative — a mesh is not state, so
 * this is a regression tripwire and never a cross-target gate.
 */
[[nodiscard]] inline core::u32 foldMapMesh(const MapMesh &mesh) noexcept
{
    core::u32 hash = 0x811C9DC5u;
    const auto fold = [&hash](float value) {
        // The bits, not a rounded rendering of them.
        core::u32 bits = 0u;
        static_assert(sizeof(bits) == sizeof(value), "a float is four bytes here");
        __builtin_memcpy(&bits, &value, sizeof(bits));
        hash = (hash ^ bits) * 0x01000193u;
    };
    for (core::usize i = 0u; i < mesh.size(); ++i)
    {
        const MapVertex &v = mesh[i];
        fold(v.x);
        fold(v.y);
        fold(v.z);
        fold(v.nx);
        fold(v.ny);
        fold(v.nz);
        fold(v.colour.r);
        fold(v.colour.g);
        fold(v.colour.b);
    }
    return hash;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_MAP_MESH_HPP
