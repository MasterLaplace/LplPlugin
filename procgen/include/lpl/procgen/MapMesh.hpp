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

/**
 * A triangle list. Two triangles per quad, no index buffer: these are rebuilt
 * per regeneration, not streamed.
 */
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
                       float cz, float dx, float dy, float dz, float nx, float ny, float nz, const Rgb &colour);

/**
 * @brief Appends one triangle with a flat normal. The @ref appendQuad of odd faces.
 */
inline void appendTriangle(MapMesh &mesh, float ax, float ay, float az, float bx, float by, float bz, float cx,
                           float cy, float cz, float nx, float ny, float nz, const Rgb &colour);

/**
 * @struct MapSurfaceStyle
 * @brief What the surface shows on top of its shading.
 *
 * The three overlays are drawn INTO the surface rather than over it, because each
 * is a property of the ground and a separate overlay would z-fight with it.
 */
struct MapSurfaceStyle {
    MapShading shading{MapShading::Biome}; /**< Which quantity colours a cell. */
    core::u32 climateAxis{0u};             /**< Which axis, when shading by climate. */
    bool rivers{true};                     /**< Paint river cells blue. */
    bool settlement{true};                 /**< Paint roads, plazas and plots. */
    bool roads{true};                      /**< Paint the long-distance network, last. */
};

/**
 * @brief The terrain surface, one quad per cell, centred on the origin.
 *
 * Per-cell normals from the central difference, which is what makes relief read at
 * all: flat shading a heightfield is very nearly invisible.
 *
 * The highway network is painted on top of the town's own streets: it is the
 * long-distance layer, grown by the grammar and steered by the field, where the
 * settlement's roads are district borders. Drawn last so a road crossing a town reads
 * as passing through it.
 */
[[nodiscard]] inline MapMesh buildSurfaceMesh(const WorldAtlas &atlas, const MapSurfaceStyle &style);

/**
 * @brief The flat underground as a real volume: a floor, and walls around it.
 *
 * A sheet of coloured quads is not a cave. What makes a dungeon legible is the
 * WALLS — the boundary between the carved space and the rock — because that is the
 * only thing that shows a corridor as a corridor and a chamber as a chamber. Each
 * open cell contributes a floor quad, and each of its solid neighbours a vertical
 * face: the mesh is the surface of the void, not a picture of it.
 *
 * Outside the map counts as rock, so the cave is walled in rather than open to
 * nothing at the border. The cave hangs a fixed distance below the ground above it,
 * so it follows the terrain instead of lying on a flat plane the surface never
 * touches. The top of the walls is capped, so from above the plan reads as a plan.
 *
 * @param atlas      The world; its heightfield puts the cave under the ground.
 * @param depthBelow World units between the ground and the cave floor.
 */
[[nodiscard]] inline MapMesh buildDungeonMesh(const WorldAtlas &atlas, float depthBelow);

/**
 * @brief Raises the settlement: one box per footprint, sunk to its lowest ground.
 *
 * Painting plots onto the ground says where a town is. It does not put a town
 * there — a settlement is read by its silhouette, and a silhouette needs height.
 * Each footprint becomes one box whose height comes from its own area and district,
 * hashed, so a quarter has a character rather than a uniform skyline: a big plot gets
 * a tall building, and the variation is reproducible.
 *
 * The base is sunk to the lowest ground the footprint covers: a box placed at the
 * centre height would hang off the downhill corner.
 */
[[nodiscard]] inline MapMesh buildTownMesh(const WorldAtlas &atlas);

/**
 * @brief Meshes a voxel volume into world coordinates, exposed faces only.
 *
 * The core of it, with no opinion about where the volume SITS. Extracted because a
 * streamed village needs exactly this walk and cannot supply a @ref WorldAtlas — an atlas
 * is a bounded map's product, and an endless world has no such thing. Writing the walk a
 * second time would have been the sixth copy of "emit a quad per exposed face" in this
 * repository, and every copy is a chance to wind one the wrong way round.
 *
 * The walk itself is @ref forEachVoxelFace's, and this is now only its sink: place
 * the face in world units and give it a colour. A second consumer arrived that has
 * no vertex buffer to append to, and rewriting the walk for it would have been the
 * sixth copy of the six faces — with the winding to get right a sixth time.
 *
 * Outside the array is EMPTY for a building: it stands in the air, so its outer
 * faces are the ones a viewer sees. A cave answers the opposite way.
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
                      float baseLift, const Rgb *palette, core::u32 paletteSize);

/**
 * @brief Meshes a voxel volume, emitting only the faces that border empty space.
 *
 * The generic mesher the grammar's products need: the town raised by @ref buildTown
 * and the fences and lamps @ref decoratePath leaves along the roads are both a
 * @ref VoxelVolume, so both arrive here.
 *
 * The ground is the one under the whole FOOTPRINT, not under each column. The
 * difference is visible from across the map: a plot on a slope with each column at
 * its own ground level shears into a staircase and, along a ridge, into a long wall
 * standing free of the hillside.
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
                                            core::usize datumCount = 0u);

/**
 * @brief Meshes the layered cave system: every floor, and the shafts joining them.
 *
 * A flat underground can be drawn as one plan because it IS one plan. A system is a
 * stack, so what has to read is the *vertical* relationship — which layer sits under
 * which, where a shaft drops from one to the next, and which shafts come out on the
 * surface. Entrances are amber for that reason: an entrance is the difference
 * between a cave and a sealed void, and it is the one property a flat generator
 * could not express at all.
 *
 * Deeper layers darken. Reading depth off a colour is crude and it works: three
 * identically lit floors stacked in a wireframe are unreadable. The shafts are
 * square columns joining the two floors they connect; a surface shaft runs all the
 * way up to the ground and is drawn in amber.
 */
[[nodiscard]] inline MapMesh buildCaveSystemMesh(const WorldAtlas &atlas, float topDepth, float layerSpacing);

/**
 * @brief The palette a liminal zone is drawn in.
 */
[[nodiscard]] inline Rgb liminalZoneColour(LiminalZone zone) noexcept;

/**
 * @brief Meshes a liminal sector as walls on a flat floor, tinted by zone.
 *
 * Flat on purpose: the whole effect depends on the ceiling being uniform and the
 * light being even, and a liminal space that follows terrain stops being one.
 */
[[nodiscard]] inline MapMesh buildLiminalMesh(const LiminalSpace &space);

/**
 * @brief The canopy colour of a plant standing in @p biome.
 */
[[nodiscard]] inline Rgb canopyColour(BiomeId biome) noexcept;

/**
 * @brief One plant: a crossed trunk and a four-sided canopy, tinted by its biome.
 *
 * The last mesher to leave the viewer, and the one that reads an ENTITY's position
 * rather than a grid — which is why it takes a world position and not a cell.
 *
 * Two crossed faces are enough trunk at this scale, and the canopy is a pyramid with
 * a distinct normal per face: sharing one normal makes it catch the light evenly and
 * read as a flat green blob instead of as volume. The canopy takes its tint from the
 * biome the plant stands in, so a conifer stand and a jungle differ.
 *
 * @param mesh  Destination.
 * @param atlas The world, for the biome the plant stands in.
 * @param cx    World X of the plant's centre.
 * @param cy    World Y (the ground it stands on).
 * @param cz    World Z.
 * @param half  Half-extent: the plant's size.
 */
inline void appendPlant(MapMesh &mesh, const WorldAtlas &atlas, float cx, float cy, float cz, float half);

/**
 * @brief FNV-1a fold of a mesh's geometry, for a test that must not need an eye.
 *
 * Folds the raw bit patterns of every float: a mesher that winds a face the other
 * way, drops a boundary quad or shifts a datum changes this, and that is the whole
 * point of moving these out of an app. Non-authoritative — a mesh is not state, so
 * this is a regression tripwire and never a cross-target gate.
 */
[[nodiscard]] inline core::u32 foldMapMesh(const MapMesh &mesh) noexcept;

} // namespace lpl::procgen

#    include "MapMesh.inl"

#endif // LPL_PROCGEN_MAP_MESH_HPP
