/**
 * @file PropLibrary.hpp
 * @brief The plants and boulders of a world: grown once, scattered, drawn batched.
 *
 * A world's props are three separate problems that keep being solved together and
 * badly: GROWING a mesh (expensive, and the same for every copy), deciding WHERE
 * copies stand (cheap, and it must not be stored), and DRAWING them in an order
 * that does not thrash the rasterizer.
 *
 * This owns all three and keeps them apart. The meshes come from
 * procgen::growTree and render::revolveProfile — grown once, at world init. The
 * positions come from a hash of the cell, so nothing is stored per prop and two
 * chunks agree along their border without talking to each other. The drawing goes
 * through render::ScatterQueue, which batches by mesh and orders by distance.
 *
 * A composable object, not a base class: a world HAS a prop library the way it has
 * a camera.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_PROP_LIBRARY_HPP
#    define LPL_ENGINE_PROP_LIBRARY_HPP

#    include <lpl/procgen/Botany.hpp>
#    include <lpl/render/Foliage.hpp>
#    include <lpl/render/Revolve.hpp>
#    include <lpl/render/Scatter.hpp>
#    include <lpl/render/Topology.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::engine {

/**
 * @struct PropLibraryParams
 * @brief How dense the props are and how far they are worth drawing.
 */
struct PropLibraryParams {
    core::u32 treeSpecies{3u};
    core::u32 rockVariants{3u};
    core::u32 rockOneIn{64u};        ///< One cell in N carries a boulder.
    core::f32 viewDistance{70.0f};   ///< Furthest a plant is worth drawing.
    core::f32 nearDistance{5.0f};    ///< Nearer than this the walker is IN the plant.
    core::f32 canopyDistance{11.0f}; ///< Nearer than this, wood only, no leaves.
    core::f32 propDistance{45.0f};   ///< Furthest a boulder is worth its triangles.
    core::f32 fogDensity{0.010f};
    core::u32 rockAlbedo{0x00877F76u};
    core::u32 plantSalt{0x9E3779B9u};
    core::u32 rockSalt{0x85EBCA6Bu};
};

/**
 * @class PropLibrary
 * @brief Grown meshes, hashed placement, batched drawing.
 */
class PropLibrary {
public:
    /**
     * @brief Grows one plant per species and one profile per boulder variant.
     *
     * Called once. Growing a tree is an L-system expansion and a 3D turtle walk;
     * doing it per instance would cost more than drawing the whole world.
     */
    void build(const PropLibraryParams &params, core::u32 seed);

    [[nodiscard]] const render::FoliageMesh *trees() const noexcept { return _trees; }
    [[nodiscard]] core::u32 treeSpecies() const noexcept { return _params.treeSpecies; }

    /** @brief How tall and wide the plant on a cell is, for its shadow. */
    void plantExtent(core::i32 cellX, core::i32 cellZ, core::f32 &outHeight,
                     core::f32 &outSpread) const noexcept;

    /** @brief Whether a boulder stands on a cell, and which one. */
    [[nodiscard]] bool rockAt(core::i32 cellX, core::i32 cellZ, core::u32 &outVariant,
                              core::f32 &outScale) const noexcept;

    /** @brief Empties the frame's queue. Call once before queueing. */
    void beginFrame() const { _queue.clear(); }

    /** @brief Queues the plant that grows on a cell, if one does. */
    void queuePlant(core::i32 cellX, core::i32 cellZ, core::f32 ground, core::f32 light) const;

    /** @brief Draws the queued plants, batched by species and ordered by distance. */
    core::u32 flushPlants(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                          const render::CameraBasis &basis, core::u32 haze) const;

    /**
     * @brief Draws the boulders of one patch of cells.
     *
     * Cells aligned to FOUR, not offset by two: terrain drawn at stride 2 or 4 only
     * has its sampled height on quad CORNERS. A prop on any other cell sits at the
     * height of a surface that was not drawn, and hangs above or sinks into the one
     * that was — which is exactly how the boulders came out floating.
     *
     * @param heightAt (localX, localZ) -> f32 for a cell of this patch.
     */
    template <typename HeightAt>
    core::u32 drawRocks(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                        const render::CameraBasis &basis, core::i32 originX, core::i32 originZ, core::u32 size,
                        core::f32 seaLevel, const math::Vec3<core::f32> &sunDirection, core::f32 ambient,
                        HeightAt &&heightAt) const
    {
        core::u32 triangles = 0u;
        for (core::u32 z = 0u; z < size; z += 4u)
            for (core::u32 x = 0u; x < size; x += 4u)
            {
                const core::i32 cellX = originX + static_cast<core::i32>(x);
                const core::i32 cellZ = originZ + static_cast<core::i32>(z);
                core::u32 variant = 0u;
                core::f32 scale = 1.0f;
                if (!rockAt(cellX, cellZ, variant, scale))
                    continue;
                const core::f32 ground = heightAt(x, z);
                if (ground < seaLevel)
                    continue;
                const core::f32 worldX = static_cast<core::f32>(cellX) + 0.5f;
                const core::f32 worldZ = static_cast<core::f32>(cellZ) + 0.5f;
                if (render::approximateLength(worldX - basis.eye.x, worldZ - basis.eye.z) > _params.propDistance)
                    continue;
                triangles += render::drawRevolved(rt, mvp, _rocks[variant], worldX, ground, worldZ, scale,
                                                  _params.rockAlbedo, sunDirection, ambient);
            }
        return triangles;
    }

    [[nodiscard]] const PropLibraryParams &params() const noexcept { return _params; }

    /** @brief Packets the last flush submitted, and the fold of that stream. */
    [[nodiscard]] core::u32 submittedDraws() const noexcept { return _queue.submittedDraws(); }
    [[nodiscard]] core::u32 latchedFold() const noexcept { return _queue.latchedFold(); }

private:
    static constexpr core::u32 kMaxSpecies = 4u;
    static constexpr core::u32 kMaxVariants = 4u;

    [[nodiscard]] static core::f32 scaleFromHash(core::u32 hash) noexcept
    {
        return 0.75f + static_cast<core::f32>((hash >> 8) & 0x3Fu) * (0.85f / 63.0f);
    }

    /**
     * @brief Boulders: a Catmull profile swept around the axis.
     *
     * render::Topology carried a Catmull-Rom tessellator that only ever produced a
     * signature — the geometry was computed and thrown away. Half of a closed loop is
     * exactly a rock's silhouette, widest near the BASE: widest in the middle is a
     * mushroom, which is what the first profile produced and what the screenshot
     * showed.
     */
    void buildRocks(core::u32 seed);

    PropLibraryParams _params{};
    lpl::pmr::vector<render::FoliageSegment> _segments[kMaxSpecies];
    lpl::pmr::vector<render::FoliageSprite> _sprites[kMaxSpecies];
    render::FoliageMesh _trees[kMaxSpecies]{};
    render::RevolvedMesh _rocks[kMaxVariants]{};
    mutable render::ScatterQueue _queue;
};

} // namespace lpl::engine

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/engine/PropLibrary.inl>

#endif // LPL_ENGINE_PROP_LIBRARY_HPP
