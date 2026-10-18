/**
 * @file CubePile.hpp
 * @brief Cube-pile sample simulation — a deterministic crowd of cubes that fall
 *        and collide into a settling pile. One reference @c samples sim that the
 *        generic application runtime can drive on any host (kernel HAL or Linux).
 *
 * Reworked to run entirely on the engine modules instead of a hand-rolled
 * loop: entities live in a real @c ecs::Registry with authoritative Fixed32
 * @c Position / @c Velocity / @c AABB / @c Mass components, and the simulation
 * is advanced by @c physics::CpuPhysicsBackend (semi-implicit Euler + damping +
 * ground bounce + AABB collision with octree broad-phase and sleeping). The
 * authoritative state is Fixed32, so both the state fold and the float
 * projection's image fold are bit-identical across the Linux oracle and the
 * i686 kernel — the cross-target signature the engine test `cube_pile`
 * records. This is the reconciliation of the two worlds: the reference sim is
 * now expressed as ECS entities stepped by engine systems, exactly like a
 * loaded scene.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_CUBEPILE_HPP
#    define LPL_SAMPLES_CUBEPILE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Component.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/physics/CpuPhysicsBackend.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::samples {

/**
 * A deterministic crowd of cubes that fall and collide into a settling pile,
 * backed by the engine ECS + physics modules. All authoritative state is
 * Fixed32 in the registry (bit-identical kernel<->oracle).
 */
struct CubePile {
    using Fixed32 = math::Fixed32;
    using FVec3 = math::Vec3<Fixed32>;

    static constexpr core::u32 kNx = 16u;                /**< Lattice columns (X). */
    static constexpr core::u32 kNy = 4u;                 /**< Lattice layers (Y). */
    static constexpr core::u32 kNz = 16u;                /**< Lattice rows (Z). */
    static constexpr core::u32 kCount = kNx * kNy * kNz; /**< 1024 entities. */

    static constexpr core::f32 kHalfF = 0.18f;    /**< Cube half-extent (render + AABB). */
    static constexpr core::f32 kSpacingF = 0.46f; /**< Initial lattice spacing. */
    static constexpr core::f32 kSpawnYF = 3.0f;   /**< First layer height above ground. */
    /** The backend clamps to y >= 0.5 (kDefaultHalfHeight); draw the grid there. */
    static constexpr core::f32 kFloorF = 0.5f - kHalfF;
    static constexpr core::f32 kDtF = 1.0f / 60.0f; /**< Fixed, deterministic timestep. */

    ecs::Registry &registry;            /**< The world's entity registry (borrowed). */
    physics::CpuPhysicsBackend backend; /**< Steps the authoritative Fixed32 state. */
    core::u32 tints[kCount]{};          /**< Per-entity face tint (render only). */

    /**
     * @brief Borrows the registry the cubes will live in.
     *
     * @param reg The registry the cubes live in — the hosting World's own
     *        registry on the live path, a throwaway local one in the parity
     *        fold. CubePile creates its entities in it and never owns it.
     */
    explicit CubePile(ecs::Registry &reg) : registry(reg), backend(reg) {}

    /**
     * @brief Seed a kNx*kNy*kNz lattice above the floor (no RNG): each entity gets a
     * tiny index-derived velocity jitter so the fall is not perfectly uniform.
     *
     * The chunk buffers are seeded in creation order: Position, Velocity and AABB go into the
     * write (back) buffer the integrator mutates, Mass into both buffers. The AABB is written as
     * the full cube size: the backend halves it internally.
     */
    void init() noexcept;

    /**
     * @brief One deterministic tick through the engine physics backend.
     */
    void step() noexcept { (void) backend.step(kDtF); }

    /**
     * @brief FNV-1a fold of every entity's authoritative Fixed32 state (position +
     * velocity), read from the integrator's write buffer, in creation order.
     */
    [[nodiscard]] core::u32 stateSignature() const noexcept;

    /**
     * @brief Number of entities in the simulation (for host HUDs).
     */
    [[nodiscard]] static constexpr core::u32 count() noexcept { return kCount; }

    /**
     * Non-authoritative orbit camera: yaw/pitch around a target at @c dist. When
     * @c possess is a valid entity index the camera follows that entity, else it
     * orbits the pile centre. Angles are non-authoritative (render only).
     */
    struct Camera {
        core::f32 yaw{0.6f};
        core::f32 pitch{0.35f};
        core::f32 dist{12.0f};
        core::i32 possess{-1};
    };

    /**
     * @brief Rasterize the scene: a ground grid, then every entity as a depth-buffered,
     * Lambert-lit cube, viewed through the orbit @p cam. Positions are read from
     * the authoritative Fixed32 components and projected in float (render only).
     *
     * The camera basis comes from CORDIC: no libm, deterministic. The ground grid lies on the
     * floor plane, 17 lines per axis 0.6 world units apart, drawn without depth. Each face is lit
     * by its constant outward normal (the cubes are axis-aligned), indexed by triangle / 2,
     * against a normalised directional light plus an ambient term (Lambert diffuse).
     */
    void render(const render::RenderTarget &rt, const Camera &cam) const noexcept;

private:
    /**
     * @brief Bresenham line into the colour buffer (no depth) — used for the grid.
     */
    static void plotLine(const render::RenderTarget &rt, const render::detail::ScreenVertex &a,
                         const render::detail::ScreenVertex &b, core::u32 color) noexcept;
};

} // namespace lpl::samples

#    include "CubePile.inl"

#endif // LPL_SAMPLES_CUBEPILE_HPP
