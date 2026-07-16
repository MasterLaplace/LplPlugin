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
 * i686 kernel — the cross-target signature checked by tests/parity. This is the
 * reconciliation of the two worlds: the reference sim is now expressed as ECS
 * entities stepped by engine systems, exactly like a loaded scene.
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

using namespace lpl::render; // RenderTarget, clearTarget, foldTarget, detail::*, perspectiveFov

/// A deterministic crowd of cubes that fall and collide into a settling pile,
/// backed by the engine ECS + physics modules. All authoritative state is
/// Fixed32 in the registry (bit-identical kernel<->oracle).
struct CubePile {
    using Fixed32 = math::Fixed32;
    using FVec3 = math::Vec3<Fixed32>;

    static constexpr core::u32 kNx = 16u;                ///< Lattice columns (X).
    static constexpr core::u32 kNy = 4u;                 ///< Lattice layers (Y).
    static constexpr core::u32 kNz = 16u;                ///< Lattice rows (Z).
    static constexpr core::u32 kCount = kNx * kNy * kNz; ///< 1024 entities.

    static constexpr core::f32 kHalfF = 0.18f;    ///< Cube half-extent (render + AABB).
    static constexpr core::f32 kSpacingF = 0.46f; ///< Initial lattice spacing.
    static constexpr core::f32 kSpawnYF = 3.0f;   ///< First layer height above ground.
    /// The backend clamps to y >= 0.5 (kDefaultHalfHeight); draw the grid there.
    static constexpr core::f32 kFloorF = 0.5f - kHalfF;
    static constexpr core::f32 kDtF = 1.0f / 60.0f; ///< Fixed, deterministic timestep.

    ecs::Registry registry;                 ///< Owns the entities + component chunks.
    physics::CpuPhysicsBackend backend;     ///< Steps the authoritative Fixed32 state.
    core::u32 tints[kCount]{};              ///< Per-entity face tint (render only).

    CubePile() : backend(registry) {}

    /// Seed a kNx*kNy*kNz lattice above the floor (no RNG): each entity gets a
    /// tiny index-derived velocity jitter so the fall is not perfectly uniform.
    void init() noexcept
    {
        using F = Fixed32;
        const ecs::ComponentId ids[] = {ecs::ComponentId::Position, ecs::ComponentId::Velocity,
                                        ecs::ComponentId::AABB, ecs::ComponentId::Mass};
        const ecs::Archetype archetype{ids};
        for (core::u32 i = 0; i < kCount; ++i)
            (void) registry.createEntity(archetype);
        (void) backend.init();

        static const core::u32 hues[6] = {0x00FF6060u, 0x0060FF60u, 0x006060FFu,
                                          0x00FFFF60u, 0x00FF60FFu, 0x0060FFFFu};
        const core::f32 ox = static_cast<core::f32>(kNx - 1u) * 0.5f;
        const core::f32 oz = static_cast<core::f32>(kNz - 1u) * 0.5f;
        const F cubeSize = F::fromFloat(2.0f * kHalfF); // backend halves the AABB internally
        const F one = F::one();

        // Seed chunk buffers in creation order: Position/Velocity/AABB go into the
        // write (back) buffer the integrator mutates; Mass into the read buffer.
        core::u32 gi = 0;
        for (const auto &partition : registry.partitions())
        {
            for (const auto &chunk : partition->chunks())
            {
                const core::u32 count = chunk->count();
                auto *pos = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
                auto *vel = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
                auto *aabb = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB));
                auto *massW = static_cast<Fixed32 *>(chunk->writeComponent(ecs::ComponentId::Mass));
                auto *massR = static_cast<Fixed32 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Mass)));
                if (!pos || !vel)
                    continue;
                for (core::u32 li = 0; li < count; ++li, ++gi)
                {
                    const core::u32 ix = gi % kNx;
                    const core::u32 iz = (gi / kNx) % kNz;
                    const core::u32 iy = gi / (kNx * kNz);
                    pos[li] = {F::fromFloat((static_cast<core::f32>(ix) - ox) * kSpacingF),
                               F::fromFloat(kSpawnYF + static_cast<core::f32>(iy) * kSpacingF),
                               F::fromFloat((static_cast<core::f32>(iz) - oz) * kSpacingF)};
                    vel[li] = {F::fromFloat((static_cast<core::f32>((gi * 13u) % 7u) - 3.0f) * 0.02f), F::zero(),
                               F::fromFloat((static_cast<core::f32>((gi * 17u) % 7u) - 3.0f) * 0.02f)};
                    if (aabb)
                        aabb[li] = {cubeSize, cubeSize, cubeSize};
                    if (massW)
                        massW[li] = one;
                    if (massR)
                        massR[li] = one;
                    tints[gi] = hues[gi % 6u];
                }
            }
        }
    }

    /// One deterministic tick through the engine physics backend.
    void step() noexcept { (void) backend.step(kDtF); }

    /// FNV-1a fold of every entity's authoritative Fixed32 state (position +
    /// velocity), read from the integrator's write buffer, in creation order.
    [[nodiscard]] core::u32 stateSignature() const noexcept
    {
        core::u32 hash = detail::kFnv1aOffsetBasis;
        for (const auto &partition : registry.partitions())
        {
            for (const auto &chunk : partition->chunks())
            {
                const core::u32 count = chunk->count();
                const auto *pos = static_cast<const FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
                const auto *vel = static_cast<const FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
                if (!pos || !vel)
                    continue;
                for (core::u32 li = 0; li < count; ++li)
                {
                    hash = detail::fnv1aStep(hash, static_cast<core::u32>(pos[li].x.raw()));
                    hash = detail::fnv1aStep(hash, static_cast<core::u32>(pos[li].y.raw()));
                    hash = detail::fnv1aStep(hash, static_cast<core::u32>(pos[li].z.raw()));
                    hash = detail::fnv1aStep(hash, static_cast<core::u32>(vel[li].x.raw()));
                    hash = detail::fnv1aStep(hash, static_cast<core::u32>(vel[li].y.raw()));
                    hash = detail::fnv1aStep(hash, static_cast<core::u32>(vel[li].z.raw()));
                }
            }
        }
        return hash;
    }

    /// Number of entities in the simulation (for host HUDs).
    [[nodiscard]] static constexpr core::u32 count() noexcept { return kCount; }

    /// Non-authoritative orbit camera: yaw/pitch around a target at @c dist. When
    /// @c possess is a valid entity index the camera follows that entity, else it
    /// orbits the pile centre. Angles are non-authoritative (render only).
    struct Camera {
        core::f32 yaw{0.6f};
        core::f32 pitch{0.35f};
        core::f32 dist{12.0f};
        core::i32 possess{-1};
    };

    /// Rasterize the scene: a ground grid, then every entity as a depth-buffered,
    /// Lambert-lit cube, viewed through the orbit @p cam. Positions are read from
    /// the authoritative Fixed32 components and projected in float (render only).
    void render(const RenderTarget &rt, const Camera &cam) const noexcept
    {
        using F = Fixed32;
        using Vec3f = math::Vec3<core::f32>;

        clearTarget(rt, 0x00141828u);

        // Gather authoritative positions (Fixed32 → float, render only) in order.
        static core::f32 px[kCount], py[kCount], pz[kCount];
        core::u32 gi = 0;
        for (const auto &partition : registry.partitions())
        {
            for (const auto &chunk : partition->chunks())
            {
                const core::u32 count = chunk->count();
                const auto *pos = static_cast<const FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
                if (!pos)
                    continue;
                for (core::u32 li = 0; li < count && gi < kCount; ++li, ++gi)
                {
                    px[gi] = pos[li].x.toFloat();
                    py[gi] = pos[li].y.toFloat();
                    pz[gi] = pos[li].z.toFloat();
                }
            }
        }
        const core::u32 total = gi;

        // ── Camera basis via CORDIC (no libm; deterministic) ─────────────────
        F sy{F::zero()}, cy{F::zero()}, sp{F::zero()}, cp{F::zero()};
        math::Cordic::sincos(F::fromFloat(cam.yaw), sy, cy);
        math::Cordic::sincos(F::fromFloat(cam.pitch), sp, cp);
        core::f32 tx = 0.0f, ty = kFloorF + 0.6f, tz = 0.0f;
        if (cam.possess >= 0 && static_cast<core::u32>(cam.possess) < total)
        {
            tx = px[cam.possess];
            ty = py[cam.possess];
            tz = pz[cam.possess];
        }
        const core::f32 dirx = cp.toFloat() * sy.toFloat();
        const core::f32 diry = sp.toFloat();
        const core::f32 dirz = cp.toFloat() * cy.toFloat();
        const Vec3f eye(tx + cam.dist * dirx, ty + cam.dist * diry, tz + cam.dist * dirz);
        const auto view = math::Mat4<core::f32>::lookAt(eye, Vec3f(tx, ty, tz), Vec3f(0.0f, 1.0f, 0.0f));
        const core::f32 aspect = static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height);
        const auto proj = perspectiveFov(F::fromFloat(1.04719755f), aspect, 0.1f, 100.0f);
        const auto mvp = proj * view;

        // ── Ground grid on the floor plane (drawn first, no depth) ───────────
        {
            constexpr core::i32 kLines = 17;  // lines per axis
            constexpr core::f32 kStep = 0.6f; // world units between lines
            const core::f32 half = static_cast<core::f32>(kLines - 1) * 0.5f * kStep;
            for (core::i32 g = 0; g < kLines; ++g)
            {
                const core::f32 t = static_cast<core::f32>(g) * kStep - half;
                const detail::ScreenVertex ax = detail::projectVertex(mvp, -half, kFloorF, t, rt.width, rt.height);
                const detail::ScreenVertex bx = detail::projectVertex(mvp, half, kFloorF, t, rt.width, rt.height);
                plotLine(rt, ax, bx, 0x00203048u);
                const detail::ScreenVertex az = detail::projectVertex(mvp, t, kFloorF, -half, rt.width, rt.height);
                const detail::ScreenVertex bz = detail::projectVertex(mvp, t, kFloorF, half, rt.width, rt.height);
                plotLine(rt, az, bz, 0x00203048u);
            }
        }

        static const core::f32 corners[8][3] = {
            {-1.0f, -1.0f, -1.0f},
            {1.0f,  -1.0f, -1.0f},
            {1.0f,  1.0f,  -1.0f},
            {-1.0f, 1.0f,  -1.0f},
            {-1.0f, -1.0f, 1.0f },
            {1.0f,  -1.0f, 1.0f },
            {1.0f,  1.0f,  1.0f },
            {-1.0f, 1.0f,  1.0f },
        };
        static const core::u32 indices[36] = {
            0, 1, 2, 0, 2, 3, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 1, 5, 6, 1, 6, 2, 4, 5, 1, 4, 1, 0, 3, 2, 6, 3, 6, 7,
        };
        // Constant outward face normals (axis-aligned cubes), indexed by t/2.
        static const core::f32 faceN[6][3] = {
            {0.0f,  0.0f,  -1.0f},
            {0.0f,  0.0f,  1.0f },
            {-1.0f, 0.0f,  0.0f },
            {1.0f,  0.0f,  0.0f },
            {0.0f,  -1.0f, 0.0f },
            {0.0f,  1.0f,  0.0f },
        };
        // Normalised directional light + ambient term (Lambert diffuse).
        constexpr core::f32 kLx = 0.384f, kLy = 0.816f, kLz = 0.432f;
        constexpr core::f32 kAmbient = 0.30f, kDiffuse = 0.70f;

        for (core::u32 ei = 0; ei < total; ++ei)
        {
            const bool possessed = (cam.possess >= 0 && static_cast<core::u32>(cam.possess) == ei);
            detail::ScreenVertex sv[8];
            for (core::u32 i = 0; i < 8u; ++i)
            {
                const core::f32 lx = corners[i][0] * kHalfF;
                const core::f32 ly = corners[i][1] * kHalfF;
                const core::f32 lz = corners[i][2] * kHalfF;
                sv[i] = detail::projectVertex(mvp, px[ei] + lx, py[ei] + ly, pz[ei] + lz, rt.width, rt.height);
            }
            const core::u32 tint = possessed ? 0x0020FF40u : tints[ei];
            for (core::u32 t = 0; t < 12u; ++t)
            {
                const core::u32 f = t / 2u;
                core::f32 ndl = faceN[f][0] * kLx + faceN[f][1] * kLy + faceN[f][2] * kLz;
                if (ndl < 0.0f)
                    ndl = 0.0f;
                const core::f32 lit = kAmbient + kDiffuse * ndl;
                const core::u32 shade = static_cast<core::u32>(lit * 256.0f);
                const core::u32 r = (((tint >> 16) & 0xFFu) * shade) >> 8;
                const core::u32 g = (((tint >> 8) & 0xFFu) * shade) >> 8;
                const core::u32 b = ((tint & 0xFFu) * shade) >> 8;
                const core::u32 color = (r << 16) | (g << 8) | b;
                detail::fillTriangle(rt, sv[indices[t * 3 + 0]], sv[indices[t * 3 + 1]], sv[indices[t * 3 + 2]], color);
            }
        }
        static_cast<void>(kLz);
    }

private:
    /// Bresenham line into the colour buffer (no depth) — used for the grid.
    static void plotLine(const RenderTarget &rt, const detail::ScreenVertex &a, const detail::ScreenVertex &b,
                         core::u32 color) noexcept
    {
        if (!a.valid || !b.valid)
            return;
        core::i32 x0 = static_cast<core::i32>(a.x), y0 = static_cast<core::i32>(a.y);
        const core::i32 x1 = static_cast<core::i32>(b.x), y1 = static_cast<core::i32>(b.y);
        core::i32 dx = x1 - x0, dy = y1 - y0;
        dx = dx < 0 ? -dx : dx;
        dy = dy < 0 ? -dy : dy;
        const core::i32 sx = x0 < x1 ? 1 : -1;
        const core::i32 sy = y0 < y1 ? 1 : -1;
        core::i32 err = dx - dy;
        const core::i32 W = static_cast<core::i32>(rt.width), H = static_cast<core::i32>(rt.height);
        for (core::u32 guard = 0; guard < 4096u; ++guard)
        {
            if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H)
                rt.color[static_cast<core::u32>(y0) * rt.width + static_cast<core::u32>(x0)] = color;
            if (x0 == x1 && y0 == y1)
                break;
            const core::i32 e2 = err * 2;
            if (e2 > -dy)
            {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }
};

/// Result of running the demo for @p ticks and rendering frame @p ticks: the
/// authoritative state signature plus the folded rendered image signature.
struct SimFoldResult {
    core::u32 state_signature{0u};
    core::u32 image_signature{0u};
};

/// Convenience used by both the oracle parity test and the in-kernel smoke:
/// seed, advance @p ticks deterministic steps, render into @p rt, fold both.
[[nodiscard]] inline SimFoldResult runCubePileAndFold(const RenderTarget &rt, core::u32 ticks) noexcept
{
    // A fresh registry per call keeps each run deterministic and independent.
    CubePile scene;
    scene.init();
    for (core::u32 t = 0; t < ticks; ++t)
        scene.step();
    scene.render(rt, CubePile::Camera{});
    return SimFoldResult{scene.stateSignature(), foldTarget(rt)};
}

} // namespace lpl::samples

#endif // LPL_SAMPLES_CUBEPILE_HPP
