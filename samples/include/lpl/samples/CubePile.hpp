/**
 * @file CubePile.hpp
 * @brief Cube-pile sample simulation — a deterministic crowd of cubes that fall
 *        and collide into a settling pile. One reference @c samples sim that the
 *        generic application runtime can drive on any host (kernel HAL or Linux).
 *
 * A swappable simulation, independent of the kernel: N entities carry Fixed32
 * position/velocity (authoritative, deterministic) advanced by gravity + floor
 * bounce + uniform-AABB inter-entity collision (uniform spatial-hash grid
 * broad-phase, no sqrt), and each is rasterized as a cube through the shared
 * software rasterizer. Authoritative state is Fixed32 (bit-identical across host
 * and kernel); the float projection/raster is non-authoritative and its folded
 * image signature is bit-identical too. Both folds are the cross-target
 * signature checked by tests/parity against the in-kernel run.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-29
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_CUBEPILE_HPP
#    define LPL_SAMPLES_CUBEPILE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/render/Projection.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/render/SoftwareRasterizer.hpp>

namespace lpl::samples {

using namespace lpl::render; // RenderTarget, clearTarget, foldTarget, detail::*, perspectiveFov

/// One authoritative entity: Fixed32 position/velocity (deterministic) + a
/// non-authoritative float scale and packed face-tint used only for drawing.
struct CubeEntity {
    math::Fixed32 x, y, z;   ///< World position (authoritative).
    math::Fixed32 vx, vy, vz; ///< Linear velocity (authoritative).
    core::f32 scale{0.4f};   ///< Cube half-extent (render only; == kHalf).
    core::u32 tint{0x00FFFFFFu}; ///< Multiplicative face tint (render only).
};

/// A deterministic crowd of cubes that fall and collide into a settling pile.
/// All authoritative state is Fixed32 (bit-identical kernel<->oracle). Collision
/// is uniform-size AABB resolved along the minimum-overlap axis (no sqrt) with a
/// uniform spatial-hash grid broad-phase, so thousands of entities stay cheap.
struct CubePile {
    static constexpr core::u32 kNx = 16u; ///< Lattice columns (X).
    static constexpr core::u32 kNy = 4u;  ///< Lattice layers (Y).
    static constexpr core::u32 kNz = 16u; ///< Lattice rows (Z).
    static constexpr core::u32 kCount = kNx * kNy * kNz; ///< 1024 entities.

    static constexpr core::f32 kHalfF = 0.18f;    ///< Cube half-extent.
    static constexpr core::f32 kSpacingF = 0.46f; ///< Initial lattice spacing.
    static constexpr core::f32 kFloorF = -1.6f;   ///< Ground plane (Y).

    /// Spatial-hash grid: power-of-two bucket count (>= kCount) for cheap masking.
    static constexpr core::u32 kBuckets = 2048u;

    CubeEntity entities[kCount];

    /// Seed a kNx*kNy*kNz lattice above the floor (no RNG): each entity gets a
    /// tiny index-derived velocity jitter so the fall is not perfectly uniform.
    void init() noexcept
    {
        using F = math::Fixed32;
        const core::f32 ox = static_cast<core::f32>(kNx - 1u) * 0.5f;
        const core::f32 oz = static_cast<core::f32>(kNz - 1u) * 0.5f;
        static const core::u32 hues[6] = {0x00FF6060u, 0x0060FF60u, 0x006060FFu,
                                          0x00FFFF60u, 0x00FF60FFu, 0x0060FFFFu};
        core::u32 i = 0u;
        for (core::u32 iy = 0; iy < kNy; ++iy)
            for (core::u32 iz = 0; iz < kNz; ++iz)
                for (core::u32 ix = 0; ix < kNx; ++ix, ++i)
                {
                    CubeEntity &e = entities[i];
                    e.x = F::fromFloat((static_cast<core::f32>(ix) - ox) * kSpacingF);
                    e.z = F::fromFloat((static_cast<core::f32>(iz) - oz) * kSpacingF);
                    e.y = F::fromFloat(0.2f + static_cast<core::f32>(iy) * kSpacingF);
                    // Deterministic per-index jitter (small, integer-derived).
                    e.vx = F::fromFloat((static_cast<core::f32>((i * 13u) % 7u) - 3.0f) * 0.004f);
                    e.vz = F::fromFloat((static_cast<core::f32>((i * 17u) % 7u) - 3.0f) * 0.004f);
                    e.vy = F::zero();
                    e.scale = kHalfF;
                    e.tint = hues[i % 6u];
                }
    }

    /// One deterministic tick: integrate under gravity, bounce on the floor,
    /// then resolve inter-entity AABB collisions via a uniform spatial-hash grid.
    void step() noexcept
    {
        using F = math::Fixed32;
        const F gravity = F::fromFloat(-0.010f);
        const F floor = F::fromFloat(kFloorF);
        const F bounce = F::fromFloat(-0.30f);     // damped floor restitution
        const F damp = F::fromFloat(0.985f);       // horizontal friction
        const F twoH = F::fromFloat(2.0f * kHalfF); // AABB full size (uniform)
        const F invCell = F::fromFloat(1.0f / (2.0f * kHalfF)); // 1 / grid cell
        const F restitution = F::fromFloat(0.20f); // pair collision energy kept

        // ── Integrate + floor ────────────────────────────────────────────────
        for (core::u32 i = 0; i < kCount; ++i)
        {
            CubeEntity &e = entities[i];
            e.vy = e.vy + gravity;
            e.x = e.x + e.vx;
            e.y = e.y + e.vy;
            e.z = e.z + e.vz;
            if (e.y < floor)
            {
                e.y = floor;
                e.vy = e.vy * bounce;
                e.vx = e.vx * damp;
                e.vz = e.vz * damp;
            }
        }

        // ── Broad-phase: hash each entity into its grid cell (BSS scratch) ─────
        static core::i32 head[kBuckets];
        static core::i32 next[kCount];
        static core::i32 cellX[kCount];
        static core::i32 cellY[kCount];
        static core::i32 cellZ[kCount];
        for (core::u32 b = 0; b < kBuckets; ++b)
            head[b] = -1;
        for (core::u32 i = 0; i < kCount; ++i)
        {
            const CubeEntity &e = entities[i];
            const core::i32 cx = (e.x * invCell).raw() >> 16; // floor() for Q16.16
            const core::i32 cy = (e.y * invCell).raw() >> 16;
            const core::i32 cz = (e.z * invCell).raw() >> 16;
            cellX[i] = cx;
            cellY[i] = cy;
            cellZ[i] = cz;
            const core::u32 h = cellHash(cx, cy, cz) & (kBuckets - 1u);
            next[i] = head[h];
            head[h] = static_cast<core::i32>(i);
        }

        // ── Narrow-phase: each entity vs neighbours in the 27 adjacent cells ──
        for (core::u32 i = 0; i < kCount; ++i)
        {
            for (core::i32 dz = -1; dz <= 1; ++dz)
                for (core::i32 dy = -1; dy <= 1; ++dy)
                    for (core::i32 dx = -1; dx <= 1; ++dx)
                    {
                        const core::i32 ncx = cellX[i] + dx;
                        const core::i32 ncy = cellY[i] + dy;
                        const core::i32 ncz = cellZ[i] + dz;
                        const core::u32 h = cellHash(ncx, ncy, ncz) & (kBuckets - 1u);
                        for (core::i32 j = head[h]; j >= 0; j = next[j])
                        {
                            const core::u32 ju = static_cast<core::u32>(j);
                            if (ju <= i) // each unordered pair once
                                continue;
                            if (cellX[ju] != ncx || cellY[ju] != ncy || cellZ[ju] != ncz)
                                continue; // hash-collision from another cell
                            resolvePair(entities[i], entities[ju], twoH, restitution);
                        }
                    }
        }
    }

    /// Deterministic integer spatial hash of a grid cell.
    [[nodiscard]] static core::u32 cellHash(core::i32 cx, core::i32 cy, core::i32 cz) noexcept
    {
        return static_cast<core::u32>(cx) * 73856093u ^ static_cast<core::u32>(cy) * 19349663u ^
               static_cast<core::u32>(cz) * 83492791u;
    }

    /// Resolve one AABB pair along its minimum-overlap axis (no sqrt): separate
    /// the boxes and exchange the axis velocity with restitution. Fully Fixed32.
    static void resolvePair(CubeEntity &a, CubeEntity &b, math::Fixed32 twoH, math::Fixed32 restitution) noexcept
    {
        using F = math::Fixed32;
        const F dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        const F ox = twoH - dx.abs();
        if (ox <= F::zero())
            return;
        const F oy = twoH - dy.abs();
        if (oy <= F::zero())
            return;
        const F oz = twoH - dz.abs();
        if (oz <= F::zero())
            return;

        const F half = F::fromFloat(0.5f);
        // Push apart along the axis of least penetration, then damp/exchange the
        // relative velocity on that axis (sign follows the separation direction).
        if (ox <= oy && ox <= oz)
        {
            const F push = (dx < F::zero() ? F::zero() - ox : ox) * half;
            a.x = a.x + push;
            b.x = b.x - push;
            const F m = (a.vx + b.vx) * half;
            a.vx = m + (a.vx - m) * restitution;
            b.vx = m + (b.vx - m) * restitution;
        }
        else if (oy <= ox && oy <= oz)
        {
            const F push = (dy < F::zero() ? F::zero() - oy : oy) * half;
            a.y = a.y + push;
            b.y = b.y - push;
            const F m = (a.vy + b.vy) * half;
            a.vy = m + (a.vy - m) * restitution;
            b.vy = m + (b.vy - m) * restitution;
        }
        else
        {
            const F push = (dz < F::zero() ? F::zero() - oz : oz) * half;
            a.z = a.z + push;
            b.z = b.z - push;
            const F m = (a.vz + b.vz) * half;
            a.vz = m + (a.vz - m) * restitution;
            b.vz = m + (b.vz - m) * restitution;
        }
    }

    /// FNV-1a fold of every entity's authoritative Fixed32 state — the
    /// deterministic, render-independent cross-target signature.
    [[nodiscard]] core::u32 stateSignature() const noexcept
    {
        core::u32 hash = detail::kFnv1aOffsetBasis;
        for (core::u32 i = 0; i < kCount; ++i)
        {
            const CubeEntity &e = entities[i];
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(e.x.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(e.y.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(e.z.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(e.vx.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(e.vy.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(e.vz.raw()));
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
    /// Lambert-lit cube, viewed through the orbit @p cam.
    void render(const RenderTarget &rt, const Camera &cam) const noexcept
    {
        using F = math::Fixed32;
        using Vec3f = math::Vec3<core::f32>;

        clearTarget(rt, 0x00141828u);

        // ── Camera basis via CORDIC (no libm; deterministic) ─────────────────
        F sy{F::zero()}, cy{F::zero()}, sp{F::zero()}, cp{F::zero()};
        math::Cordic::sincos(F::fromFloat(cam.yaw), sy, cy);
        math::Cordic::sincos(F::fromFloat(cam.pitch), sp, cp);
        core::f32 tx = 0.0f, ty = kFloorF + 0.6f, tz = 0.0f;
        if (cam.possess >= 0 && static_cast<core::u32>(cam.possess) < kCount)
        {
            const CubeEntity &pe = entities[static_cast<core::u32>(cam.possess)];
            tx = pe.x.toFloat();
            ty = pe.y.toFloat();
            tz = pe.z.toFloat();
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
            constexpr core::i32 kLines = 17;                 // lines per axis
            constexpr core::f32 kStep = 0.6f;                // world units between lines
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
            {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f},
            {-1.0f, -1.0f, 1.0f},  {1.0f, -1.0f, 1.0f},  {1.0f, 1.0f, 1.0f},  {-1.0f, 1.0f, 1.0f},
        };
        static const core::u32 indices[36] = {
            0, 1, 2, 0, 2, 3, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7,
            1, 5, 6, 1, 6, 2, 4, 5, 1, 4, 1, 0, 3, 2, 6, 3, 6, 7,
        };
        // Constant outward face normals (axis-aligned cubes), indexed by t/2.
        static const core::f32 faceN[6][3] = {
            {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},  {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        };
        // Normalised directional light + ambient term (Lambert diffuse).
        constexpr core::f32 kLx = 0.384f, kLy = 0.816f, kLz = 0.432f;
        constexpr core::f32 kAmbient = 0.30f, kDiffuse = 0.70f;

        // Per-face shade for the possessed entity is overridden to a highlight.
        for (core::u32 ei = 0; ei < kCount; ++ei)
        {
            const CubeEntity &e = entities[ei];
            const bool possessed = (cam.possess >= 0 && static_cast<core::u32>(cam.possess) == ei);
            const core::f32 px = e.x.toFloat();
            const core::f32 py = e.y.toFloat();
            const core::f32 pz = e.z.toFloat();

            detail::ScreenVertex sv[8];
            for (core::u32 i = 0; i < 8u; ++i)
            {
                const core::f32 lx = corners[i][0] * e.scale;
                const core::f32 ly = corners[i][1] * e.scale;
                const core::f32 lz = corners[i][2] * e.scale;
                sv[i] = detail::projectVertex(mvp, px + lx, py + ly, pz + lz, rt.width, rt.height);
            }
            const core::u32 tint = possessed ? 0x0020FF40u : e.tint;
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
    // Static (BSS) — a 1024-entity CubePile is far too large for the kernel
    // stack. Single-threaded deterministic use; re-init makes each call fresh.
    static CubePile scene;
    scene.init();
    for (core::u32 t = 0; t < ticks; ++t)
        scene.step();
    // Fixed camera keeps the image fold deterministic across host and kernel.
    scene.render(rt, CubePile::Camera{});
    return SimFoldResult{scene.stateSignature(), foldTarget(rt)};
}

} // namespace lpl::samples

#endif // LPL_SAMPLES_CUBEPILE_HPP
