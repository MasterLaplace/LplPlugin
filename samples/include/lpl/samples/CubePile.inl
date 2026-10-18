#ifndef LPL_SAMPLES_CUBEPILE_INL
#define LPL_SAMPLES_CUBEPILE_INL

namespace lpl::samples {

inline void CubePile::init() noexcept
{
    using F = Fixed32;
    const ecs::ComponentId ids[] = {ecs::ComponentId::Position, ecs::ComponentId::Velocity, ecs::ComponentId::AABB,
                                    ecs::ComponentId::Mass};
    const ecs::Archetype archetype{ids};
    for (core::u32 i = 0; i < kCount; ++i)
        (void) registry.createEntity(archetype);
    (void) backend.init();

    static const core::u32 hues[6] = {0x00FF6060u, 0x0060FF60u, 0x006060FFu, 0x00FFFF60u, 0x00FF60FFu, 0x0060FFFFu};
    const core::f32 ox = static_cast<core::f32>(kNx - 1u) * 0.5f;
    const core::f32 oz = static_cast<core::f32>(kNz - 1u) * 0.5f;
    const F cubeSize = F::fromFloat(2.0f * kHalfF);
    const F one = F::one();

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

inline core::u32 CubePile::stateSignature() const noexcept
{
    core::u32 hash = render::detail::kFnv1aOffsetBasis;
    for (const auto &partition : registry.partitions())
    {
        for (const auto &chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->latestComponent(ecs::ComponentId::Position));
            const auto *vel = static_cast<const FVec3 *>(chunk->latestComponent(ecs::ComponentId::Velocity));
            if (!pos || !vel)
                continue;
            for (core::u32 li = 0; li < count; ++li)
            {
                hash = render::detail::fnv1aStep(hash, static_cast<core::u32>(pos[li].x.raw()));
                hash = render::detail::fnv1aStep(hash, static_cast<core::u32>(pos[li].y.raw()));
                hash = render::detail::fnv1aStep(hash, static_cast<core::u32>(pos[li].z.raw()));
                hash = render::detail::fnv1aStep(hash, static_cast<core::u32>(vel[li].x.raw()));
                hash = render::detail::fnv1aStep(hash, static_cast<core::u32>(vel[li].y.raw()));
                hash = render::detail::fnv1aStep(hash, static_cast<core::u32>(vel[li].z.raw()));
            }
        }
    }
    return hash;
}

inline void CubePile::render(const render::RenderTarget &rt, const Camera &cam) const noexcept
{
    using F = Fixed32;
    using Vec3f = math::Vec3<core::f32>;

    render::clearTarget(rt, 0x00141828u);

    static core::f32 px[kCount], py[kCount], pz[kCount];
    core::u32 gi = 0;
    for (const auto &partition : registry.partitions())
    {
        for (const auto &chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->latestComponent(ecs::ComponentId::Position));
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
    const auto proj = render::perspectiveFov(F::fromFloat(1.04719755f), aspect, 0.1f, 100.0f);
    const auto mvp = proj * view;

    {
        constexpr core::i32 kLines = 17;
        constexpr core::f32 kStep = 0.6f;
        const core::f32 half = static_cast<core::f32>(kLines - 1) * 0.5f * kStep;
        for (core::i32 g = 0; g < kLines; ++g)
        {
            const core::f32 t = static_cast<core::f32>(g) * kStep - half;
            const render::detail::ScreenVertex ax =
                render::detail::projectVertex(mvp, -half, kFloorF, t, rt.width, rt.height);
            const render::detail::ScreenVertex bx =
                render::detail::projectVertex(mvp, half, kFloorF, t, rt.width, rt.height);
            plotLine(rt, ax, bx, 0x00203048u);
            const render::detail::ScreenVertex az =
                render::detail::projectVertex(mvp, t, kFloorF, -half, rt.width, rt.height);
            const render::detail::ScreenVertex bz =
                render::detail::projectVertex(mvp, t, kFloorF, half, rt.width, rt.height);
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
    static const core::f32 faceN[6][3] = {
        {0.0f,  0.0f,  -1.0f},
        {0.0f,  0.0f,  1.0f },
        {-1.0f, 0.0f,  0.0f },
        {1.0f,  0.0f,  0.0f },
        {0.0f,  -1.0f, 0.0f },
        {0.0f,  1.0f,  0.0f },
    };
    constexpr core::f32 kLx = 0.384f, kLy = 0.816f, kLz = 0.432f;
    constexpr core::f32 kAmbient = 0.30f, kDiffuse = 0.70f;

    for (core::u32 ei = 0; ei < total; ++ei)
    {
        const bool possessed = (cam.possess >= 0 && static_cast<core::u32>(cam.possess) == ei);
        render::detail::ScreenVertex sv[8];
        for (core::u32 i = 0; i < 8u; ++i)
        {
            const core::f32 lx = corners[i][0] * kHalfF;
            const core::f32 ly = corners[i][1] * kHalfF;
            const core::f32 lz = corners[i][2] * kHalfF;
            sv[i] = render::detail::projectVertex(mvp, px[ei] + lx, py[ei] + ly, pz[ei] + lz, rt.width, rt.height);
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
            render::detail::fillTriangle(rt, sv[indices[t * 3 + 0]], sv[indices[t * 3 + 1]], sv[indices[t * 3 + 2]],
                                         color);
        }
    }
    static_cast<void>(kLz);
}

inline void CubePile::plotLine(const render::RenderTarget &rt, const render::detail::ScreenVertex &a,
                               const render::detail::ScreenVertex &b, core::u32 color) noexcept
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

} // namespace lpl::samples

#endif // LPL_SAMPLES_CUBEPILE_INL
