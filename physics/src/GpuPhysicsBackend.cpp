/**
 * @file GpuPhysicsBackend.cpp
 * @brief Host-side bridge: drives the physics_tick compute kernel over ECS.
 *
 * Mirrors the CpuPhysicsBackend chunk traversal but delegates integration to an
 * IComputeBackend. Only the integration pass (gravity + semi-implicit Euler) is
 * GPU-accelerated; collision and sleeping remain CPU/future work.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-29
 * @copyright MIT License
 */

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/physics/GpuPhysicsBackend.hpp>

#include <cstring>
#include <utility>

namespace lpl::physics {

namespace {

/** @brief Argument buffer for the "physics_tick" kernel (see CudaBackend). */
struct PhysicsTickArgs {
    float *posX, *posY, *posZ;
    float *velX, *velY, *velZ;
    float *frcX, *frcY, *frcZ;
    const float *masses;
    core::u32 count;
    float dt;
};

constexpr core::u32 kBlockDim = 256u;

} // namespace

GpuPhysicsBackend::GpuPhysicsBackend(ecs::Registry &registry, IComputeBackend &compute) noexcept
    : _registry(registry), _compute(compute)
{
}

core::Expected<void> GpuPhysicsBackend::init() { return _compute.init(); }

void GpuPhysicsBackend::shutdown()
{
    releaseScratch();
    _compute.shutdown();
}

const char *GpuPhysicsBackend::name() const noexcept { return "GpuPhysicsBackend"; }

void GpuPhysicsBackend::releaseScratch() noexcept
{
    void *const ptrs[] = {_scratch.dPosX, _scratch.dPosY, _scratch.dPosZ, _scratch.dVelX, _scratch.dVelY,
                          _scratch.dVelZ, _scratch.dFrcX, _scratch.dFrcY, _scratch.dFrcZ, _scratch.dMass};
    for (void *p : ptrs)
    {
        if (p)
            _compute.free(p);
    }
    _scratch = DeviceScratch{};
}

core::Expected<void> GpuPhysicsBackend::ensureCapacity(core::u32 count)
{
    if (count <= _scratch.capacity)
        return {};

    releaseScratch();

    const core::usize bytes = static_cast<core::usize>(count) * sizeof(float);
    void **const slots[] = {&_scratch.dPosX, &_scratch.dPosY, &_scratch.dPosZ, &_scratch.dVelX, &_scratch.dVelY,
                            &_scratch.dVelZ, &_scratch.dFrcX, &_scratch.dFrcY, &_scratch.dFrcZ, &_scratch.dMass};
    for (void **slot : slots)
    {
        auto alloc = _compute.allocate(bytes);
        if (!alloc)
            return core::Unexpected(std::move(alloc.error()));
        *slot = alloc.value();
    }

    _scratch.hostX.resize(count);
    _scratch.hostY.resize(count);
    _scratch.hostZ.resize(count);
    _scratch.hostVX.resize(count);
    _scratch.hostVY.resize(count);
    _scratch.hostVZ.resize(count);
    _scratch.hostMass.resize(count);
    _scratch.capacity = count;
    return {};
}

core::Expected<void> GpuPhysicsBackend::step(core::f32 dt)
{
    for (const auto &partition : _registry.partitions())
    {
        const auto &archetype = partition->archetype();
        if (!archetype.has(ecs::ComponentId::Position) || !archetype.has(ecs::ComponentId::Velocity) ||
            !archetype.has(ecs::ComponentId::Mass))
        {
            continue;
        }

        for (const auto &chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
                continue;

            auto *positions =
                static_cast<math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *velocities =
                static_cast<math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto *masses = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
            if (!positions || !velocities || !masses)
                continue;

            if (auto sized = ensureCapacity(count); !sized)
                return sized;

            // Deinterleave AoS Vec3 → planar SoA host staging. The GPU kernel is
            // float; convert authoritative Fixed32 → float at this staging edge.
            // NOTE: this backend is compiled only under --cuda and was not built
            // in the migration sandbox — treat as compile-unverified.
            for (core::u32 i = 0; i < count; ++i)
            {
                _scratch.hostX[i] = positions[i].x.toFloat();
                _scratch.hostY[i] = positions[i].y.toFloat();
                _scratch.hostZ[i] = positions[i].z.toFloat();
                _scratch.hostVX[i] = velocities[i].x.toFloat();
                _scratch.hostVY[i] = velocities[i].y.toFloat();
                _scratch.hostVZ[i] = velocities[i].z.toFloat();
                _scratch.hostMass[i] = masses[i].toFloat();
            }

            const core::usize bytes = static_cast<core::usize>(count) * sizeof(float);
            struct Upload {
                void *dst;
                const float *src;
            };
            const Upload uploads[] = {
                {_scratch.dPosX, _scratch.hostX.data()   },
                {_scratch.dPosY, _scratch.hostY.data()   },
                {_scratch.dPosZ, _scratch.hostZ.data()   },
                {_scratch.dVelX, _scratch.hostVX.data()  },
                {_scratch.dVelY, _scratch.hostVY.data()  },
                {_scratch.dVelZ, _scratch.hostVZ.data()  },
                {_scratch.dMass, _scratch.hostMass.data()}
            };
            for (const auto &up : uploads)
            {
                if (auto r = _compute.uploadSync(up.dst, up.src, bytes); !r)
                    return r;
            }

            PhysicsTickArgs args{};
            args.posX = static_cast<float *>(_scratch.dPosX);
            args.posY = static_cast<float *>(_scratch.dPosY);
            args.posZ = static_cast<float *>(_scratch.dPosZ);
            args.velX = static_cast<float *>(_scratch.dVelX);
            args.velY = static_cast<float *>(_scratch.dVelY);
            args.velZ = static_cast<float *>(_scratch.dVelZ);
            args.frcX = static_cast<float *>(_scratch.dFrcX);
            args.frcY = static_cast<float *>(_scratch.dFrcY);
            args.frcZ = static_cast<float *>(_scratch.dFrcZ);
            args.masses = static_cast<const float *>(_scratch.dMass);
            args.count = count;
            args.dt = dt;

            const core::u32 grid = (count + kBlockDim - 1u) / kBlockDim;
            const auto *argBytes = reinterpret_cast<const core::byte *>(&args);
            if (auto r = _compute.dispatch("physics_tick", grid, kBlockDim,
                                           std::span<const core::byte>(argBytes, sizeof(args)));
                !r)
            {
                return r;
            }
            if (auto r = _compute.synchronize(); !r)
                return r;

            // Read positions + velocities back; reinterleave SoA → AoS Vec3.
            struct Download {
                float *dst;
                const void *src;
            };
            const Download downloads[] = {
                {_scratch.hostX.data(),  _scratch.dPosX},
                {_scratch.hostY.data(),  _scratch.dPosY},
                {_scratch.hostZ.data(),  _scratch.dPosZ},
                {_scratch.hostVX.data(), _scratch.dVelX},
                {_scratch.hostVY.data(), _scratch.dVelY},
                {_scratch.hostVZ.data(), _scratch.dVelZ}
            };
            for (const auto &dn : downloads)
            {
                if (auto r = _compute.downloadSync(dn.dst, dn.src, bytes); !r)
                    return r;
            }

            for (core::u32 i = 0; i < count; ++i)
            {
                positions[i].x = math::Fixed32::fromFloat(_scratch.hostX[i]);
                positions[i].y = math::Fixed32::fromFloat(_scratch.hostY[i]);
                positions[i].z = math::Fixed32::fromFloat(_scratch.hostZ[i]);
                velocities[i].x = math::Fixed32::fromFloat(_scratch.hostVX[i]);
                velocities[i].y = math::Fixed32::fromFloat(_scratch.hostVY[i]);
                velocities[i].z = math::Fixed32::fromFloat(_scratch.hostVZ[i]);
            }
        }
    }

    return {};
}

} // namespace lpl::physics
