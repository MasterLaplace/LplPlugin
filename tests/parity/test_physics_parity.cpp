/**
 * @file test_physics_parity.cpp
 * @brief Parity test: ECS entity creation and component determinism.
 *
 * Verifies that creating the same entities twice with identical
 * archetypes produces identical component memory layouts and entity
 * slot assignments (bit-exact determinism).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/Vec3.hpp>
#include <vector>

using namespace lpl;

static int failures = 0;

struct RunResult {
    std::vector<core::u32> slots;
    std::vector<float> positions; // x,y,z flattened
};

/**
 * @brief Creates a registry with N entities, writes positions, reads them back.
 */
static RunResult runOnce(core::u32 entityCount)
{
    ecs::Registry reg;

    ecs::Archetype arch;
    arch.add(ecs::ComponentId::Position);
    arch.add(ecs::ComponentId::Velocity);
    arch.add(ecs::ComponentId::Mass);

    RunResult result;

    for (core::u32 i = 0; i < entityCount; ++i)
    {
        auto res = reg.createEntity(arch);
        if (!res)
            continue;
        result.slots.push_back(res.value().slot());
    }

    // Write deterministic positions via back buffer, then swap
    for (const auto &part : reg.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const core::u32 n = chunk->count();
            if (n == 0)
                continue;

            auto *vel = static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            if (vel)
            {
                auto ids = chunk->entities();
                for (core::u32 i = 0; i < n; ++i)
                {
                    const float s = static_cast<float>(ids[i].slot());
                    vel[i] = {s * 0.1f, s * -9.81f, s * 0.05f};
                }
            }

            chunk->swapBuffers();
        }
    }

    // Read back positions from front buffer
    for (const auto &part : reg.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const core::u32 n = chunk->count();
            if (n == 0)
                continue;

            auto *vel = static_cast<const math::Vec3<float> *>(chunk->readComponent(ecs::ComponentId::Velocity));
            if (!vel)
                continue;

            for (core::u32 i = 0; i < n; ++i)
            {
                result.positions.push_back(vel[i].x);
                result.positions.push_back(vel[i].y);
                result.positions.push_back(vel[i].z);
            }
        }
    }
    return result;
}

int main()
{
    core::Log::info("=== ECS Parity Test ===");

    constexpr core::u32 kEntityCount = 100;

    auto result1 = runOnce(kEntityCount);
    auto result2 = runOnce(kEntityCount);

    // Check slot assignments are deterministic
    if (result1.slots.size() != result2.slots.size())
    {
        std::printf("  FAIL: Slot count mismatch %zu vs %zu\n", result1.slots.size(), result2.slots.size());
        ++failures;
    }
    else
    {
        bool same = true;
        for (size_t i = 0; i < result1.slots.size(); ++i)
        {
            if (result1.slots[i] != result2.slots[i])
            {
                std::printf("  FAIL: Slot divergence at entity %zu: %u vs %u\n", i, result1.slots[i], result2.slots[i]);
                same = false;
                ++failures;
                break;
            }
        }
        if (same)
            std::printf("  PASS: %u entity slots, deterministic assignment\n", kEntityCount);
    }

    // Check component data is deterministic
    if (result1.positions.size() != result2.positions.size())
    {
        std::printf("  FAIL: Data size mismatch %zu vs %zu\n", result1.positions.size(), result2.positions.size());
        ++failures;
    }
    else if (result1.positions.empty())
    {
        std::printf("  FAIL: No position data read back\n");
        ++failures;
    }
    else
    {
        bool identical = true;
        for (size_t i = 0; i < result1.positions.size(); ++i)
        {
            if (std::memcmp(&result1.positions[i], &result2.positions[i], sizeof(float)) != 0)
            {
                std::printf("  FAIL: Data divergence at index %zu: %.6f vs %.6f\n", i, result1.positions[i],
                            result2.positions[i]);
                identical = false;
                ++failures;
                break;
            }
        }
        if (identical)
        {
            std::printf("  PASS: %zu floats, bit-exact determinism\n", result1.positions.size());
        }
    }

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
