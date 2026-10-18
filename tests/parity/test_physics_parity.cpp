#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/Vec3.hpp>
#include <vector>

static int failures = 0;

static constexpr lpl::core::u32 kEntityCount = 100;

struct RunResult {
    std::vector<lpl::core::u32> slots;
    std::vector<float> positions; /**< x,y,z flattened */
};

/**
 * @brief Creates a registry with N entities, writes velocities, reads them back.
 *
 * @details Writes deterministic velocities through the back buffer, swaps, then reads them back from the
 *          front buffer.
 */
static RunResult runOnce(lpl::core::u32 entityCount)
{
    lpl::ecs::Registry reg;

    lpl::ecs::Archetype arch;
    arch.add(lpl::ecs::ComponentId::Position);
    arch.add(lpl::ecs::ComponentId::Velocity);
    arch.add(lpl::ecs::ComponentId::Mass);

    RunResult result;

    for (lpl::core::u32 i = 0; i < entityCount; ++i)
    {
        auto res = reg.createEntity(arch);
        if (!res)
            continue;
        result.slots.push_back(res.value().slot());
    }

    for (const auto &part : reg.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            if (n == 0)
                continue;

            auto *vel = static_cast<lpl::math::Vec3<float> *>(chunk->writeComponent(lpl::ecs::ComponentId::Velocity));
            if (vel)
            {
                auto ids = chunk->entities();
                for (lpl::core::u32 i = 0; i < n; ++i)
                {
                    const float s = static_cast<float>(ids[i].slot());
                    vel[i] = {s * 0.1f, s * -9.81f, s * 0.05f};
                }
            }

            lpl::ecs::ComponentBytes moved{};
            chunk->swapBuffers(moved);
        }
    }

    for (const auto &part : reg.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            if (n == 0)
                continue;

            auto *vel =
                static_cast<const lpl::math::Vec3<float> *>(chunk->readComponent(lpl::ecs::ComponentId::Velocity));
            if (!vel)
                continue;

            for (lpl::core::u32 i = 0; i < n; ++i)
            {
                result.positions.push_back(vel[i].x);
                result.positions.push_back(vel[i].y);
                result.positions.push_back(vel[i].z);
            }
        }
    }
    return result;
}

/**
 * @brief Checks that both runs assign the same slots to their entities.
 */
static void checkSlotAssignmentsAreDeterministic(const RunResult &result1, const RunResult &result2)
{
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
}

/**
 * @brief Checks that both runs read back bit-identical component data.
 */
static void checkComponentDataIsDeterministic(const RunResult &result1, const RunResult &result2)
{
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
}

/**
 * @brief Parity test: ECS entity creation and component determinism.
 *
 * Verifies that creating the same entities twice with identical
 * archetypes produces identical component memory layouts and entity
 * slot assignments (bit-exact determinism).
 */
int main()
{
    lpl::core::Log::info("=== ECS Parity Test ===");

    auto result1 = runOnce(kEntityCount);
    auto result2 = runOnce(kEntityCount);

    checkSlotAssignmentsAreDeterministic(result1, result2);

    checkComponentDataIsDeterministic(result1, result2);

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
