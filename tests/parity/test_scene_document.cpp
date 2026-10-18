#include <cstdio>
#include <cstring>
#include <span>

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

using FVec3 = lpl::math::Vec3<lpl::math::Fixed32>;

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

/**
 * @brief FNV-1a fold of Position/Velocity (Fixed32 raw) + Health (i32) over a registry.
 */
static lpl::core::u32 foldState(const lpl::ecs::Registry &registry)
{
    lpl::core::u32 h = 0x811C9DC5u;
    auto step = [&](lpl::core::u32 v) { h = (h ^ v) * 0x01000193u; };
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
            const auto *vel = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Velocity));
            const auto *hp = static_cast<const lpl::core::i32 *>(chunk->readComponent(lpl::ecs::ComponentId::Health));
            for (lpl::core::u32 i = 0; i < n; ++i)
            {
                if (pos)
                {
                    step(static_cast<lpl::core::u32>(pos[i].x.raw()));
                    step(static_cast<lpl::core::u32>(pos[i].y.raw()));
                    step(static_cast<lpl::core::u32>(pos[i].z.raw()));
                }
                if (vel)
                {
                    step(static_cast<lpl::core::u32>(vel[i].x.raw()));
                    step(static_cast<lpl::core::u32>(vel[i].y.raw()));
                    step(static_cast<lpl::core::u32>(vel[i].z.raw()));
                }
                if (hp)
                    step(static_cast<lpl::core::u32>(hp[i]));
            }
        }
    }
    return h;
}

/**
 * @brief Builds a small source world.
 *
 * @param src Receives the entities, their Position, Velocity and Health seeded in both buffers.
 * @param kN  The number of entities to create.
 */
static void buildSmallSourceWorld(lpl::ecs::Registry &src, lpl::core::u32 kN)
{
    const lpl::ecs::ComponentId ids[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::Velocity,
                                         lpl::ecs::ComponentId::Health};
    const lpl::ecs::Archetype arch{ids};
    for (lpl::core::u32 i = 0; i < kN; ++i)
        (void) src.createEntity(arch);

    lpl::core::u32 gi = 0;
    for (const auto &part : src.partitions())
    {
        for (const auto &chunk : part->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            auto seed = [&](lpl::ecs::ComponentId id, auto writer) {
                if (auto *wb = static_cast<lpl::core::byte *>(chunk->writeComponent(id)))
                    writer(wb);
                if (auto *rb = static_cast<lpl::core::byte *>(const_cast<void *>(chunk->readComponent(id))))
                    writer(rb);
            };
            for (lpl::core::u32 li = 0; li < n; ++li, ++gi)
            {
                const lpl::core::u32 k = gi;
                seed(lpl::ecs::ComponentId::Position, [&](lpl::core::byte *b) {
                    reinterpret_cast<FVec3 *>(b)[li] = {
                        lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(k)),
                        lpl::math::Fixed32::fromFloat(1.5f * static_cast<lpl::core::f32>(k)),
                        lpl::math::Fixed32::fromInt(-static_cast<lpl::core::i32>(k))};
                });
                seed(lpl::ecs::ComponentId::Velocity, [&](lpl::core::byte *b) {
                    reinterpret_cast<FVec3 *>(b)[li] = {lpl::math::Fixed32::fromFloat(0.25f),
                                                        lpl::math::Fixed32::zero(),
                                                        lpl::math::Fixed32::fromFloat(-0.5f)};
                });
                seed(lpl::ecs::ComponentId::Health, [&](lpl::core::byte *b) {
                    reinterpret_cast<lpl::core::i32 *>(b)[li] = 100 + static_cast<lpl::core::i32>(k) * 10;
                });
            }
        }
    }
}

/**
 * @brief Serialize → load → re-serialize.
 *
 * @param src The source world.
 * @param kN  The number of entities it holds.
 */
static void checkSerializeLoadReserializeIsLossless(const lpl::ecs::Registry &src, lpl::core::u32 kN)
{
    const std::string doc1 = lpl::editor::toLplScene(src);
    std::printf("-- .lplscene document --\n%s\n\n", doc1.c_str());

    lpl::ecs::Registry dst;
    const auto loaded = lpl::editor::fromLplScene(doc1, dst);
    check(loaded.has_value(), "fromLplScene succeeds");
    if (loaded.has_value())
        check(loaded.value() == kN, "all entities loaded");

    const std::string doc2 = lpl::editor::toLplScene(dst);
    check(doc1 == doc2, "re-serialized document is byte-identical");
    check(foldState(src) == foldState(dst), "authoritative state fold matches");

    std::printf("\n  src fold = 0x%08X   dst fold = 0x%08X\n", foldState(src), foldState(dst));
}

/**
 * @brief Round-trip proof for the reflection-driven `.lplscene` serializer.
 *
 * @details Builds a small ECS world (Position/Velocity in authoritative Fixed32 raw, plus an integer Health),
 *          serializes it to a `.lplscene` document, loads it back into a fresh registry, and checks that (1)
 *          re-serializing yields a byte-identical document and (2) an FNV-1a fold of the authoritative state
 *          matches. This proves the JSON is a faithful, lossless view of the ECS — the "JSON is the source of
 *          truth, the registry is a view" foundation, driven entirely by the component reflection registry (no
 *          per-component IO code).
 *
 *          Host-only. Build via xmake: `xmake run test-scene-document`.
 */
int main()
{
    std::printf("== .lplscene round-trip parity ==\n\n");

    lpl::ecs::Registry src;
    constexpr lpl::core::u32 kN = 5u;
    buildSmallSourceWorld(src, kN);

    checkSerializeLoadReserializeIsLossless(src, kN);

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
