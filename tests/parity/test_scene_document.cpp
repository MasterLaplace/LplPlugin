/**
 * @file test_scene_document.cpp
 * @brief Round-trip proof for the reflection-driven `.lplscene` serializer.
 *
 * Builds a small ECS world (Position/Velocity in authoritative Fixed32 raw, plus
 * an integer Health), serializes it to a `.lplscene` document, loads it back into
 * a fresh registry, and checks that (1) re-serializing yields a byte-identical
 * document and (2) an FNV-1a fold of the authoritative state matches. This proves
 * the JSON is a faithful, lossless view of the ECS — the "JSON is the source of
 * truth, the registry is a view" foundation, driven entirely by the component
 * reflection registry (no per-component IO code).
 *
 * Host-only. Build via xmake: `xmake run test-scene-document`.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

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

using namespace lpl;
using math::Fixed32;
using FVec3 = math::Vec3<Fixed32>;

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

// FNV-1a fold of Position/Velocity (Fixed32 raw) + Health (i32) over a registry.
static core::u32 foldState(const ecs::Registry &registry)
{
    core::u32 h = 0x811C9DC5u;
    auto step = [&](core::u32 v) { h = (h ^ v) * 0x01000193u; };
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
            const auto *vel = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Velocity));
            const auto *hp = static_cast<const core::i32 *>(chunk->readComponent(ecs::ComponentId::Health));
            for (core::u32 i = 0; i < n; ++i)
            {
                if (pos)
                {
                    step(static_cast<core::u32>(pos[i].x.raw()));
                    step(static_cast<core::u32>(pos[i].y.raw()));
                    step(static_cast<core::u32>(pos[i].z.raw()));
                }
                if (vel)
                {
                    step(static_cast<core::u32>(vel[i].x.raw()));
                    step(static_cast<core::u32>(vel[i].y.raw()));
                    step(static_cast<core::u32>(vel[i].z.raw()));
                }
                if (hp)
                    step(static_cast<core::u32>(hp[i]));
            }
        }
    }
    return h;
}

int main()
{
    std::printf("== .lplscene round-trip parity ==\n\n");

    // ── Build a small source world ───────────────────────────────────────────
    ecs::Registry src;
    const ecs::ComponentId ids[] = {ecs::ComponentId::Position, ecs::ComponentId::Velocity, ecs::ComponentId::Health};
    const ecs::Archetype arch{ids};
    constexpr core::u32 kN = 5u;
    for (core::u32 i = 0; i < kN; ++i)
        (void) src.createEntity(arch);

    core::u32 gi = 0;
    for (const auto &part : src.partitions())
    {
        for (const auto &chunk : part->chunks())
        {
            const core::u32 n = chunk->count();
            auto seed = [&](ecs::ComponentId id, auto writer) {
                if (auto *wb = static_cast<core::byte *>(chunk->writeComponent(id)))
                    writer(wb);
                if (auto *rb = static_cast<core::byte *>(const_cast<void *>(chunk->readComponent(id))))
                    writer(rb);
            };
            for (core::u32 li = 0; li < n; ++li, ++gi)
            {
                const core::u32 k = gi;
                seed(ecs::ComponentId::Position, [&](core::byte *b) {
                    reinterpret_cast<FVec3 *>(b)[li] = {Fixed32::fromInt(static_cast<core::i32>(k)),
                                                        Fixed32::fromFloat(1.5f * static_cast<core::f32>(k)),
                                                        Fixed32::fromInt(-static_cast<core::i32>(k))};
                });
                seed(ecs::ComponentId::Velocity, [&](core::byte *b) {
                    reinterpret_cast<FVec3 *>(b)[li] = {Fixed32::fromFloat(0.25f), Fixed32::zero(),
                                                        Fixed32::fromFloat(-0.5f)};
                });
                seed(ecs::ComponentId::Health, [&](core::byte *b) {
                    reinterpret_cast<core::i32 *>(b)[li] = 100 + static_cast<core::i32>(k) * 10;
                });
            }
        }
    }

    // ── Serialize → load → re-serialize ──────────────────────────────────────
    const std::string doc1 = editor::toLplScene(src);
    std::printf("-- .lplscene document --\n%s\n\n", doc1.c_str());

    ecs::Registry dst;
    const auto loaded = editor::fromLplScene(doc1, dst);
    check(loaded.has_value(), "fromLplScene succeeds");
    if (loaded.has_value())
        check(loaded.value() == kN, "all entities loaded");

    const std::string doc2 = editor::toLplScene(dst);
    check(doc1 == doc2, "re-serialized document is byte-identical");
    check(foldState(src) == foldState(dst), "authoritative state fold matches");

    std::printf("\n  src fold = 0x%08X   dst fold = 0x%08X\n", foldState(src), foldState(dst));
    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
