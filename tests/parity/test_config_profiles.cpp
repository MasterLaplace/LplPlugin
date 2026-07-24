/*
** LplPlugin — game-profile presets + config validation test (book chapter 6)
**
** Proves two things: every genre preset (applyGameProfile) is internally
** self-consistent — it raises zero configuration warnings — and the validator
** (forEachConfigWarning) actually catches the contradictions it is meant to, so a
** misconfiguration is loud instead of a silent null-deref or wasted bandwidth.
*/

#include <lpl/engine/ConfigValidation.hpp>
#include <lpl/engine/GameProfile.hpp>

#include <cstdio>
#include <cstring>

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

core::u32 warningCount(const engine::Config &cfg)
{
    return engine::forEachConfigWarning(cfg, [](const char *) {});
}

bool anyWarningContains(const engine::Config &cfg, const char *needle)
{
    bool found = false;
    engine::forEachConfigWarning(cfg, [&](const char *msg) {
        if (std::strstr(msg, needle) != nullptr)
            found = true;
    });
    return found;
}

engine::Config withProfile(engine::GameProfile p)
{
    auto b = engine::Config::Builder{};
    b.serverMode(true).tickRate(144);
    engine::applyGameProfile(b, p);
    return b.build();
}

} // namespace

int main()
{
    std::printf("== game profiles + config validation ==\n");

    // ── Every preset is conflict-free ──────────────────────────────────────── //
    {
        const engine::GameProfile all[] = {engine::GameProfile::Mmorpg, engine::GameProfile::Fps,
                                           engine::GameProfile::Rts, engine::GameProfile::Fighting,
                                           engine::GameProfile::CoOp};
        for (auto p : all)
        {
            const auto cfg = withProfile(p);
            char label[96];
            std::snprintf(label, sizeof(label), "%s preset raises no configuration warnings",
                          engine::gameProfileName(p));
            check(warningCount(cfg) == 0, label);
        }
    }

    // ── The MMORPG preset really does turn the scaling levers on ───────────── //
    {
        const auto cfg = withProfile(engine::GameProfile::Mmorpg);
        check(cfg.interestRadius() > math::Fixed32::zero(), "MMORPG enables AOI");
        check(cfg.lodNearRadius() > math::Fixed32::zero() && cfg.lodNearRadius() < cfg.interestRadius(),
              "MMORPG enables network LOD with a near ring inside the interest radius");
        check(cfg.worldExtent() > math::Fixed32::zero(), "MMORPG enables precision LOD");
    }

    // ── The FPS preset keeps precision and reliability, drops LOD ───────────── //
    {
        const auto cfg = withProfile(engine::GameProfile::Fps);
        check(cfg.lodNearRadius() == math::Fixed32::zero(), "FPS has no LOD (precision matters for hits)");
        check(cfg.reliableBaseline(), "FPS uses the reliable acked baseline");
        check(warningCount(cfg) == 0, "FPS is still conflict-free");
    }

    // ── The validator catches each contradiction ───────────────────────────── //
    {
        using B = engine::Config::Builder;

        // AOI without physics → would deref a null spatial index.
        const auto noPhysics =
            B{}.serverMode(true).enablePhysics(false).interestRadius(math::Fixed32::fromFloat(50.0f)).build();
        check(anyWarningContains(noPhysics, "physics"), "AOI without physics is flagged");

        // AOI without networking.
        const auto noNet =
            B{}.serverMode(true).enableNetworking(false).interestRadius(math::Fixed32::fromFloat(50.0f)).build();
        check(anyWarningContains(noNet, "networking"), "AOI without networking is flagged");

        // Near ring >= interest radius → far ring empty.
        const auto badRings = B{}
                                  .serverMode(true)
                                  .interestRadius(math::Fixed32::fromFloat(50.0f))
                                  .lodNearRadius(math::Fixed32::fromFloat(60.0f))
                                  .build();
        check(anyWarningContains(badRings, "near ring"), "a near ring wider than the interest radius is flagged");

        // Precision LOD without network LOD.
        const auto badPrecision = B{}
                                      .serverMode(true)
                                      .interestRadius(math::Fixed32::fromFloat(50.0f))
                                      .worldExtent(math::Fixed32::fromFloat(1000.0f))
                                      .build();
        check(anyWarningContains(badPrecision, "quantization"), "precision LOD without network LOD is flagged");

        // Server-side AOI set on a client.
        const auto clientAoi =
            B{}.serverMode(false).interestRadius(math::Fixed32::fromFloat(50.0f)).build();
        check(anyWarningContains(clientAoi, "client"), "server-side AOI on a client is flagged");

        // A clean full-broadcast server has nothing to warn about.
        const auto clean = B{}.serverMode(true).enablePhysics(true).enableNetworking(true).build();
        check(warningCount(clean) == 0, "a plain full-broadcast server is conflict-free");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
