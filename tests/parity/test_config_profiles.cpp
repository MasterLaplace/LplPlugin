#include <lpl/engine/ConfigValidation.hpp>
#include <lpl/engine/GameProfile.hpp>
#include <lpl/engine/HostProfile.hpp>

#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

lpl::core::u32 warningCount(const lpl::engine::Config &cfg)
{
    return lpl::engine::forEachConfigWarning(cfg, [](const char *) {});
}

bool anyWarningContains(const lpl::engine::Config &cfg, const char *needle)
{
    bool found = false;
    lpl::engine::forEachConfigWarning(cfg, [&](const char *msg) {
        if (std::strstr(msg, needle) != nullptr)
            found = true;
    });
    return found;
}

lpl::engine::Config withProfile(lpl::engine::GameProfile p)
{
    auto b = lpl::engine::Config::Builder{};
    b.serverMode(true).tickRate(144);
    lpl::engine::applyGameProfile(b, p);
    return b.build();
}

/**
 * @brief Every preset is conflict-free.
 */
void checkEveryPresetIsConflictFree()
{
    const lpl::engine::GameProfile all[] = {lpl::engine::GameProfile::Mmorpg, lpl::engine::GameProfile::Fps,
                                            lpl::engine::GameProfile::Rts, lpl::engine::GameProfile::Fighting,
                                            lpl::engine::GameProfile::CoOp};
    for (auto p : all)
    {
        const auto cfg = withProfile(p);
        char label[96];
        std::snprintf(label, sizeof(label), "%s preset raises no configuration warnings",
                      lpl::engine::gameProfileName(p));
        check(warningCount(cfg) == 0, label);
    }
}

/**
 * @brief The MMORPG preset really does turn the scaling levers on.
 */
void checkTheMmorpgPresetTurnsTheScalingLeversOn()
{
    const auto cfg = withProfile(lpl::engine::GameProfile::Mmorpg);
    check(cfg.interestRadius() > lpl::math::Fixed32::zero(), "MMORPG enables AOI");
    check(cfg.lodNearRadius() > lpl::math::Fixed32::zero() && cfg.lodNearRadius() < cfg.interestRadius(),
          "MMORPG enables network LOD with a near ring inside the interest radius");
    check(cfg.worldExtent() > lpl::math::Fixed32::zero(), "MMORPG enables precision LOD");
}

/**
 * @brief The FPS preset keeps precision and reliability, drops LOD.
 */
void checkTheFpsPresetKeepsPrecisionAndReliabilityAndDropsLod()
{
    const auto cfg = withProfile(lpl::engine::GameProfile::Fps);
    check(cfg.lodNearRadius() == lpl::math::Fixed32::zero(), "FPS has no LOD (precision matters for hits)");
    check(cfg.reliableBaseline(), "FPS uses the reliable acked baseline");
    check(warningCount(cfg) == 0, "FPS is still conflict-free");
}

/**
 * @brief The validator catches each contradiction.
 *
 * @details AOI without physics would dereference a null spatial index; a near ring at least as wide as the interest
 *          radius leaves the far ring empty. A clean full-broadcast server has nothing to warn about.
 */
void checkTheValidatorCatchesEachContradiction()
{
    using B = lpl::engine::Config::Builder;

    const auto noPhysics =
        B{}.serverMode(true).enablePhysics(false).interestRadius(lpl::math::Fixed32::fromFloat(50.0f)).build();
    check(anyWarningContains(noPhysics, "physics"), "AOI without physics is flagged");

    const auto noNet =
        B{}.serverMode(true).enableNetworking(false).interestRadius(lpl::math::Fixed32::fromFloat(50.0f)).build();
    check(anyWarningContains(noNet, "networking"), "AOI without networking is flagged");

    const auto badRings = B{}.serverMode(true)
                              .interestRadius(lpl::math::Fixed32::fromFloat(50.0f))
                              .lodNearRadius(lpl::math::Fixed32::fromFloat(60.0f))
                              .build();
    check(anyWarningContains(badRings, "near ring"), "a near ring wider than the interest radius is flagged");

    const auto badPrecision = B{}.serverMode(true)
                                  .interestRadius(lpl::math::Fixed32::fromFloat(50.0f))
                                  .worldExtent(lpl::math::Fixed32::fromFloat(1000.0f))
                                  .build();
    check(anyWarningContains(badPrecision, "quantization"), "precision LOD without network LOD is flagged");

    const auto clientAoi = B{}.serverMode(false).interestRadius(lpl::math::Fixed32::fromFloat(50.0f)).build();
    check(anyWarningContains(clientAoi, "client"), "server-side AOI on a client is flagged");

    const auto clean = B{}.serverMode(true).enablePhysics(true).enableNetworking(true).build();
    check(warningCount(clean) == 0, "a plain full-broadcast server is conflict-free");
}

/**
 * @brief Host profiles declare the SHAPE of the world.
 *
 * @details Which host streams is a contract, not a default that happens to hold: the parity smokes fold a BOUNDED
 *          world, and they fold it on the server profiles. While the shape was inferred from "is there a display",
 *          that held only by coincidence — so it is asserted here, where breaking it is a failing test rather than
 *          a moved signature three layers downstream.
 *
 * A budget with no switch was the previous state: maxResidentChunks meant nothing until something decided the world
 * was endless.
 */
void checkHostProfilesDeclareTheShapeOfTheWorld()
{
    using B = lpl::engine::Config::Builder;
    const auto streams = [](lpl::engine::HostProfile profile) {
        B builder;
        return lpl::engine::applyHostProfile(builder, profile).build().enableStreaming();
    };
    check(streams(lpl::engine::HostProfile::Ring0Client), "the kernel client streams, without a line of its own code");
    check(streams(lpl::engine::HostProfile::DesktopClient), "so does the desktop client");
    check(!streams(lpl::engine::HostProfile::Ring0Server), "the kernel server keeps a bounded world");
    check(!streams(lpl::engine::HostProfile::DedicatedServer), "and so does the dedicated server");
    check(!B{}.build().enableStreaming(), "a world is bounded unless something says otherwise");
    check(B{}.enableStreaming(true).maxResidentChunks(64u).build().maxResidentChunks() == 64u,
          "the switch and the budget travel together");
}

} // namespace

/**
 * @brief LplPlugin — game-profile presets + config validation test (book chapter 6).
 *
 * Proves two things: every genre preset (applyGameProfile) is internally
 * self-consistent — it raises zero configuration warnings — and the validator
 * (forEachConfigWarning) actually catches the contradictions it is meant to, so a
 * misconfiguration is loud instead of a silent null-deref or wasted bandwidth.
 */
int main()
{
    std::printf("== game profiles + config validation ==\n");

    checkEveryPresetIsConflictFree();

    checkTheMmorpgPresetTurnsTheScalingLeversOn();

    checkTheFpsPresetKeepsPrecisionAndReliabilityAndDropsLod();

    checkTheValidatorCatchesEachContradiction();

    checkHostProfilesDeclareTheShapeOfTheWorld();

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
