/**
 * @file test_cubepile_parity.cpp
 * @brief Parity oracle for the kernel-mode CubePile sample simulation.
 *
 * Seeds a deterministic N-entity scene (Fixed32 authoritative position/velocity),
 * advances a fixed number of gravity+bounce ticks, rasterizes every entity as a
 * depth-buffered cube, and folds both the authoritative state and the rendered
 * image. These signatures are the cross-target authority the in-kernel sim
 * smoke must reproduce bit-for-bit.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-29
 * @copyright MIT License
 */

#include <cstdio>
#include <vector>

#include <lpl/samples/CubePile.hpp>

using namespace lpl;

static int failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

int main()
{
    std::printf("== cube-pile sample parity ==\n");

    constexpr core::u32 W = 192u;
    constexpr core::u32 H = 120u;
    std::vector<core::u32> color(W * H, 0u);
    std::vector<core::f32> depth(W * H, 0.0f);
    render::RenderTarget rt{color.data(), depth.data(), W, H};

    const auto a = samples::runCubePileAndFold(rt, 64u);
    const auto b = samples::runCubePileAndFold(rt, 64u);
    check(a.state_signature == b.state_signature, "state fold is deterministic");
    check(a.image_signature == b.image_signature, "image fold is deterministic");

    // Different tick counts must yield a different authoritative state.
    const auto early = samples::runCubePileAndFold(rt, 8u);
    check(early.state_signature != a.state_signature, "state evolves with ticks");
    check(a.image_signature != render::detail::kFnv1aOffsetBasis, "rendered image non-trivial");

    std::printf("\n== signatures (must match kernel sim fold) ==\n");
    std::printf("  ticks=8   state_sig = 0x%08X image_sig = 0x%08X\n", early.state_signature, early.image_signature);
    std::printf("  ticks=64  state_sig = 0x%08X image_sig = 0x%08X\n", a.state_signature, a.image_signature);

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
