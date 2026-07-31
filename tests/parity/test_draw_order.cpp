/*
** EPITECH PROJECT, 2026
** LplPlugin
** File description:
** Draw-order oracle: the radix sort's ordering, stability and fold.
**
** A sort is the kind of code that looks obviously right and is quietly wrong on
** the case nobody wrote down. Two properties are checked here because the
** renderer depends on both: the keys come out non-decreasing (so materials batch
** and near draws precede far ones), and EQUAL keys keep their submission order
** (so the resulting stream is reproducible, and therefore foldable).
*/
#include <lpl/render/CommandBuffer.hpp>

#include <cstdio>

namespace {

int gFailures = 0;

void check(const char *what, bool condition)
{
    std::printf("  %-52s %s\n", what, condition ? "ok" : "FAIL");
    if (!condition)
        ++gFailures;
}

} // namespace

int main()
{
    std::printf("draw order oracle\n");

    // Field packing: material must dominate mesh, and mesh must dominate depth.
    check("material outranks mesh",
          lpl::render::packDrawKey(2u, 0u, 0xFFFFu) > lpl::render::packDrawKey(1u, 0xFFu, 0xFFFFu));
    check("mesh outranks depth",
          lpl::render::packDrawKey(1u, 2u, 0u) > lpl::render::packDrawKey(1u, 1u, 0xFFFFu));
    check("depth orders within a material and mesh",
          lpl::render::packDrawKey(1u, 1u, 10u) < lpl::render::packDrawKey(1u, 1u, 11u));

    // A deliberately adversarial list: reversed, with ties, spanning every byte
    // lane so all four passes have something to do.
    const lpl::core::u32 kCount = 512u;
    lpl::pmr::vector<lpl::render::DrawKey> keys;
    lpl::pmr::vector<lpl::render::DrawKey> scratch;
    keys.resize(kCount);
    scratch.resize(kCount);
    for (lpl::core::u32 i = 0u; i < kCount; ++i)
    {
        const lpl::core::u32 material = (kCount - i) % 5u;
        const lpl::core::u32 mesh = i % 3u;
        const lpl::core::u32 depth = ((kCount - i) * 127u) & 0xFFFFu;
        keys[i].key = lpl::render::packDrawKey(material, mesh, depth);
        keys[i].payload = i;
    }

    lpl::render::radixSortDrawKeys(&keys[0], &scratch[0], kCount);

    bool ordered = true;
    for (lpl::core::u32 i = 1u; i < kCount; ++i)
        ordered = ordered && keys[i - 1u].key <= keys[i].key;
    check("sorted non-decreasing", ordered);

    bool stable = true;
    for (lpl::core::u32 i = 1u; i < kCount; ++i)
        if (keys[i - 1u].key == keys[i].key)
            stable = stable && keys[i - 1u].payload < keys[i].payload;
    check("ties keep submission order (stable)", stable);

    // Every element is still present exactly once: a sort that loses a draw
    // would pass both checks above.
    lpl::pmr::vector<lpl::core::u32> seen;
    seen.resize(kCount, 0u);
    for (lpl::core::u32 i = 0u; i < kCount; ++i)
        if (keys[i].payload < kCount)
            ++seen[keys[i].payload];
    bool permutation = true;
    for (lpl::core::u32 i = 0u; i < kCount; ++i)
        permutation = permutation && seen[i] == 1u;
    check("output is a permutation of the input", permutation);

    // One element, and zero, must not corrupt anything.
    lpl::render::radixSortDrawKeys(&keys[0], &scratch[0], 1u);
    lpl::render::radixSortDrawKeys(&keys[0], &scratch[0], 0u);
    check("degenerate counts are safe", true);

    const lpl::core::u32 fold = lpl::render::foldDrawKeys(&keys[0], kCount);
    std::printf("\n  draw_order_fold = 0x%08X\n", fold);

    std::printf("\n%s (%d failures)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures);
    return gFailures == 0 ? 0 : 1;
}
