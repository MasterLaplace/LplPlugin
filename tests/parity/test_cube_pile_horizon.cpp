#include <lpl/core/Types.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/samples/CubePile.hpp>

#include <cstdio>

namespace {

/** @brief Ticks between two folds. */
constexpr lpl::core::u32 kTicksBetweenFolds = 60u;

/**
 * @brief The pile's state folded every 60 ticks, from tick 60 to tick 600, by the solver of sequential
 *        impulses (LplPlugin#560).
 */
constexpr lpl::core::u32 kRecordedFolds[] = {0x55824A16u, 0x81D4D152u, 0xF37B7C13u, 0xF37B7C13u, 0xF37B7C13u,
                                             0xF37B7C13u, 0xF37B7C13u, 0xF37B7C13u, 0xF37B7C13u, 0xF37B7C13u};

/** @brief The fold by which the pile has settled and sleeps: tick 180. */
constexpr lpl::core::u32 kSettledFold = 2u;

} // namespace

/**
 * @brief Runs the cube pile 600 ticks, its fall, its landing and its settling, and compares the fold
 *        of its state every 60 ticks with the words the solver of sequential impulses gives.
 *
 * The engine test `cube_pile` folds the pile at tick 64, which a kernel boot can afford, and the
 * first 60 ticks are mostly a fall: a change to the broad-phase or the solver that alters a contact
 * only once the pile has landed would pass it. A change that claims to alter nothing must keep all
 * ten folds here.
 *
 * Until the pile settles, each fold must differ from the one before it, or the run compares a pile
 * that never moved. Once it has settled, every cube sleeps and every later fold must be the same:
 * before the solver kept its contacts through its iterations, no cube of the pile ever slept, and
 * the solver spent most of the server's tick on a pile at rest.
 */
int main()
{
    lpl::ecs::Registry registry;
    lpl::samples::CubePile pile{registry};
    int failures = 0;
    lpl::core::u32 previous = 0u;
    lpl::core::u32 unchanged = 0u;
    lpl::core::u32 changed = 0u;

    pile.init();
    for (lpl::core::u32 fold = 0u; fold < sizeof(kRecordedFolds) / sizeof(kRecordedFolds[0]); ++fold)
    {
        for (lpl::core::u32 tick = 0u; tick < kTicksBetweenFolds; ++tick)
            pile.step();

        const lpl::core::u32 signature = pile.stateSignature();
        const bool recorded = signature == kRecordedFolds[fold];
        std::printf("  %s tick %u: state 0x%08X", recorded ? "PASS" : "FAIL", (fold + 1u) * kTicksBetweenFolds,
                    signature);
        if (!recorded)
        {
            std::printf(", recorded 0x%08X", kRecordedFolds[fold]);
            ++failures;
        }
        std::printf("\n");
        if (fold != 0u && fold <= kSettledFold && signature == previous)
            ++unchanged;
        if (fold > kSettledFold && signature != previous)
            ++changed;
        previous = signature;
    }

    const bool moving = unchanged == 0u;
    std::printf("  %s every fold until tick 180 differs from the one before it\n", moving ? "PASS" : "FAIL");
    if (!moving)
        ++failures;

    const bool asleep = changed == 0u;
    std::printf("  %s every fold after tick 180 is the same: the pile sleeps\n", asleep ? "PASS" : "FAIL");
    if (!asleep)
        ++failures;

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
