#include <lpl/core/Types.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/samples/CubePile.hpp>

#include <cstdio>

namespace {

/** @brief Ticks between two folds. */
constexpr lpl::core::u32 kTicksBetweenFolds = 60u;

/** @brief The pile's state folded every 60 ticks, from tick 60 to tick 600, by LplPlugin 0.7.0. */
constexpr lpl::core::u32 kRecordedFolds[] = {0x2BD7F83Du, 0xC70DD28Du, 0xBBFF9C59u, 0x8CAAEFEAu, 0xD76E17EEu,
                                             0x008D47D9u, 0x8FB8FE48u, 0xC68893D2u, 0x834F9AABu, 0xDA40CD08u};

} // namespace

/**
 * @brief Runs the cube pile 600 ticks, its fall, its landing and its settling, and compares the fold
 *        of its state every 60 ticks with the words LplPlugin 0.7.0 gives.
 *
 * The engine test `cube_pile` folds the pile at tick 64, which a kernel boot can afford, and the
 * first 60 ticks are a free fall in which no two cubes touch: a change to the broad-phase or the
 * solver that alters a contact only once the pile has landed would pass it. A change that claims
 * to alter nothing must keep all ten folds here. Each fold must also differ from the one before
 * it, or the pile has stopped moving and the later folds compare nothing.
 */
int main()
{
    lpl::ecs::Registry registry;
    lpl::samples::CubePile pile{registry};
    int failures = 0;
    lpl::core::u32 previous = 0u;
    lpl::core::u32 unchanged = 0u;

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
        if (fold != 0u && signature == previous)
            ++unchanged;
        previous = signature;
    }

    const bool moving = unchanged == 0u;
    std::printf("  %s every fold differs from the one before it\n", moving ? "PASS" : "FAIL");
    if (!moving)
        ++failures;

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
