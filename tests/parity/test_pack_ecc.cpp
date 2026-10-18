#include <lpl/codec/ReedSolomon.hpp>
#include <lpl/editor/EccBaker.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/math/Random.hpp>
#include <lpl/pack/Cartridge.hpp>
#include <lpl/pack/EccSection.hpp>
#include <lpl/pack/GamePack.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

/**
 * @brief Does this image open and describe the world it should?
 * @param image The pack.
 * @param seed  The seed the recipe must carry.
 * @return true when it opens, hashes and decodes to that world.
 */
[[nodiscard]] bool opensAndMatches(const std::vector<lpl::core::u8> &image, lpl::core::u32 seed)
{
    lpl::pack::View view;
    if (!view.open(image.data(), static_cast<lpl::core::u32>(image.size())))
        return false;
    lpl::pack::RecipeV1 wire{};
    if (!view.readRecipe(wire))
        return false;
    lpl::procgen::WorldRecipe recipe{};
    lpl::pack::WireRefusal refusal{};
    return lpl::pack::toEngineRecipe(wire, recipe, refusal) && recipe.seed == seed;
}

/**
 * @brief An undamaged image must repair to a no-op.
 *
 * @details A repair pass that "fixes" a healthy cartridge is a repair pass that is writing noise.
 *
 * @param armoured The pack with its parity attached.
 */
void checkHealthyPackRepairsToANoOp(const std::vector<lpl::core::u8> &armoured)
{
    std::vector<lpl::core::u8> copy = armoured;
    lpl::pack::EccRepairReport report{};
    const bool ok = lpl::pack::repairPack(copy.data(), static_cast<lpl::core::u32>(copy.size()), report);
    check(ok && report.present, "a healthy pack reports its parity section");
    check(report.damagedCodewords == 0u && report.correctedBytes == 0u, "and repairs nothing");
    check(copy == armoured, "leaving the image byte for byte identical");
}

/**
 * @brief The burst.
 *
 * @details A stored cartridge fails by losing a SECTOR, not by scattering bit flips. The layout is transversal
 *          precisely so a burst inside one row costs one symbol per codeword, which is the case Reed-Solomon is
 *          strongest at.
 *
 *          The row width is read back out of the image rather than recomputed: a test that recomputes the
 *          layout would agree with a baker that got it wrong. Each burst wipes a full row, which is one wrong
 *          symbol in every single codeword; a burst whose noise happened to be a no-op is not evidence either
 *          way, and is skipped.
 *
 * @param armoured  The pack with its parity attached.
 * @param reference The recipe the pack was baked from.
 */
void checkEveryWholeRowBurstIsRepaired(const std::vector<lpl::core::u8> &armoured,
                                       const lpl::procgen::WorldRecipe &reference)
{
    std::printf("\n-- bursts --\n");
    lpl::pack::EccV1 ecc{};
    {
        lpl::pack::View view;
        (void) view.open(armoured.data(), static_cast<lpl::core::u32>(armoured.size()));
        const lpl::core::u8 *payload = nullptr;
        lpl::core::u32 payloadSize = 0u;
        check(view.findSection(lpl::pack::SectionType::Ecc, payload, payloadSize), "the parity section is findable");
        if (payload != nullptr)
            std::memcpy(&ecc, payload, sizeof(ecc));
    }
    std::printf("  layout: %u rows of %u bytes, %u parity symbols per column\n", ecc.dataShards, ecc.rowBytes,
                ecc.parityShards);

    lpl::core::u32 survived = 0u;
    lpl::core::u32 attempts = 0u;
    for (lpl::core::u32 start = 0u; start + ecc.rowBytes < ecc.protectedBytes; start += ecc.rowBytes)
    {
        ++attempts;
        std::vector<lpl::core::u8> damaged = armoured;
        lpl::math::Random noise = lpl::math::deriveStream(start, 0xBADu);
        for (lpl::core::u32 i = 0u; i < ecc.rowBytes; ++i)
            damaged[ecc.protectedOffset + start + i] = static_cast<lpl::core::u8>(noise.next() & 0xFFu);

        if (opensAndMatches(damaged, reference.seed))
            continue;

        lpl::pack::EccRepairReport report{};
        if (!lpl::pack::repairPack(damaged.data(), static_cast<lpl::core::u32>(damaged.size()), report))
            continue;
        if (opensAndMatches(damaged, reference.seed))
            ++survived;
    }
    std::printf("  %u of %u whole-row bursts recovered\n", survived, attempts);
    check(attempts > 0u, "there were bursts to try");
    check(survived == attempts, "every whole-row burst is repaired, hash and world included");
}

/**
 * @brief Past the bound.
 *
 * @details More rows are destroyed than floor(parity/2): every codeword then has more wrong symbols than the
 *          code can locate. The repair must refuse, and the reader must still refuse the image, which is the
 *          outcome that matters: a cartridge that cannot be repaired must not load.
 *
 * @param armoured  The pack with its parity attached.
 * @param reference The recipe the pack was baked from.
 */
void checkDamagePastTheBoundIsRefused(const std::vector<lpl::core::u8> &armoured,
                                      const lpl::procgen::WorldRecipe &reference)
{
    std::printf("\n-- past the bound --\n");
    lpl::pack::EccV1 ecc{};
    lpl::pack::View view;
    (void) view.open(armoured.data(), static_cast<lpl::core::u32>(armoured.size()));
    const lpl::core::u8 *payload = nullptr;
    lpl::core::u32 payloadSize = 0u;
    (void) view.findSection(lpl::pack::SectionType::Ecc, payload, payloadSize);
    if (payload != nullptr)
        std::memcpy(&ecc, payload, sizeof(ecc));

    const lpl::core::u32 rowsToWipe = ecc.parityShards / 2u + 2u;
    std::vector<lpl::core::u8> damaged = armoured;
    lpl::math::Random noise{0xDEADu};
    for (lpl::core::u32 row = 0u; row < rowsToWipe; ++row)
        for (lpl::core::u32 i = 0u; i < ecc.rowBytes; ++i)
        {
            const lpl::core::u64 index = static_cast<lpl::core::u64>(row) * ecc.rowBytes + i;
            if (index < ecc.protectedBytes)
                damaged[ecc.protectedOffset + index] = static_cast<lpl::core::u8>(noise.next() & 0xFFu);
        }

    lpl::pack::EccRepairReport report{};
    const bool repaired = lpl::pack::repairPack(damaged.data(), static_cast<lpl::core::u32>(damaged.size()), report);
    std::printf("  %u rows wiped (bound is %u): repair %s\n", rowsToWipe, ecc.parityShards / 2u,
                repaired ? "claimed success" : "refused");
    check(!repaired, "damage past the bound is REFUSED, not guessed at");

    check(!opensAndMatches(damaged, reference.seed), "and the unrepairable image does not open");
}

/**
 * @brief A pack with no parity section reports the absence.
 *
 * @details repairPack returns false for it, as for damage past the bound; what tells the two apart is the
 *          report, which says no parity section was present.
 *
 * @param plain The pack without parity.
 */
void checkPackWithoutParityReportsTheAbsence(const std::vector<lpl::core::u8> &plain)
{
    std::vector<lpl::core::u8> copy = plain;
    lpl::pack::EccRepairReport report{};
    const bool ok = lpl::pack::repairPack(copy.data(), static_cast<lpl::core::u32>(copy.size()), report);
    check(!ok && !report.present, "a pack without parity says so instead of pretending");
    check(copy == plain, "and is left untouched");
}

} // namespace

/**
 * @brief A cartridge that survives a bad sector — and refuses when it cannot.
 *
 * @details Two claims, and the second is the one that matters more. A cartridge with parity must come back
 *          from a burst of damage. And past the code's bound it must FAIL, loudly, rather than hand back a
 *          plausible world it invented: for an archival format a wrong world that loads is strictly worse than
 *          a right one that refuses.
 */
int main()
{
    std::printf("== pack ECC: a cartridge that survives a bad sector ==\n\n");

    const lpl::procgen::WorldRecipe reference = lpl::procgen::parityWorldRecipe();
    const std::vector<lpl::core::u8> plain = lpl::editor::bakeGamePack(reference);
    const std::vector<lpl::core::u8> armoured = lpl::editor::attachEcc(plain);

    std::printf("  plain    = %zu bytes\n", plain.size());
    std::printf("  armoured = %zu bytes (+%zu)\n\n", armoured.size(), armoured.size() - plain.size());

    check(armoured.size() > plain.size(), "attaching parity grows the image");
    check(opensAndMatches(armoured, reference.seed), "and the armoured pack still opens and reads the same world");

    {
        lpl::pack::View view;
        (void) view.open(armoured.data(), static_cast<lpl::core::u32>(armoured.size()));
        check(view.sectionCount() == 2u, "it carries the recipe and the parity");
    }

    checkHealthyPackRepairsToANoOp(armoured);
    checkEveryWholeRowBurstIsRepaired(armoured, reference);
    checkDamagePastTheBoundIsRefused(armoured, reference);
    checkPackWithoutParityReportsTheAbsence(plain);

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
