/**
 * @file test_pack_ecc.cpp
 * @brief A cartridge that survives a bad sector — and refuses when it cannot.
 *
 * Two claims, and the second is the one that matters more. A cartridge with parity
 * must come back from a burst of damage. And past the code's bound it must FAIL,
 * loudly, rather than hand back a plausible world it invented: for an archival format
 * a wrong world that loads is strictly worse than a right one that refuses.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

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
    return lpl::pack::toEngineRecipe(wire).seed == seed;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== pack ECC: a cartridge that survives a bad sector ==\n\n");

    const procgen::WorldRecipe reference = procgen::parityWorldRecipe();
    const std::vector<core::u8> plain = editor::bakeGamePack(reference);
    const std::vector<core::u8> armoured = editor::attachEcc(plain);

    std::printf("  plain    = %zu bytes\n", plain.size());
    std::printf("  armoured = %zu bytes (+%zu)\n\n", armoured.size(), armoured.size() - plain.size());

    check(armoured.size() > plain.size(), "attaching parity grows the image");
    check(opensAndMatches(armoured, reference.seed), "and the armoured pack still opens and reads the same world");

    {
        pack::View view;
        (void) view.open(armoured.data(), static_cast<core::u32>(armoured.size()));
        check(view.sectionCount() == 2u, "it carries the recipe and the parity");
    }

    // An undamaged image must repair to a no-op. A repair pass that "fixes" a healthy
    // cartridge is a repair pass that is writing noise.
    {
        std::vector<core::u8> copy = armoured;
        pack::EccRepairReport report{};
        const bool ok = pack::repairPack(copy.data(), static_cast<core::u32>(copy.size()), report);
        check(ok && report.present, "a healthy pack reports its parity section");
        check(report.damagedCodewords == 0u && report.correctedBytes == 0u, "and repairs nothing");
        check(copy == armoured, "leaving the image byte for byte identical");
    }

    // ── The burst ─────────────────────────────────────────────────────────────
    //
    // A stored cartridge fails by losing a SECTOR, not by scattering bit flips. The
    // layout is transversal precisely so a burst inside one row costs one symbol per
    // codeword, which is the case Reed-Solomon is strongest at.
    std::printf("\n-- bursts --\n");
    {
        pack::EccV1 ecc{};
        {
            // Read the row width back out of the image rather than recomputing it: a
            // test that recomputes the layout would agree with a baker that got it
            // wrong.
            pack::View view;
            (void) view.open(armoured.data(), static_cast<core::u32>(armoured.size()));
            const core::u8 *payload = nullptr;
            core::u32 payloadSize = 0u;
            check(view.findSection(pack::SectionType::Ecc, payload, payloadSize), "the parity section is findable");
            if (payload != nullptr)
                std::memcpy(&ecc, payload, sizeof(ecc));
        }
        std::printf("  layout: %u rows of %u bytes, %u parity symbols per column\n", ecc.dataShards, ecc.rowBytes,
                    ecc.parityShards);

        core::u32 survived = 0u;
        core::u32 attempts = 0u;
        for (core::u32 start = 0u; start + ecc.rowBytes < ecc.protectedBytes; start += ecc.rowBytes)
        {
            ++attempts;
            std::vector<core::u8> damaged = armoured;
            math::Random noise = math::deriveStream(start, 0xBADu);
            // A full row, wiped. That is one wrong symbol in every single codeword.
            for (core::u32 i = 0u; i < ecc.rowBytes; ++i)
                damaged[ecc.protectedOffset + start + i] = static_cast<core::u8>(noise.next() & 0xFFu);

            if (opensAndMatches(damaged, reference.seed))
                continue; // the noise happened to be a no-op; not evidence either way

            pack::EccRepairReport report{};
            if (!pack::repairPack(damaged.data(), static_cast<core::u32>(damaged.size()), report))
                continue;
            if (opensAndMatches(damaged, reference.seed))
                ++survived;
        }
        std::printf("  %u of %u whole-row bursts recovered\n", survived, attempts);
        check(attempts > 0u, "there were bursts to try");
        check(survived == attempts, "every whole-row burst is repaired, hash and world included");
    }

    // ── Past the bound ────────────────────────────────────────────────────────
    std::printf("\n-- past the bound --\n");
    {
        pack::EccV1 ecc{};
        pack::View view;
        (void) view.open(armoured.data(), static_cast<core::u32>(armoured.size()));
        const core::u8 *payload = nullptr;
        core::u32 payloadSize = 0u;
        (void) view.findSection(pack::SectionType::Ecc, payload, payloadSize);
        if (payload != nullptr)
            std::memcpy(&ecc, payload, sizeof(ecc));

        // More rows destroyed than floor(parity/2). Every codeword now has more wrong
        // symbols than the code can locate.
        const core::u32 rowsToWipe = ecc.parityShards / 2u + 2u;
        std::vector<core::u8> damaged = armoured;
        math::Random noise{0xDEADu};
        for (core::u32 row = 0u; row < rowsToWipe; ++row)
            for (core::u32 i = 0u; i < ecc.rowBytes; ++i)
            {
                const core::u64 index = static_cast<core::u64>(row) * ecc.rowBytes + i;
                if (index < ecc.protectedBytes)
                    damaged[ecc.protectedOffset + index] = static_cast<core::u8>(noise.next() & 0xFFu);
            }

        pack::EccRepairReport report{};
        const bool repaired = pack::repairPack(damaged.data(), static_cast<core::u32>(damaged.size()), report);
        std::printf("  %u rows wiped (bound is %u): repair %s\n", rowsToWipe, ecc.parityShards / 2u,
                    repaired ? "claimed success" : "refused");
        check(!repaired, "damage past the bound is REFUSED, not guessed at");

        // And the reader still refuses the image, which is the outcome that matters:
        // a cartridge that cannot be repaired must not load.
        check(!opensAndMatches(damaged, reference.seed), "and the unrepairable image does not open");
    }

    // A pack with no parity section reports the absence rather than failing.
    {
        std::vector<core::u8> copy = plain;
        pack::EccRepairReport report{};
        const bool ok = pack::repairPack(copy.data(), static_cast<core::u32>(copy.size()), report);
        check(!ok && !report.present, "a pack without parity says so instead of pretending");
        check(copy == plain, "and is left untouched");
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
