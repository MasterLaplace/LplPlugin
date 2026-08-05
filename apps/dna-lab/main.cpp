/**
 * @file main.cpp
 * @brief Simulate a millennium of molecular decay in a few seconds.
 *
 * Every nucleotide that does not have to be synthesised is money saved, linearly,
 * so density and redundancy are the two levers worth pulling — and neither can be
 * tuned without a way to measure survival. This is that instrument: a thousand
 * years of strand breaks, substitutions and dropout, applied in silico to an
 * encoded corpus, with no chemistry and no cost.
 *
 * The curve it prints is reproducible from a seed, which is the only property that
 * makes it an instrument rather than an anecdote.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/codec/Erasure.hpp>
#include <lpl/codec/Fountain.hpp>
#include <lpl/codec/Prng.hpp>
#include <lpl/codec/ReedSolomon.hpp>
#include <lpl/math/Random.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

/**
 * @brief Builds a corpus of @p bytes, reproducibly.
 * @param bytes How much.
 * @param seed  Determinism anchor.
 * @param out   Receives the corpus.
 */
void buildCorpus(lpl::core::u32 bytes, lpl::core::u32 seed, lpl::pmr::vector<lpl::core::u8> &out)
{
    out.clear();
    out.resize(bytes, lpl::core::u8{0});
    lpl::math::Random stream{seed};
    for (lpl::core::u32 i = 0u; i < bytes; ++i)
        out[i] = static_cast<lpl::core::u8>(stream.next() & 0xFFu);
}

/**
 * @brief Are two byte sequences identical?
 * @param a First.
 * @param b Second.
 * @return true when they match exactly.
 */
[[nodiscard]] bool identical(const lpl::pmr::vector<lpl::core::u8> &a, const lpl::pmr::vector<lpl::core::u8> &b)
{
    if (a.size() != b.size())
        return false;
    for (lpl::core::usize i = 0u; i < a.size(); ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

/**
 * @brief Appends Reed-Solomon parity to every strand's payload.
 *
 * The INNER code of the two-level scheme (SIM-076). The fountain between strands
 * answers a strand VANISHING; nothing in it answers a strand arriving WRONG, because
 * a wrong strand is indistinguishable from a right one until something checks. This
 * is that something, and it lives inside the strand because that is where the
 * substitution happens.
 *
 * @param pool        Strands, each payload grown by @p parityBytes.
 * @param parityBytes Parity symbols per strand.
 */
void armourStrands(lpl::pmr::vector<lpl::codec::Droplet> &pool, lpl::core::u32 parityBytes)
{
    for (lpl::core::usize i = 0u; i < pool.size(); ++i)
    {
        lpl::core::u8 parity[lpl::codec::kMaxParitySymbols]{};
        const lpl::core::u32 dataCount = static_cast<lpl::core::u32>(pool[i].payload.size());
        if (!lpl::codec::reedSolomonEncode(pool[i].payload.data(), dataCount, parityBytes, parity))
            continue;
        for (lpl::core::u32 p = 0u; p < parityBytes; ++p)
            pool[i].payload.push_back(parity[p]);
    }
}

/**
 * @brief Corrects each strand, and discards the ones past the code's bound.
 *
 * A strand that cannot be corrected is DROPPED rather than passed on. That turns an
 * error the fountain cannot see into an erasure it can: the payload is simply absent,
 * and the fountain replaces it with the next strand. Converting one failure mode into
 * the other is the whole reason the scheme has two levels.
 *
 * @param pool        Strands, stripped back to their payload on success.
 * @param parityBytes Parity symbols each strand carries.
 * @param outFixed    Receives strands that needed correction and got it.
 * @param outDropped  Receives strands that were beyond repair.
 */
void healStrands(lpl::pmr::vector<lpl::codec::Droplet> &pool, lpl::core::u32 parityBytes, lpl::core::u32 &outFixed,
                 lpl::core::u32 &outDropped)
{
    outFixed = 0u;
    outDropped = 0u;

    lpl::pmr::vector<lpl::codec::Droplet> healthy;
    for (lpl::core::usize i = 0u; i < pool.size(); ++i)
    {
        lpl::codec::ReedSolomonRepair repair{};
        const lpl::core::u32 symbols = static_cast<lpl::core::u32>(pool[i].payload.size());
        if (!lpl::codec::reedSolomonCorrect(pool[i].payload.data(), symbols, parityBytes, repair))
        {
            ++outDropped;
            continue;
        }
        if (!repair.clean)
            ++outFixed;

        lpl::codec::Droplet stripped;
        stripped.seed = pool[i].seed;
        for (lpl::core::u32 b = 0u; b + parityBytes < symbols; ++b)
            stripped.payload.push_back(pool[i].payload[b]);
        healthy.push_back(stripped);
    }

    pool.clear();
    for (lpl::core::usize i = 0u; i < healthy.size(); ++i)
        pool.push_back(healthy[i]);
}

} // namespace

int main(int argc, char **argv)
{
    using namespace lpl;

    core::u32 corpusBytes = 4096u;
    core::u32 seed = 20260804u;
    core::u32 overheadPermille = 800u;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        if (std::strcmp(argv[i], "--bytes") == 0)
            corpusBytes = static_cast<core::u32>(std::strtoul(argv[i + 1], nullptr, 10));
        else if (std::strcmp(argv[i], "--seed") == 0)
            seed = static_cast<core::u32>(std::strtoul(argv[i + 1], nullptr, 10));
        else if (std::strcmp(argv[i], "--overhead") == 0)
            overheadPermille = static_cast<core::u32>(std::strtoul(argv[i + 1], nullptr, 10));
    }

    lpl::pmr::vector<core::u8> corpus;
    buildCorpus(corpusBytes, seed, corpus);

    codec::ErasureParams params;
    params.blockBytes = 32u;
    params.overheadPermille = overheadPermille;
    params.firstSeed = seed;

    codec::ErasureShape shape{};
    lpl::pmr::vector<codec::Droplet> pristine;
    if (!codec::encodeErasure(corpus.data(), static_cast<core::u32>(corpus.size()), params, shape, pristine))
    {
        std::fprintf(stderr, "lpl-dna-lab: the corpus could not be encoded\n");
        return 1;
    }

    // A droplet carries a 32-bit seed, not the list of source blocks it combines: the
    // decoder re-runs the generator to rebuild that list. Storing the generator rather
    // than the result — the same move as a world recipe or a command journal, and the
    // reason a strand's header is negligible next to its payload.
    const core::u32 headerBits = 32u;
    const core::u32 payloadBits = params.blockBytes * 8u;
    const core::u32 basesPerStrand = (headerBits + payloadBits) / 2u;
    const core::u64 totalBases = static_cast<core::u64>(pristine.size()) * basesPerStrand;
    const core::u64 corpusBits = static_cast<core::u64>(corpusBytes) * 8u;

    std::printf("== lpl-dna-lab: a millennium, in silico ==\n\n");
    std::printf("  corpus     : %u bytes, K = %u blocks of %u\n", corpusBytes, shape.blockCount, shape.blockBytes);
    std::printf("  strands    : %zu at %u bases each\n", pristine.size(), basesPerStrand);
    std::printf("  density    : %llu bits over %llu bases = %u/1000 bits per nucleotide\n",
                static_cast<unsigned long long>(corpusBits), static_cast<unsigned long long>(totalBases),
                totalBases == 0u ? 0u : static_cast<core::u32>((corpusBits * 1000u) / totalBases));
    std::printf("  (the ceiling is 2000/1000: four symbols carry two bits, and no code beats that)\n\n");

    // How much of the fountain the biological filter throws away, measured rather than
    // assumed. Rejecting a droplet costs nothing in bits — the next one is already
    // available — which is the whole trick, and this is what it costs in effort.
    {
        lpl::pmr::vector<core::u8> padded;
        padded.resize(static_cast<core::usize>(shape.blockCount) * shape.blockBytes, core::u8{0});
        for (core::u32 i = 0u; i < corpusBytes && i < padded.size(); ++i)
            padded[i] = corpus[i];

        codec::SourceView source;
        source.bytes = padded.data();
        source.blockBytes = shape.blockBytes;
        source.blockCount = shape.blockCount;
        const codec::Fountain fountain{source, params.tuning};

        codec::BiologicalLimits limits;
        lpl::pmr::vector<codec::Droplet> valid;
        const core::u32 examined = fountain.emitValid(shape.blockCount, limits, params.firstSeed, valid);
        std::printf("  in-silico filter: %zu synthesisable strands out of %u examined",
                    valid.size(), examined);
        if (examined != 0u)
            std::printf(" (%u%% kept)", static_cast<core::u32>((valid.size() * 100u) / examined));
        std::printf("\n  GC within [%u,%u] permille, homopolymer runs at most %u\n\n", limits.minGcPermille,
                    limits.maxGcPermille, limits.maxHomopolymer);
    }

    // The same experiment twice: once with the fountain alone, once with a
    // Reed-Solomon code inside each strand as well. Side by side, because the whole
    // claim of the two-level scheme is a difference between two columns.
    const codec::DecayParams decay;
    for (core::u32 innerParity : {0u, 4u})
    {
        std::printf("  %s\n", innerParity == 0u ? "-- fountain only --"
                                                 : "-- fountain, plus Reed-Solomon inside each strand --");
        std::printf("  years   strands  intact  lost  substituted  fixed  dropped  payload\n");
        std::printf("  ------  -------  ------  ----  -----------  -----  -------  -------\n");

        for (core::u32 years : {0u, 100u, 500u, 1000u, 2000u, 5000u, 10000u})
        {
            lpl::pmr::vector<codec::Droplet> aged;
            for (core::usize i = 0u; i < pristine.size(); ++i)
            {
                codec::Droplet copy;
                copy.seed = pristine[i].seed;
                copy.payload = pristine[i].payload;
                aged.push_back(copy);
            }
            if (innerParity != 0u)
                armourStrands(aged, innerParity);

            // The stream is seeded from the SPAN, so each row of the table is its own
            // reproducible experiment rather than a continuation of the row above it.
            math::Random stream = math::deriveStream(seed, years + 1u);
            codec::DecayReport report{};
            codec::simulateDecay(aged, years, decay, stream, report);

            core::u32 fixed = 0u;
            core::u32 dropped = 0u;
            if (innerParity != 0u)
                healStrands(aged, innerParity, fixed, dropped);

            lpl::pmr::vector<core::u8> recovered;
            codec::DecodeReport decoded{};
            const bool ok = codec::decodeErasure(aged, shape, params, recovered, decoded);
            const bool exact = ok && identical(recovered, corpus);

            std::printf("  %6u  %7u  %6u  %4u  %11u  %5u  %7u  %s\n", years, report.strands, report.intact,
                        report.lost, report.substitutions, fixed, dropped,
                        exact ? "RECOVERED" : (ok ? "wrong" : "LOST"));
        }
        std::printf("\n");
    }

    std::printf("\n  Two failure modes, two codes. A strand that breaks or is never sequenced is an\n");
    std::printf("  ERASURE: it is simply absent, and the fountain replaces it. A base read as a\n");
    std::printf("  different base is an ERROR: the strand arrives, looks fine, and is wrong. The\n");
    std::printf("  fountain cannot see that one — only Reed-Solomon inside the strand can, which is\n");
    std::printf("  why the archival scheme is two-level and why the substituted column above is the\n");
    std::printf("  one that eventually costs the payload.\n");
    return 0;
}
