/**
 * @file test_erasure_channel.cpp
 * @brief What "rateless" actually claims, asserted rather than repeated.
 *
 * A fountain's selling point is not that it survives loss — a fixed-rate code does
 * too, up to its rate. It is that it does not care WHICH droplets were lost, only how
 * many arrived. That is a testable statement and it is what this file tests.
 *
 * Every claim here is comparative or monotone. No absolute threshold: the decode
 * probability of an LT code at small K is a distribution, and a fixed number chosen
 * so that today's tuning passes is the mistake this repository has paid for most.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/Erasure.hpp>
#include <lpl/codec/Fountain.hpp>
#include <lpl/math/Random.hpp>

#include <cstdio>

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
 * @brief How droplets go missing.
 */
enum class LossShape : lpl::core::u32 {
    Prefix = 0u, ///< The first ones never arrived.
    Suffix = 1u, ///< The transfer was cut off.
    Stride = 2u, ///< One in N dropped, evenly.
    Random = 3u, ///< Scattered.
};

/**
 * @brief Builds a corpus of @p bytes from @p seed.
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
 * @brief Decodes a corpus after losing droplets in a given shape.
 * @param shape       How the loss is distributed.
 * @param keep        Droplets to deliver.
 * @param trialSeed   Which trial this is.
 * @return true when the corpus came back byte for byte.
 */
[[nodiscard]] bool survives(LossShape shape, lpl::core::u32 keep, lpl::core::u32 trialSeed)
{
    using namespace lpl;

    lpl::pmr::vector<core::u8> corpus;
    buildCorpus(1024u, 0xC0FFEEu + trialSeed, corpus);

    codec::ErasureParams params;
    params.blockBytes = 32u;
    params.overheadPermille = 2000u; // emit plenty; the test decides what ARRIVES
    params.firstSeed = 0x5EEDu + trialSeed * 7919u;

    codec::ErasureShape form{};
    lpl::pmr::vector<codec::Droplet> emitted;
    if (!codec::encodeErasure(corpus.data(), static_cast<core::u32>(corpus.size()), params, form, emitted))
        return false;
    if (keep > emitted.size())
        return false;

    // An ARRIVAL ORDER per shape, then the first `keep` of it. Building the order
    // first is what makes the shapes comparable: every one of them then delivers
    // exactly the same NUMBER of droplets, which is the variable the claim is about.
    //
    // The first version picked indices by arithmetic instead, and `(i * 3) % total`
    // can only ever name total/3 distinct droplets — so every shape past that count
    // "failed" for want of droplets the harness never fetched. The codec was fine; the
    // measurement was not.
    const core::u32 total = static_cast<core::u32>(emitted.size());
    lpl::pmr::vector<core::u32> order;
    switch (shape)
    {
    case LossShape::Prefix:
        for (core::u32 i = 0u; i < total; ++i)
            order.push_back(total - 1u - i);
        break;
    case LossShape::Suffix:
        for (core::u32 i = 0u; i < total; ++i)
            order.push_back(i);
        break;
    case LossShape::Stride:
        for (core::u32 i = 0u; i < total; ++i)
            if (i % 3u != 0u)
                order.push_back(i);
        for (core::u32 i = 0u; i < total; ++i)
            if (i % 3u == 0u)
                order.push_back(i);
        break;
    case LossShape::Random: {
        for (core::u32 i = 0u; i < total; ++i)
            order.push_back(i);
        math::Random pick = math::deriveStream(trialSeed, 0x1055u);
        for (core::u32 i = total; i > 1u; --i)
        {
            const core::u32 j = pick.below(i);
            const core::u32 held = order[i - 1u];
            order[i - 1u] = order[j];
            order[j] = held;
        }
        break;
    }
    }

    lpl::pmr::vector<codec::Droplet> delivered;
    for (core::u32 i = 0u; i < keep && i < order.size(); ++i)
    {
        codec::Droplet copy;
        copy.seed = emitted[order[i]].seed;
        copy.payload = emitted[order[i]].payload;
        delivered.push_back(copy);
    }

    if (delivered.size() < keep)
        return false;

    lpl::pmr::vector<core::u8> recovered;
    codec::DecodeReport report{};
    if (!codec::decodeErasure(delivered, form, params, recovered, report))
        return false;
    if (recovered.size() != corpus.size())
        return false;
    for (core::usize i = 0u; i < corpus.size(); ++i)
        if (recovered[i] != corpus[i])
            return false;
    return true;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== erasure channel: what rateless actually claims ==\n\n");

    // K = 1024/32 = 32 blocks.
    constexpr core::u32 kBlocks = 32u;
    constexpr core::u32 kTrials = 32u;

    // ── Claim 1: more arrivals never decode less often ────────────────────────
    //
    // Monotone rather than a threshold. The decode probability of an LT code at K = 32
    // is a distribution, and pinning a number to it would be pinning today's tuning.
    std::printf("-- delivering more never helps less --\n");
    core::u32 previous = 0u;
    bool monotone = true;
    for (core::u32 keep : {kBlocks, kBlocks + 4u, kBlocks + 8u, kBlocks + 16u, kBlocks + 32u})
    {
        core::u32 ok = 0u;
        for (core::u32 t = 0u; t < kTrials; ++t)
            ok += survives(LossShape::Stride, keep, t) ? 1u : 0u;
        std::printf("    %2u droplets for %u blocks: %2u/%u decoded\n", keep, kBlocks, ok, kTrials);
        monotone = monotone && ok >= previous;
        previous = ok;
    }
    check(monotone, "the success rate never falls as more droplets arrive");
    check(previous == kTrials, "and with enough of them every trial decodes");

    // ── Claim 2: WHICH ones were lost does not matter ─────────────────────────
    //
    // This is the claim that separates a fountain from a fixed-rate code, and it is
    // the reason a fountain is the right answer for a medium whose losses are not
    // independent — a bad sector, a cut transfer, a strand that never sequenced.
    std::printf("\n-- and it does not matter WHICH ones were lost --\n");
    const core::u32 keep = kBlocks + 32u;
    core::u32 perShape[4]{};
    const char *names[4] = {"first ones lost", "last ones lost", "one in three lost", "scattered"};
    for (core::u32 s = 0u; s < 4u; ++s)
    {
        for (core::u32 t = 0u; t < kTrials; ++t)
            perShape[s] += survives(static_cast<LossShape>(s), keep, t) ? 1u : 0u;
        std::printf("    %-20s %2u/%u decoded\n", names[s], perShape[s], kTrials);
    }
    bool shapeBlind = true;
    for (core::u32 s = 0u; s < 4u; ++s)
        shapeBlind = shapeBlind && perShape[s] == kTrials;
    check(shapeBlind, "every loss shape decodes, given the same NUMBER of arrivals");

    // ── Claim 3: rejecting a droplet costs nothing but effort ─────────────────
    //
    // The in-silico filter throws away most of what the fountain emits, and the code
    // is unharmed — because the next droplet is already available. A fixed-rate code
    // would have to reserve redundancy in advance to survive the same rejections.
    std::printf("\n-- rejecting a droplet costs effort, not bits --\n");
    {
        lpl::pmr::vector<core::u8> corpus;
        buildCorpus(1024u, 0xC0FFEEu, corpus);
        lpl::pmr::vector<core::u8> padded;
        padded.resize(static_cast<core::usize>(kBlocks) * 32u, core::u8{0});
        for (core::u32 i = 0u; i < corpus.size(); ++i)
            padded[i] = corpus[i];

        codec::SourceView source;
        source.bytes = padded.data();
        source.blockBytes = 32u;
        source.blockCount = kBlocks;
        codec::SolitonParams tuning;
        const codec::Fountain fountain{source, tuning};

        codec::BiologicalLimits limits;
        lpl::pmr::vector<codec::Droplet> valid;
        const core::u32 examined = fountain.emitValid(kBlocks, limits, 1u, valid);
        std::printf("    %zu synthesisable strands out of %u examined\n", valid.size(), examined);
        check(valid.size() == kBlocks, "the fountain still produces every strand asked for");
        check(examined > kBlocks, "having thrown some away on the way");

        bool allValid = true;
        for (core::usize i = 0u; i < valid.size(); ++i)
            allValid =
                allValid && codec::satisfiesBiologicalLimits(valid[i].payload.data(),
                                                             static_cast<core::u32>(valid[i].payload.size()), limits);
        check(allValid, "and every one it kept really does satisfy the limits");
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
