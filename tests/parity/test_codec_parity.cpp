/**
 * @file test_codec_parity.cpp
 * @brief Gate P11 codec — the oracle side.
 *
 * Two claims, and only the second one is new to this repository.
 *
 * The first is the usual: the same case folds the same signatures on the Linux host
 * and inside the i686 kernel. The second is what makes P11 different — the two sides
 * run genuinely DIFFERENT code. Every gate before this compiled one source twice; here
 * the host XORs 128 bits at a time and ring 0 XORs one word at a time, and the claim
 * is that reordering associative, commutative, rounding-free operations changes
 * nothing. Nothing else in the tree can test that claim.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/BitMatrix.hpp>
#include <lpl/codec/FourRussians.hpp>
#include <lpl/codec/GaloisField.hpp>
#include <lpl/codec/GaussJordan.hpp>
#include <lpl/codec/Parity.hpp>
#include <lpl/codec/XorKernel.hpp>
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
 * @brief The field is a field, which every routine above silently assumes.
 */
void testTheFieldIsAField()
{
    std::printf("\n-- GF(256) --\n");
    using namespace lpl;

    bool inverses = true;
    bool distributes = true;
    for (core::u32 a = 1u; a < 256u; ++a)
    {
        const core::u8 x = static_cast<core::u8>(a);
        inverses = inverses && codec::gf256Mul(x, codec::gf256Inv(x)) == 1u;
        for (core::u32 b = 1u; b < 16u; ++b)
        {
            const core::u8 y = static_cast<core::u8>(b);
            const core::u8 z = static_cast<core::u8>((a * 7u + b) & 0xFFu);
            distributes = distributes && codec::gf256Mul(x, codec::gf256Add(y, z)) ==
                                             codec::gf256Add(codec::gf256Mul(x, y), codec::gf256Mul(x, z));
        }
    }
    check(inverses, "every non-zero element has a multiplicative inverse");
    check(distributes, "multiplication distributes over addition");
    check(codec::gf256Mul(0u, 123u) == 0u && codec::gf256Div(45u, 0u) == 0u, "zero absorbs rather than traps");

    // The tables are constexpr, so this is a statement about the image rather than
    // about a runtime initialisation that could differ between targets.
    static_assert(codec::kGf256.exp[0] == 1u, "the generator's zeroth power is one");
    static_assert(codec::kGf256.exp[255] == 1u, "the multiplicative group has order 255");
}

/**
 * @brief The two XOR kernels agree, and the build really took the wide one.
 */
void testTheKernelsAgree()
{
    std::printf("\n-- the XOR kernel --\n");
    using namespace lpl;

    // A test that passed because both sides quietly ran the scalar loop would be a
    // verification incapable of failing, so the path is asserted rather than assumed.
    check(codec::activeXorPath() == codec::XorPath::Sse2, "the host build took the widened path");

    constexpr core::u32 kWords = 37u; // deliberately not a multiple of the unroll
    lpl::pmr::vector<core::u64> a;
    lpl::pmr::vector<core::u64> b;
    lpl::pmr::vector<core::u64> reference;
    a.resize(kWords, core::u64{0});
    b.resize(kWords, core::u64{0});
    reference.resize(kWords, core::u64{0});

    math::Random stream{0x1234u};
    for (core::u32 i = 0u; i < kWords; ++i)
    {
        a[i] = (static_cast<core::u64>(stream.next()) << 32) | stream.next();
        b[i] = (static_cast<core::u64>(stream.next()) << 32) | stream.next();
        reference[i] = a[i] ^ b[i];
    }

    lpl::pmr::vector<core::u64> widened = a;
    codec::xorRow(widened.data(), b.data(), kWords);
    bool same = true;
    for (core::u32 i = 0u; i < kWords; ++i)
        same = same && widened[i] == reference[i];
    check(same, "the widened kernel matches a word-at-a-time XOR, tail included");

    lpl::pmr::vector<core::u64> three;
    three.resize(kWords, core::u64{0});
    codec::xorRowInto(three.data(), a.data(), b.data(), kWords);
    same = true;
    for (core::u32 i = 0u; i < kWords; ++i)
        same = same && three[i] == reference[i];
    check(same, "and so does the three-operand form");

    check(!codec::rowIsZero(a.data(), kWords), "a random row is not zero");
    codec::xorRow(widened.data(), reference.data(), kWords);
    check(codec::rowIsZero(widened.data(), kWords), "and a row XORed with itself is");
}

/**
 * @brief M4RI computes reduced row echelon form, not something that resembles it.
 */
void testTheTwoEliminationsAgree()
{
    std::printf("\n-- elimination --\n");
    using namespace lpl;

    core::u32 disagreements = 0u;
    core::u32 rankSum = 0u;
    for (core::u32 seed = 0u; seed < 24u; ++seed)
    {
        constexpr core::u32 kRows = 40u;
        constexpr core::u32 kColumns = 33u;
        codec::BitMatrix plain{kRows, kColumns};
        codec::BitMatrix blocked{kRows, kColumns};

        math::Random stream{0xA5A5u + seed * 977u};
        for (core::u32 r = 0u; r < kRows; ++r)
            for (core::u32 c = 0u; c < kColumns; ++c)
                if ((stream.next() & 1u) != 0u)
                {
                    plain.set(r, c);
                    blocked.set(r, c);
                }

        const codec::EliminationResult plainResult = codec::gaussJordan(plain, kColumns);
        const codec::EliminationResult blockedResult = codec::fourRussiansEliminate(blocked, kColumns, 4u);

        rankSum += plainResult.rank;
        if (plainResult.rank != blockedResult.rank || plain.fold(0x811C9DC5u) != blocked.fold(0x811C9DC5u))
            ++disagreements;
    }
    std::printf("    24 systems, mean rank %u\n", rankSum / 24u);
    check(disagreements == 0u, "M4RI and the plain path reduce to the same matrix, bit for bit");

    // The Gray-code table is the reason M4RI is worth having: built naively it costs
    // 2^k * k row XORs, and the whole gain evaporates. Measured, not asserted by
    // comment.
    codec::BitMatrix source{8u, 64u};
    math::Random stream{0x77u};
    for (core::u32 r = 0u; r < 8u; ++r)
        for (core::u32 c = 0u; c < 64u; ++c)
            if ((stream.next() & 1u) != 0u)
                source.set(r, c);

    codec::GrayCodeTable table;
    table.build(source, 0u, 6u);
    std::printf("    k=6 table: %u entries in %u XORs (naive would be %u)\n", table.entries(), table.xorsPerformed(),
                (1u << 6) * 6u);
    check(table.entries() == 64u, "a k=6 table holds 2^6 combinations");
    check(table.xorsPerformed() < table.entries(), "and costs fewer XORs than it has entries");

    // Every entry must genuinely be the combination its index names, or the
    // elimination is XORing the wrong rows and only the parity fold would notice.
    bool correct = true;
    for (core::u32 index = 0u; index < table.entries(); ++index)
    {
        const core::u64 *const combination = table.combination(index);
        for (core::u32 word = 0u; word < source.rowWords() && correct; ++word)
        {
            core::u64 expected = 0u;
            for (core::u32 bit = 0u; bit < 6u; ++bit)
                if (((index >> bit) & 1u) != 0u)
                    expected ^= source.row(bit)[word];
            correct = combination[word] == expected;
        }
    }
    check(correct, "and every entry is the exact combination its index names");
}

} // namespace

int main()
{
    std::printf("== codec: GF(2), the fountain, and the two kernels ==\n");

    testTheFieldIsAField();
    testTheKernelsAgree();
    testTheTwoEliminationsAgree();

    std::printf("\n-- signatures the kernel must reproduce --\n");
    lpl::codec::CodecFoldResult folded{};
    lpl::codec::foldCodecState(folded);

    std::printf("  soliton_sig  = 0x%08X\n", folded.solitonSignature);
    std::printf("  droplet_sig  = 0x%08X\n", folded.dropletSignature);
    std::printf("  matrix_sig   = 0x%08X\n", folded.matrixSignature);
    std::printf("  payload_sig  = 0x%08X\n", folded.payloadSignature);
    std::printf("  emitted      = %u\n", folded.emitted);
    std::printf("  delivered    = %u\n", folded.delivered);
    std::printf("  peeled       = %u\n", folded.peeledBlocks);
    std::printf("  eliminated   = %u\n", folded.eliminatedBlocks);
    std::printf("  residual     = %u\n", folded.residualRows);
    std::printf("  recovered    = %u\n", folded.recovered);
    std::printf("  vector       = %u\n", folded.vectorKernel);

    check(folded.recovered == 1u, "the canonical payload survives losing one droplet in seven");
    check(folded.delivered < folded.emitted, "and droplets really were dropped");

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
