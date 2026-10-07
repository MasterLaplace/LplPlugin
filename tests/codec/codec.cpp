#include <lpl/codec/BitMatrix.hpp>
#include <lpl/codec/FourRussians.hpp>
#include <lpl/codec/GaloisField.hpp>
#include <lpl/codec/GaussJordan.hpp>
#include <lpl/codec/Parity.hpp>
#include <lpl/codec/XorKernel.hpp>
#include <lpl/math/Random.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(codec);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;

[[nodiscard]] lpl::core::u64 randomWord(lpl::math::Random &stream)
{
    const lpl::core::u64 high = stream.next();

    return (high << 32) | stream.next();
}

[[nodiscard]] bool sameRows(const lpl::pmr::vector<lpl::core::u64> &lhs, const lpl::pmr::vector<lpl::core::u64> &rhs)
{
    bool same = lhs.size() == rhs.size();

    for (lpl::core::usize index = 0u; same && index < lhs.size(); ++index)
        same = lhs[index] == rhs[index];
    return same;
}

void fillRandomly(lpl::codec::BitMatrix &matrix, lpl::core::u32 rows, lpl::core::u32 columns, lpl::math::Random &stream)
{
    for (lpl::core::u32 row = 0u; row < rows; ++row)
    {
        for (lpl::core::u32 column = 0u; column < columns; ++column)
        {
            if ((stream.next() & 1u) != 0u)
                matrix.set(row, column);
        }
    }
}

} // namespace

LPL_TEST(galois_field_is_a_field)
{
    bool inverses = true;
    bool distributes = true;

    for (lpl::core::u32 a = 1u; a < 256u; ++a)
    {
        const lpl::core::u8 x = static_cast<lpl::core::u8>(a);

        inverses = inverses && lpl::codec::gf256Mul(x, lpl::codec::gf256Inv(x)) == 1u;
        for (lpl::core::u32 b = 1u; b < 16u; ++b)
        {
            const lpl::core::u8 y = static_cast<lpl::core::u8>(b);
            const lpl::core::u8 z = static_cast<lpl::core::u8>((a * 7u + b) & 0xFFu);

            distributes =
                distributes && lpl::codec::gf256Mul(x, lpl::codec::gf256Add(y, z)) ==
                                   lpl::codec::gf256Add(lpl::codec::gf256Mul(x, y), lpl::codec::gf256Mul(x, z));
        }
    }
    test.check(inverses, "every element but zero has an inverse");
    test.check(distributes, "multiplication distributes over addition");
    test.check(lpl::codec::gf256Mul(0u, 123u) == 0u && lpl::codec::gf256Div(45u, 0u) == 0u,
               "zero absorbs instead of trapping");
}

/**
 * @brief The XOR kernel this target compiled matches a plain XOR, tail included: 37 words is not a
 *        multiple of the unrolling.
 */
LPL_TEST(xor_kernel_matches_a_plain_xor)
{
    constexpr lpl::core::u32 kWords = 37u;
    lpl::pmr::vector<lpl::core::u64> a;
    lpl::pmr::vector<lpl::core::u64> b;
    lpl::pmr::vector<lpl::core::u64> expected;
    lpl::pmr::vector<lpl::core::u64> threeOperands;
    lpl::math::Random stream{0x1234u};

    a.resize(kWords, lpl::core::u64{0});
    b.resize(kWords, lpl::core::u64{0});
    expected.resize(kWords, lpl::core::u64{0});
    threeOperands.resize(kWords, lpl::core::u64{0});
    for (lpl::core::u32 index = 0u; index < kWords; ++index)
    {
        a[index] = randomWord(stream);
        b[index] = randomWord(stream);
        expected[index] = a[index] ^ b[index];
    }

    lpl::pmr::vector<lpl::core::u64> inPlace = a;

    lpl::codec::xorRow(inPlace.data(), b.data(), kWords);
    lpl::codec::xorRowInto(threeOperands.data(), a.data(), b.data(), kWords);
    test.check(sameRows(inPlace, expected), "XOR in place matches, tail included");
    test.check(sameRows(threeOperands, expected), "and so does the three-operand form");
    test.check(!lpl::codec::rowIsZero(a.data(), kWords), "a random row is not zero");

    lpl::codec::xorRow(inPlace.data(), expected.data(), kWords);
    test.check(lpl::codec::rowIsZero(inPlace.data(), kWords), "a row XORed with itself is");
}

/**
 * @brief The host XORs 128 bits at a time and ring 0 one word at a time, which is what makes gate
 *        P11 more than one source compiled twice: the records must agree all the same.
 */
LPL_TEST(each_target_takes_its_own_xor_path)
{
#if LPL_TARGET_KERNEL
    test.check(lpl::codec::activeXorPath() == lpl::codec::XorPath::Scalar, "ring 0 XORs a word at a time");
#else
    test.check(lpl::codec::activeXorPath() == lpl::codec::XorPath::Sse2, "the host XORs 128 bits at a time");
#endif
}

LPL_TEST(the_two_eliminations_agree)
{
    constexpr lpl::core::u32 kRows = 40u;
    constexpr lpl::core::u32 kColumns = 33u;
    lpl::core::u32 disagreements = 0u;

    for (lpl::core::u32 seed = 0u; seed < 24u; ++seed)
    {
        lpl::codec::BitMatrix plain{kRows, kColumns};
        lpl::codec::BitMatrix blocked{kRows, kColumns};
        lpl::math::Random plainStream{0xA5A5u + seed * 977u};
        lpl::math::Random blockedStream{0xA5A5u + seed * 977u};

        fillRandomly(plain, kRows, kColumns, plainStream);
        fillRandomly(blocked, kRows, kColumns, blockedStream);

        const lpl::codec::EliminationResult plainResult = lpl::codec::gaussJordan(plain, kColumns);
        const lpl::codec::EliminationResult blockedResult = lpl::codec::fourRussiansEliminate(blocked, kColumns, 4u);

        if (plainResult.rank != blockedResult.rank || plain.fold(kFnv1aOffsetBasis) != blocked.fold(kFnv1aOffsetBasis))
            ++disagreements;
    }
    test.check(disagreements == 0u, "the Four Russians method and Gauss-Jordan reduce 24 systems to the same matrix");
}

LPL_TEST(gray_code_table_is_cheap_and_exact)
{
    constexpr lpl::core::u32 kBits = 6u;
    lpl::codec::BitMatrix source{8u, 64u};
    lpl::math::Random stream{0x77u};
    lpl::codec::GrayCodeTable table;
    bool exact = true;

    fillRandomly(source, 8u, 64u, stream);
    table.build(source, 0u, kBits);
    test.check(table.entries() == 1u << kBits, "a table over six rows holds 2^6 combinations");
    test.check(table.xorsPerformed() < table.entries(), "and costs fewer XORs than it has entries");

    for (lpl::core::u32 index = 0u; exact && index < table.entries(); ++index)
    {
        const lpl::core::u64 *const combination = table.combination(index);

        for (lpl::core::u32 word = 0u; exact && word < source.rowWords(); ++word)
        {
            lpl::core::u64 expected = 0u;

            for (lpl::core::u32 bit = 0u; bit < kBits; ++bit)
            {
                if (((index >> bit) & 1u) != 0u)
                    expected ^= source.row(bit)[word];
            }
            exact = combination[word] == expected;
        }
    }
    test.check(exact, "every entry is the combination its index names");
}

/**
 * @brief Gate P11 codec: a payload sent through a fountain that loses one droplet in seven comes
 *        back, through belief propagation and a Gaussian tail, with the same signatures.
 */
LPL_TEST(payload_survives_a_lossy_channel)
{
    lpl::codec::CodecFoldResult folded{};

    lpl::codec::foldCodecState(folded);
    test.check(folded.recovered == 1u, "the payload comes back byte for byte");
    test.check(folded.delivered < folded.emitted, "although droplets were dropped");

    test.measureHexadecimal("soliton_signature", folded.solitonSignature);
    test.measureHexadecimal("droplet_signature", folded.dropletSignature);
    test.measureHexadecimal("matrix_signature", folded.matrixSignature);
    test.measureHexadecimal("payload_signature", folded.payloadSignature);
    test.measure("emitted", folded.emitted);
    test.measure("delivered", folded.delivered);
    test.measure("peeled_blocks", folded.peeledBlocks);
    test.measure("eliminated_blocks", folded.eliminatedBlocks);
    test.measure("residual_rows", folded.residualRows);
}
