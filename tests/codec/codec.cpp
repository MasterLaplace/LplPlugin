#include <lpl/codec/BitMatrix.hpp>
#include <lpl/codec/Erasure.hpp>
#include <lpl/codec/FourRussians.hpp>
#include <lpl/codec/GaloisField.hpp>
#include <lpl/codec/GaussJordan.hpp>
#include <lpl/codec/XorKernel.hpp>
#include <lpl/math/Random.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(codec);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Bytes of the payload gate P11 codec encodes.
 */
constexpr lpl::core::u32 kCanonicalPayloadBytes = 384u;

/**
 * @brief Droplets the gate discards before decoding, one in this many.
 *
 * @details Dropping is what makes the gate test a code rather than a copy: a rateless code is
 *          defined by surviving loss, and a run that keeps every droplet only proves that the
 *          encoder and the decoder are inverses. It does not force the decode through the Gaussian
 *          tail: at the overhead the gate uses, peeling resolves all twenty-four blocks on its own,
 *          which is why the fold reduces a system of its own, unconditionally.
 */
constexpr lpl::core::u32 kDropStride = 7u;

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

/**
 * @struct CodecFoldResult
 * @brief What gate P11 codec records: the signatures both targets must reproduce, and the counters
 *        behind them.
 */
struct CodecFoldResult {
    lpl::core::u32 solitonSignature{0u}; /**< Fold of the degree distribution's weights. */
    lpl::core::u32 dropletSignature{0u}; /**< Fold of every emitted droplet, seed and payload. */
    lpl::core::u32 matrixSignature{0u};  /**< Fold of a reduced GF(2) system. */
    lpl::core::u32 payloadSignature{0u}; /**< Fold of the recovered payload. */
    lpl::core::u32 emitted{0u};          /**< Droplets the fountain produced. */
    lpl::core::u32 delivered{0u};        /**< Droplets left after the gate's drops. */
    lpl::core::u32 peeledBlocks{0u};     /**< Blocks belief propagation resolved. */
    lpl::core::u32 eliminatedBlocks{0u}; /**< Blocks the Gaussian tail finished. */
    lpl::core::u32 residualRows{0u};     /**< Rows the elimination was given. */
    lpl::core::u32 recovered{0u};        /**< 1 when the payload came back byte for byte. */
};

/**
 * @brief The case gate P11 codec encodes and decodes.
 *
 * @details Sized for the kernel's 4 MiB heap: twenty-four blocks of sixteen bytes, and a residual
 *          system of a few hundred columns. Its tuning puts c near 0.04 and delta near 0.05.
 *
 *          The overhead, 80 %, is measured rather than chosen. The paper's epsilon of 2 to 5 % is
 *          asymptotic, and at twenty-four blocks the soliton's guarantees do not hold. Over the
 *          first 64 seeds, with one droplet in seven discarded, 44 decode at 30 % overhead, 52 at
 *          40 %, 57 at 50 %, 61 at 60 %, and all 64 from 80 % on. A gate has to succeed every time on
 *          both targets, so it takes the first value that does, not the smallest that passes today.
 */
[[nodiscard]] constexpr lpl::codec::ErasureParams canonicalErasureParams() noexcept
{
    lpl::codec::ErasureParams params;

    params.blockBytes = 16u;
    params.overheadPermille = 800u;
    params.firstSeed = 0x5EEDu;
    params.tuning.c = lpl::math::Fixed32::fromRaw(2621);
    params.tuning.delta = lpl::math::Fixed32::fromRaw(3277);
    return params;
}

void foldWord(lpl::core::u32 &hash, lpl::core::u32 word) noexcept { hash = (hash ^ word) * kFnv1aPrime; }

[[nodiscard]] lpl::core::u32 foldBytes(const lpl::pmr::vector<lpl::core::u8> &bytes) noexcept
{
    lpl::core::u32 hash = kFnv1aOffsetBasis;

    for (lpl::core::usize index = 0u; index < bytes.size(); ++index)
        foldWord(hash, bytes[index]);
    return hash;
}

[[nodiscard]] bool sameBytes(const lpl::pmr::vector<lpl::core::u8> &lhs, const lpl::pmr::vector<lpl::core::u8> &rhs)
{
    bool same = lhs.size() == rhs.size();

    for (lpl::core::usize index = 0u; same && index < lhs.size(); ++index)
        same = lhs[index] == rhs[index];
    return same;
}

/**
 * @brief Builds the payload the gate encodes, from a seed.
 *
 * @details Not a literal: a 384-byte array is a thing that gets edited, and the gate would then
 *          compare two different payloads while reporting a signature mismatch as an arithmetic
 *          fault.
 */
void buildCanonicalPayload(lpl::pmr::vector<lpl::core::u8> &out)
{
    lpl::math::Random stream{0xC0DECu};

    out.clear();
    out.resize(kCanonicalPayloadBytes, lpl::core::u8{0});
    for (lpl::core::u32 index = 0u; index < kCanonicalPayloadBytes; ++index)
        out[index] = static_cast<lpl::core::u8>(stream.next() & 0xFFu);
}

/**
 * @brief Folds the degree distribution the encoder drew from.
 *
 * @details Folded on its own: two targets that disagree about one weight disagree about which
 *          droplets exist, and that has to be a gate failure rather than an occasional undecodable
 *          payload months later.
 */
[[nodiscard]] lpl::core::u32 foldSolitonTable(const lpl::codec::ErasureParams &params,
                                              const lpl::codec::ErasureShape &shape)
{
    lpl::codec::SolitonParams tuning = params.tuning;
    lpl::codec::SolitonTable table;

    tuning.sourceBlocks = shape.blockCount;
    table.build(tuning);
    return table.fold(kFnv1aOffsetBasis);
}

[[nodiscard]] lpl::core::u32 foldDroplets(const lpl::pmr::vector<lpl::codec::Droplet> &droplets)
{
    lpl::core::u32 hash = kFnv1aOffsetBasis;

    for (lpl::core::usize droplet = 0u; droplet < droplets.size(); ++droplet)
    {
        foldWord(hash, droplets[droplet].seed);
        for (lpl::core::usize byte = 0u; byte < droplets[droplet].payload.size(); ++byte)
            foldWord(hash, droplets[droplet].payload[byte]);
    }
    return hash;
}

void dropOneInStride(const lpl::pmr::vector<lpl::codec::Droplet> &droplets,
                     lpl::pmr::vector<lpl::codec::Droplet> &delivered)
{
    for (lpl::core::usize index = 0u; index < droplets.size(); ++index)
    {
        if ((index + 1u) % kDropStride == 0u)
            continue;

        lpl::codec::Droplet kept;

        kept.seed = droplets[index].seed;
        kept.payload = droplets[index].payload;
        delivered.push_back(kept);
    }
}

/**
 * @brief Reduces one GF(2) system with both eliminations, and folds the result.
 *
 * @details A stage of its own because the decode only reaches the elimination when peeling stalls,
 *          which depends on the droplets: this runs it on every boot. M4RI and the plain path must
 *          produce the same reduced form, bit for bit, or one of them is not computing reduced row
 *          echelon form.
 *
 * @param outMismatch Set to 1 when the two eliminations disagree.
 * @return The fold of the reduced matrix.
 */
[[nodiscard]] lpl::core::u32 foldReducedSystem(lpl::core::u32 &outMismatch)
{
    constexpr lpl::core::u32 kRows = 48u;
    constexpr lpl::core::u32 kColumns = 40u;
    lpl::codec::BitMatrix plain{kRows, kColumns};
    lpl::codec::BitMatrix blocked{kRows, kColumns};
    lpl::math::Random stream{0xB17Au};

    for (lpl::core::u32 row = 0u; row < kRows; ++row)
    {
        for (lpl::core::u32 column = 0u; column < kColumns; ++column)
        {
            if ((stream.next() & 1u) == 0u)
                continue;
            plain.set(row, column);
            blocked.set(row, column);
        }
    }

    const lpl::codec::EliminationResult plainResult = lpl::codec::gaussJordan(plain, kColumns);
    const lpl::codec::EliminationResult blockedResult = lpl::codec::fourRussiansEliminate(blocked, kColumns, 4u);

    outMismatch =
        (plain.fold(kFnv1aOffsetBasis) == blocked.fold(kFnv1aOffsetBasis) && plainResult.rank == blockedResult.rank) ?
            0u :
            1u;
    return plain.fold(kFnv1aOffsetBasis);
}

/**
 * @brief Runs the canonical case and folds every stage of it.
 *
 * @details The stages are folded, not only the answer: a payload that comes back proves the decode
 *          worked, and says nothing about whether the two targets built the same distribution or
 *          reduced the same matrix on the way. Two eliminations that disagree are a fault, not a
 *          signature to compare: they clear the verdict, which the test already checks, rather
 *          than set a flag a test would have to remember.
 *
 * @param out Receives the signatures; a payload that does not encode leaves them at zero.
 */
void foldCodecState(CodecFoldResult &out)
{
    out = CodecFoldResult{};

    const lpl::codec::ErasureParams params = canonicalErasureParams();
    lpl::pmr::vector<lpl::core::u8> payload;
    lpl::codec::ErasureShape shape{};
    lpl::pmr::vector<lpl::codec::Droplet> droplets;

    buildCanonicalPayload(payload);
    if (!lpl::codec::encodeErasure(payload.data(), static_cast<lpl::core::u32>(payload.size()), params, shape,
                                   droplets))
        return;
    out.emitted = static_cast<lpl::core::u32>(droplets.size());
    out.solitonSignature = foldSolitonTable(params, shape);
    out.dropletSignature = foldDroplets(droplets);

    lpl::pmr::vector<lpl::codec::Droplet> delivered;

    dropOneInStride(droplets, delivered);
    out.delivered = static_cast<lpl::core::u32>(delivered.size());

    lpl::pmr::vector<lpl::core::u8> recovered;
    lpl::codec::DecodeReport report{};
    const bool decoded = lpl::codec::decodeErasure(delivered, shape, params, recovered, report);

    out.peeledBlocks = report.peeledBlocks;
    out.eliminatedBlocks = report.eliminatedBlocks;
    out.residualRows = report.residualRows;
    out.recovered = (decoded && sameBytes(recovered, payload)) ? 1u : 0u;
    out.payloadSignature = foldBytes(recovered);

    lpl::core::u32 mismatch = 0u;

    out.matrixSignature = foldReducedSystem(mismatch);
    if (mismatch != 0u)
        out.recovered = 0u;
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
    CodecFoldResult folded{};

    foldCodecState(folded);
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
