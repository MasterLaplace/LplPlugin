/**
 * @file Parity.cpp
 * @brief The canonical codec case, folded stage by stage.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/Parity.hpp>

#include <lpl/codec/FourRussians.hpp>
#include <lpl/codec/GaussJordan.hpp>
#include <lpl/codec/XorKernel.hpp>
#include <lpl/math/Random.hpp>

namespace lpl::codec {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Folds one 32-bit word into a running FNV-1a hash.
 * @param hash Running value.
 * @param word Word to absorb.
 */
void fold(core::u32 &hash, core::u32 word) noexcept { hash = (hash ^ word) * kFnv1aPrime; }

/**
 * @brief Builds the payload the gate encodes.
 *
 * Generated from a seed rather than written out as a literal: a 384-byte array in a
 * header is a thing that gets edited, and the gate would then be comparing two
 * different payloads while reporting a signature mismatch as an arithmetic fault.
 *
 * @param out Receives @ref parityPayloadBytes bytes.
 */
void buildParityPayload(lpl::pmr::vector<core::u8> &out)
{
    out.clear();
    out.resize(parityPayloadBytes(), core::u8{0});
    math::Random stream{0xC0DECu};
    for (core::u32 i = 0u; i < parityPayloadBytes(); ++i)
        out[i] = static_cast<core::u8>(stream.next() & 0xFFu);
}

/**
 * @brief Builds and reduces a small GF(2) system, and folds the result.
 *
 * A stage of its own, on top of the erasure round trip, because the decode only
 * reaches the elimination when peeling stalls — which depends on the droplets and is
 * therefore not guaranteed. This runs it unconditionally, so the gate covers the
 * elimination on every boot rather than on the boots where the cascade happened to
 * break.
 *
 * It also checks the two eliminations against each other: M4RI and the plain path
 * must produce the same reduced form, bit for bit, or one of them is not computing
 * reduced row echelon form.
 *
 * @param outMismatch Set to 1 when the two eliminations disagree.
 * @return The fold of the reduced matrix.
 */
core::u32 foldReducedSystem(core::u32 &outMismatch)
{
    constexpr core::u32 kRows = 48u;
    constexpr core::u32 kColumns = 40u;

    BitMatrix plain{kRows, kColumns};
    BitMatrix blocked{kRows, kColumns};

    math::Random stream{0xB17Au};
    for (core::u32 r = 0u; r < kRows; ++r)
        for (core::u32 c = 0u; c < kColumns; ++c)
            if ((stream.next() & 1u) != 0u)
            {
                plain.set(r, c);
                blocked.set(r, c);
            }

    const EliminationResult plainResult = gaussJordan(plain, kColumns);
    const EliminationResult blockedResult = fourRussiansEliminate(blocked, kColumns, 4u);

    outMismatch =
        (plain.fold(kFnv1aOffsetBasis) == blocked.fold(kFnv1aOffsetBasis) && plainResult.rank == blockedResult.rank) ?
            0u :
            1u;

    return plain.fold(kFnv1aOffsetBasis);
}

} // namespace

void foldCodecState(CodecFoldResult &out)
{
    out = CodecFoldResult{};
    out.vectorKernel = activeXorPath() == XorPath::Sse2 ? 1u : 0u;

    const ErasureParams params = parityErasureParams();

    lpl::pmr::vector<core::u8> payload;
    buildParityPayload(payload);

    ErasureShape shape{};
    lpl::pmr::vector<Droplet> droplets;
    if (!encodeErasure(payload.data(), static_cast<core::u32>(payload.size()), params, shape, droplets))
        return;

    out.emitted = static_cast<core::u32>(droplets.size());

    // The distribution, folded on its own. Two targets that disagree about one weight
    // disagree about which droplets exist, and that has to be a gate failure rather
    // than an occasional undecodable payload months later.
    SolitonParams tuning = params.tuning;
    tuning.sourceBlocks = shape.blockCount;
    SolitonTable table;
    table.build(tuning);
    out.solitonSignature = table.fold(kFnv1aOffsetBasis);

    core::u32 dropletHash = kFnv1aOffsetBasis;
    for (core::usize d = 0u; d < droplets.size(); ++d)
    {
        fold(dropletHash, droplets[d].seed);
        for (core::usize b = 0u; b < droplets[d].payload.size(); ++b)
            fold(dropletHash, droplets[d].payload[b]);
    }
    out.dropletSignature = dropletHash;

    // Lose one droplet in seven. A rateless code is supposed to survive that, and
    // dropping is what forces the decode through the elimination the peeling loop
    // cannot finish on its own.
    lpl::pmr::vector<Droplet> delivered;
    for (core::usize d = 0u; d < droplets.size(); ++d)
    {
        if ((d + 1u) % parityDropStride() == 0u)
            continue;
        Droplet kept;
        kept.seed = droplets[d].seed;
        kept.payload = droplets[d].payload;
        delivered.push_back(kept);
    }
    out.delivered = static_cast<core::u32>(delivered.size());

    lpl::pmr::vector<core::u8> recovered;
    DecodeReport report{};
    const bool ok = decodeErasure(delivered, shape, params, recovered, report);

    out.peeledBlocks = report.peeledBlocks;
    out.eliminatedBlocks = report.eliminatedBlocks;
    out.residualRows = report.residualRows;

    bool identical = ok && recovered.size() == payload.size();
    for (core::usize i = 0u; identical && i < payload.size(); ++i)
        identical = recovered[i] == payload[i];
    out.recovered = identical ? 1u : 0u;

    core::u32 payloadHash = kFnv1aOffsetBasis;
    for (core::usize i = 0u; i < recovered.size(); ++i)
        fold(payloadHash, recovered[i]);
    out.payloadSignature = payloadHash;

    core::u32 mismatch = 0u;
    out.matrixSignature = foldReducedSystem(mismatch);
    if (mismatch != 0u)
    {
        // The two eliminations disagreeing is not a signature to compare, it is a
        // fault. Reported by poisoning the verdict rather than by a separate flag the
        // kernel smoke would have to remember to print.
        out.recovered = 0u;
    }
}

} // namespace lpl::codec
