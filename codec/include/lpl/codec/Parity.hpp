/**
 * @file Parity.hpp
 * @brief The single constexpr case both sides of the gate encode.
 *
 * Same discipline as parityWorldRecipe(): one definition, read by the Linux
 * oracle and by the kernel smoke, so the two cannot drift by editing their own
 * copy of the parameters. foldCodecState() folds raw words, never a decimal.
 *
 * What makes gate P11 different from every gate before it: the two targets run
 * genuinely DIFFERENT code. Everywhere else the contract is "the same source
 * compiled twice"; here the host takes a 128-bit XOR kernel and ring 0 takes a
 * word-at-a-time one, and the claim is that reordering associative, commutative,
 * rounding-free operations changes nothing. A signature mismatch would say that claim
 * is wrong, which is precisely what nothing else in this repository can test.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_PARITY_HPP
#    define LPL_LPL_CODEC_PARITY_HPP

#    include <lpl/codec/Erasure.hpp>
#    include <lpl/core/Types.hpp>

namespace lpl::codec {

/**
 * @brief The canonical case both targets encode and decode.
 *
 * Sized for the kernel's 4 MiB heap rather than a desktop's: twenty-four blocks of
 * sixteen bytes is a payload of 384 bytes and a residual system of a few hundred
 * columns, which is comparable to the world recipe the gate already bakes.
 *
 * @return The parameters, identical on both sides by construction.
 */
[[nodiscard]] constexpr ErasureParams parityErasureParams() noexcept
{
    ErasureParams params;
    params.blockBytes = 16u;
    // 80 %, and the number is measured rather than chosen.
    //
    // The paper's epsilon of 2 to 5 % is ASYMPTOTIC, and K here is twenty-four — the
    // soliton's guarantees do not hold at that size. Sweeping 64 first seeds with one
    // droplet in seven discarded: 44/64 decodes at 30 % overhead, 52 at 40, 57 at 50,
    // 61 at 60, and 64/64 from 80 % on. A gate has to succeed every time on both
    // targets, so it takes the first value that does, not the smallest that passes
    // today.
    params.overheadPermille = 800u;
    params.firstSeed = 0x5EEDu;
    params.tuning.c = math::Fixed32::fromRaw(2621);     // ~0.04
    params.tuning.delta = math::Fixed32::fromRaw(3277); // ~0.05
    return params;
}

/**
 * @brief Bytes of the canonical payload.
 *
 * @return The payload length the gate encodes.
 */
[[nodiscard]] constexpr core::u32 parityPayloadBytes() noexcept { return 384u; }

/**
 * @brief Droplets the gate discards before decoding, one in this many.
 *
 * Dropping is what makes the gate test a CODE rather than a copy: a rateless code is
 * defined by surviving loss, and a run that keeps every droplet only proves the
 * encoder and the decoder are inverses.
 *
 * It does NOT force the decode through the Gaussian tail, and an earlier version of
 * this comment claimed it did. At the overhead the gate uses, peeling resolves all
 * twenty-four blocks on its own and the elimination never runs — which is why
 * foldCodecState reduces a system of its own, unconditionally, instead of hoping the
 * cascade breaks.
 *
 * @return The drop stride.
 */
[[nodiscard]] constexpr core::u32 parityDropStride() noexcept { return 7u; }

/**
 * @struct CodecFoldResult
 * @brief The signatures the kernel must reproduce.
 *
 * Deliberately free of Fixed32 and bool so the kernel smoke can copy it into a plain
 * C struct field by field, the same reason WorldRecipeResult is.
 */
struct CodecFoldResult {
    core::u32 solitonSignature{0u}; ///< Fold of the degree distribution's weights.
    core::u32 dropletSignature{0u}; ///< Fold of every emitted droplet, seed and payload.
    core::u32 matrixSignature{0u};  ///< Fold of a reduced GF(2) system.
    core::u32 payloadSignature{0u}; ///< Fold of the recovered payload.
    core::u32 emitted{0u};          ///< Droplets the fountain produced.
    core::u32 delivered{0u};        ///< Droplets left after the gate's drops.
    core::u32 peeledBlocks{0u};     ///< Blocks belief propagation resolved.
    core::u32 eliminatedBlocks{0u}; ///< Blocks the Gaussian tail finished.
    core::u32 residualRows{0u};     ///< Rows the elimination was given.
    core::u32 recovered{0u};        ///< 1 when the payload came back byte for byte.
    core::u32 vectorKernel{0u};     ///< 1 when this build took the widened XOR path.
};

/**
 * @brief Runs the canonical case and folds every stage of it.
 *
 * One function, called by the host oracle and by the kernel smoke. Folding the
 * intermediate stages and not only the answer is deliberate: a payload that comes
 * back correct proves the decode worked, and says nothing about whether the two
 * targets built the same distribution or reduced the same matrix on the way there.
 *
 * @param out Receives the signatures.
 */
void foldCodecState(CodecFoldResult &out);

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_PARITY_HPP
