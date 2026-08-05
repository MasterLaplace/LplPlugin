/**
 * @file ReedSolomon.hpp
 * @brief RS(n, k) over GF(256), the bounded-distance decoder.
 *
 * Corrects floor(s/2) unknown errors or s erasures for s parity symbols, via
 * Berlekamp-Massey. Used per-strand for DNA and per-section for cartridges: the
 * inner code of the two-level scheme.
 *
 * Where the fountain and this code divide the work: a fountain survives a shard
 * VANISHING, which it knows about, and Reed-Solomon survives a symbol being WRONG,
 * which nobody told it about. A bad sector produces the second; a lost packet
 * produces the first. Neither code does the other one's job, which is why the
 * archival scheme uses both.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_REEDSOLOMON_HPP
#    define LPL_LPL_CODEC_REEDSOLOMON_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::codec {

/**
 * @brief Parity symbols a codeword may carry.
 *
 * Bounded so the decoder needs no heap: every working polynomial is at most this
 * long, and they live on the stack. Sixteen corrects eight unknown errors per
 * codeword, which is far past what a transversal layout ever needs.
 */
inline constexpr core::u32 kMaxParitySymbols = 16u;

/**
 * @brief Symbols in a codeword, data and parity together.
 *
 * 255 is the field's own limit: GF(256) has 255 non-zero elements, so a longer
 * codeword would need two roots to be the same element.
 */
inline constexpr core::u32 kMaxCodewordSymbols = 255u;

/**
 * @brief Builds the generator polynomial for @p parityCount parity symbols.
 *
 * g(x) = (x - a^0)(x - a^1)...(x - a^(s-1)), ascending coefficient order. Exposed
 * rather than hidden because the encoder and the syndrome check must agree on which
 * roots the code has, and a second opinion about that is a code that encodes one way
 * and verifies another.
 *
 * @param parityCount Parity symbols; clamped to @ref kMaxParitySymbols.
 * @param out         Receives parityCount + 1 coefficients, lowest degree first.
 * @return Coefficients written.
 */
core::u32 generatorPolynomial(core::u32 parityCount, core::u8 *out) noexcept;

/**
 * @brief Appends @p parityCount parity symbols to @p data.
 *
 * Systematic: the data is left untouched and the parity follows it, so a reader that
 * trusts its medium can ignore the code entirely and a reader that does not can use
 * it without rearranging anything.
 *
 * @param data        The message.
 * @param dataCount   Symbols in it.
 * @param parityCount Parity symbols to produce.
 * @param outParity   Receives @p parityCount symbols.
 * @return false when the parameters exceed the field or the fixed bounds.
 */
[[nodiscard]] bool reedSolomonEncode(const core::u8 *data, core::u32 dataCount, core::u32 parityCount,
                                     core::u8 *outParity) noexcept;

/**
 * @struct ReedSolomonRepair
 * @brief What a decode found and did.
 */
struct ReedSolomonRepair {
    bool clean{false};         ///< The codeword already satisfied the code.
    bool corrected{false};     ///< It did not, and every error was located and fixed.
    core::u32 errorCount{0u};  ///< Symbols changed.
    core::u32 errorDegree{0u}; ///< Degree of the error locator, i.e. errors the syndromes implied.
};

/**
 * @brief Corrects a codeword in place, or reports that it cannot.
 *
 * Bounded distance, and the bound is the honest part: with @p parityCount parity
 * symbols this finds and fixes at most floor(s/2) errors. Beyond that it does not
 * silently produce a plausible codeword — it reports failure, because for an archival
 * format a wrong world that loads is strictly worse than a right one that refuses.
 *
 * @param codeword     Data followed by parity; modified on success.
 * @param symbolCount  Total symbols, data and parity together.
 * @param parityCount  Parity symbols at the end.
 * @param outRepair    Receives what happened.
 * @return true when the codeword is now valid, whether or not it needed work.
 */
[[nodiscard]] bool reedSolomonCorrect(core::u8 *codeword, core::u32 symbolCount, core::u32 parityCount,
                                      ReedSolomonRepair &outRepair) noexcept;

/**
 * @struct TransversalReport
 * @brief What a transversal repair found.
 */
struct TransversalReport {
    core::u32 codewords{0u};        ///< Columns examined.
    core::u32 damagedCodewords{0u}; ///< Columns that were not already clean.
    core::u32 correctedBytes{0u};   ///< Symbols changed.
};

/**
 * @brief Lays a span out as rows and codes DOWN the columns.
 *
 * The arrangement, not the code, is what makes this survive the failure a stored
 * artifact actually suffers. A bad sector, a scratch, a page that did not come back
 * are all BURSTS — contiguous — and coding along a row would put the whole burst into
 * one codeword and lose it. Cutting the span into rows and taking a codeword down each
 * column instead puts a burst confined to one row into a single wrong symbol per
 * codeword, which is the case Reed-Solomon is strongest at.
 *
 * Bytes past @p protectedBytes read as zero, and the repair pads the same way. The two
 * have to agree or every last codeword is wrong by construction.
 *
 * @param protectedBytesPtr The span.
 * @param protectedBytes    Its length.
 * @param dataShards        Rows to cut it into.
 * @param parityShards      Parity symbols per column.
 * @param rowBytes          Bytes per row, i.e. codewords; ceil(protectedBytes/dataShards).
 * @param outParity         Receives parityShards * rowBytes bytes.
 * @return false when the parameters exceed the field or the fixed bounds.
 */
[[nodiscard]] bool transversalEncode(const core::u8 *protectedBytesPtr, core::u32 protectedBytes,
                                     core::u32 dataShards, core::u32 parityShards, core::u32 rowBytes,
                                     core::u8 *outParity) noexcept;

/**
 * @brief Corrects a transversally coded span in place.
 *
 * Corrections land in the data AND in the parity, because a burst is as likely to have
 * hit one as the other and a codeword is only valid when every symbol of it is.
 *
 * @param protectedBytesPtr The span, modified where it was wrong.
 * @param protectedBytes    Its length.
 * @param parity            The parity rows, modified likewise.
 * @param dataShards        Rows the span was cut into.
 * @param parityShards      Parity symbols per column.
 * @param rowBytes          Bytes per row.
 * @param outReport         Receives the tally.
 * @return true when every column is now valid. False means damage past the bound, and
 *         nothing was written back for the columns that failed.
 */
[[nodiscard]] bool transversalRepair(core::u8 *protectedBytesPtr, core::u32 protectedBytes, core::u8 *parity,
                                     core::u32 dataShards, core::u32 parityShards, core::u32 rowBytes,
                                     TransversalReport &outReport) noexcept;

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_REEDSOLOMON_HPP
