/**
 * @file XorKernel.hpp
 * @brief One row-XOR contract, several widths behind it.
 *
 * The hot loop of every GF(2) operation in this module: `dst ^= src` over
 * 64-bit words. Elimination, table construction and decoding all funnel here, so
 * this is the only place worth widening.
 *
 * Written with intrinsics rather than hand-written assembly, deliberately. The
 * generated instructions are the same; what changes is that the compiler keeps the
 * register allocation, the aliasing information and the unrolling, and a scalar
 * fallback stays one `#if` away instead of one port away.
 *
 * **Why widening is safe here and is not safe for floats.** The two targets take
 * genuinely different code paths — a 128-bit path on the host, a word-at-a-time path
 * in ring 0 — and the determinism contract still holds, because the only thing the
 * vector path changes is the ORDER in which independent XORs are issued. Over GF(2)
 * addition is associative and commutative and has no rounding, so reordering is the
 * identity on the result. That sentence is the whole licence: any future kernel that
 * needs an operation without those properties does not belong in this file.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_XORKERNEL_HPP
#    define LPL_LPL_CODEC_XORKERNEL_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::codec {

/**
 * @enum XorPath
 * @brief Which kernel a build actually compiled.
 *
 * Reported rather than assumed, and asserted by the parity test: "the host and the
 * kernel fold the same signature" is only evidence if the two really did take
 * different paths. A test that passes because both sides quietly ran the scalar
 * loop would be the repository's second-most-repeated mistake — a verification
 * incapable of failing.
 */
enum class XorPath : core::u32 {
    Scalar = 0u, ///< 64-bit words, unrolled by four. The i686 and generic path.
    Sse2 = 1u,   ///< 128-bit lanes. Baseline on x86-64, so no runtime dispatch.
};

/**
 * @return The path this translation unit compiled.
 */
[[nodiscard]] XorPath activeXorPath() noexcept;

/**
 * @brief dst ^= src, over @p words 64-bit words.
 *
 * Unrolled by four (SIM-098): four independent XORs saturate the execution ports and
 * hide load latency, which a dependent chain cannot. The tail is handled one word at
 * a time rather than by a masked store, because the tail is at most three words and a
 * mask register costs more to set up than the words cost to move.
 *
 * @param destination Row to accumulate into.
 * @param source      Row to add.
 * @param words       Length of both, in 64-bit words.
 */
void xorRow(core::u64 *destination, const core::u64 *source, core::u32 words) noexcept;

/**
 * @brief destination = a ^ b, over @p words words.
 *
 * The three-operand form. It exists because the peeling decoder never wants to
 * destroy either input — a droplet is XORed out of several others — and writing
 * `copy then xorRow` doubles the traffic through the only loop that matters.
 */
void xorRowInto(core::u64 *destination, const core::u64 *a, const core::u64 *b, core::u32 words) noexcept;

/**
 * @brief Is every word zero?
 *
 * SIM-096's third pillar: test before calling the kernel. A row that is already zero
 * is the common case in the tail of an elimination, and skipping it is free next to
 * a full pass over the row.
 */
[[nodiscard]] bool rowIsZero(const core::u64 *row, core::u32 words) noexcept;

/**
 * @brief Index of the first word that is not zero, or @p words when all are.
 *
 * Word skipping (SIM-096): after k pivots the first k/64 words of every row below are
 * known zero, so the elimination should not read them. This is what turns the
 * bit-packed form from "the same algorithm on smaller data" into a different
 * complexity in practice.
 */
[[nodiscard]] core::u32 firstNonZeroWord(const core::u64 *row, core::u32 words) noexcept;

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_XORKERNEL_HPP
