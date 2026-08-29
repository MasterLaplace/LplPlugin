/**
 * @file Fold.hpp
 * @brief The one function that turns state into a signature.
 *
 * @warning Promoted because it was written TWICE in this module -- `Timeline.cpp` and `Chronicle.cpp`
 * each carried their own copy of the same two constants -- and a third consumer was about to make
 * it three. A signature has exactly one job: to be the same number on two machines. The function
 * that produces it is the last thing that should exist in several versions, because two copies
 * that drift produce two gates that disagree for a reason no measurement can name.
 *
 * FNV-1a, offset basis 0x811C9DC5, prime 0x01000193 -- the constants every gate in this project
 * folds with.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_FOLD_HPP
#    define LPL_LPL_HISTORY_FOLD_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::history {

/// FNV-1a offset basis. Every fold in this project starts here.
inline constexpr core::u32 kFoldOffsetBasis = 0x811C9DC5u;

/// FNV-1a prime.
inline constexpr core::u32 kFoldPrime = 0x01000193u;

/**
 * @brief Folds one 32-bit word into a running signature.
 *
 * @warning The WHOLE word, matching what `Chronicle::fold` and `Timeline::fold` already do -- and a first
 * version of this function folded byte by byte "for endianness", which was wrong twice over. The
 * arithmetic runs on the VALUE of a `core::u32`, not on its bytes, so it is already independent of
 * byte order; and a different mixing here would have produced different numbers from the two
 * folds already gated, quietly moving P13 and P18. Byte-wise folding is right for a byte BUFFER,
 * which is a different problem.
 *
 * @param signature The running value.
 * @param word      What to fold in.
 * @return The new value.
 */
[[nodiscard]] constexpr core::u32 foldWord(core::u32 signature, core::u32 word) noexcept
{
    return (signature ^ word) * kFoldPrime;
}

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_FOLD_HPP
