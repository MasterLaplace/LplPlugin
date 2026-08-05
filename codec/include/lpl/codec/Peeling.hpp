/**
 * @file Peeling.hpp
 * @brief Belief propagation, and the Gaussian fallback.
 *
 * Find a degree-one droplet, resolve it, XOR it out of every droplet that
 * contained it, repeat. O(K log K) when it works; when the chain stalls, the
 * residual system goes to GaussJordan. The fallback is not an error path, it is
 * the normal tail of the decode.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_PEELING_HPP
#    define LPL_LPL_CODEC_PEELING_HPP

#    include <lpl/codec/Fountain.hpp>
#    include <lpl/codec/Prng.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::codec {

/**
 * @struct DecodeReport
 * @brief How the decode went, in numbers a test can compare.
 *
 * Reported rather than inferred, because "it worked" and "it worked the way it was
 * supposed to" are different claims: a decode that recovered everything through the
 * Gaussian fallback has the same output and a completely different cost, and only
 * these counters tell the two apart.
 */
struct DecodeReport {
    bool recovered{false};          ///< Every source block resolved.
    core::u32 dropletsUsed{0u};     ///< Droplets handed to the decoder.
    core::u32 peeledBlocks{0u};     ///< Blocks belief propagation resolved on its own.
    core::u32 eliminatedBlocks{0u}; ///< Blocks the Gaussian fallback had to finish.
    core::u32 residualRows{0u};     ///< Rows the fallback was given; 0 when peeling sufficed.
    core::u32 rank{0u};             ///< Rank of the residual system.
};

/**
 * @brief Rebuilds the source from a bag of droplets.
 *
 * Peeling first, elimination for whatever is left. The split is not an optimisation
 * with an error path bolted on: the robust soliton is tuned so that peeling resolves
 * almost everything and stalls near the end, so the fallback IS the normal tail of a
 * successful decode, and a decoder without it fails on the last two or three blocks
 * of most runs.
 *
 * @param droplets    What arrived. Order is respected, so two runs of the same bag
 *                    fold identically.
 * @param table       The distribution the encoder used; the seeds mean nothing else.
 * @param blockCount  K.
 * @param blockBytes  Bytes per block.
 * @param outBlocks   Receives K * blockBytes bytes.
 * @param outReport   Receives the counters.
 * @return true when every block was recovered.
 */
[[nodiscard]] bool decodeDroplets(const lpl::pmr::vector<Droplet> &droplets, const SolitonTable &table,
                                  core::u32 blockCount, core::u32 blockBytes, lpl::pmr::vector<core::u8> &outBlocks,
                                  DecodeReport &outReport);

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_PEELING_HPP
