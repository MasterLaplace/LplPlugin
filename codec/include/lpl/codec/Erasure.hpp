/**
 * @file Erasure.hpp
 * @brief The façade the rest of the engine calls.
 *
 * Encode a block into n+m shards, decode from any sufficient subset. Callers in
 * net/ and pack/ should never need to know whether the answer came from peeling
 * or from elimination.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_ERASURE_HPP
#    define LPL_LPL_CODEC_ERASURE_HPP

#    include <lpl/codec/Fountain.hpp>
#    include <lpl/codec/Peeling.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::codec {

/**
 * @struct ErasureParams
 * @brief What an encode is asked for.
 */
struct ErasureParams {
    core::u32 blockBytes{32u};       ///< Payload bytes per droplet.
    core::u32 overheadPermille{50u}; ///< Droplets beyond K, in thousandths. 50 is the paper's 5 %.
    core::u32 firstSeed{1u};         ///< Seed of the first droplet; they increase by one.
    SolitonParams tuning{};          ///< c and delta; sourceBlocks is derived.
};

/**
 * @struct ErasureShape
 * @brief How a payload was cut up, so a decoder can put it back.
 *
 * Three numbers, and they are the whole header a stream needs. The list of which
 * source blocks each droplet combines is NOT here, and that absence is the point:
 * it is recoverable from the droplet's own seed.
 */
struct ErasureShape {
    core::u32 blockCount{0u};    ///< K.
    core::u32 blockBytes{0u};    ///< Bytes per block.
    core::u32 originalBytes{0u}; ///< Payload length before padding.
};

/**
 * @brief Cuts @p payload into blocks and emits droplets over them.
 *
 * The last block is zero-padded, and the true length travels in @ref ErasureShape
 * rather than in a length prefix inside the payload: a prefix would be part of the
 * data the code protects, so losing the droplets that carry it would cost the length
 * of everything else too.
 *
 * @param payload    Bytes to protect.
 * @param bytes      Length of @p payload.
 * @param params     Block size, overhead and tuning.
 * @param outShape   Receives how it was cut.
 * @param outDroplets Receives the droplets, in emission order.
 * @return false when the payload is empty or the parameters are degenerate.
 */
[[nodiscard]] bool encodeErasure(const core::u8 *payload, core::u32 bytes, const ErasureParams &params,
                                 ErasureShape &outShape, lpl::pmr::vector<Droplet> &outDroplets);

/**
 * @brief Rebuilds a payload from whatever droplets survived.
 *
 * Any sufficient subset will do, in any order — that is what rateless means, and it
 * is why this takes no notion of "which" droplets are missing. A caller that lost
 * half of them simply passes the other half.
 *
 * @param droplets  What arrived.
 * @param shape     What @ref encodeErasure reported.
 * @param params    The same tuning the encoder used; a different one reads different seeds.
 * @param outBytes  Receives @c originalBytes bytes on success.
 * @param outReport Receives how the decode went.
 * @return true when the payload was recovered in full.
 */
[[nodiscard]] bool decodeErasure(const lpl::pmr::vector<Droplet> &droplets, const ErasureShape &shape,
                                 const ErasureParams &params, lpl::pmr::vector<core::u8> &outBytes,
                                 DecodeReport &outReport);

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_ERASURE_HPP
