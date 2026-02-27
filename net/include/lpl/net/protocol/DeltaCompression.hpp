/**
 * @file DeltaCompression.hpp
 * @brief XOR-based delta compression / decompression for snapshots.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_DELTACOMPRESSION_HPP
    #define LPL_NET_PROTOCOL_DELTACOMPRESSION_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

#include <span>
#include <vector>

namespace lpl::net::protocol {

/**
 * @class DeltaCompression
 * @brief Stateless XOR delta encoder/decoder for deterministic state
 *        snapshots.
 *
 * Given a baseline state and a new state (same size), produces a delta
 * containing only the differing bytes (zero-run-length encoded for
 * bandwidth minimisation).
 */
class DeltaCompression
{
public:
    /**
     * @brief Encodes the delta between @p baseline and @p current.
     * @param baseline Previous snapshot bytes.
     * @param current  Current snapshot bytes.
     * @return Compressed delta buffer.
     */
    [[nodiscard]] static core::Expected<std::vector<core::byte>> encode(
        std::span<const core::byte> baseline,
        std::span<const core::byte> current);

    /**
     * @brief Applies a delta on top of @p baseline to reconstruct the
     *        current snapshot.
     * @param baseline Previous state.
     * @param delta    Delta produced by @ref encode.
     * @return Reconstructed current state.
     */
    [[nodiscard]] static core::Expected<std::vector<core::byte>> decode(
        std::span<const core::byte> baseline,
        std::span<const core::byte> delta);
};

} // namespace lpl::net::protocol

#endif // LPL_NET_PROTOCOL_DELTACOMPRESSION_HPP
