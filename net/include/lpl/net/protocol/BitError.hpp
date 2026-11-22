/**
 * @file BitError.hpp
 * @brief Why a BitWriter or a BitReader refused a field.
 *
 * @author MasterLaplace
 * @version 0.0.0
 * @date 2026-11-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_BITERROR_HPP
#    define LPL_NET_PROTOCOL_BITERROR_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::net::protocol {

/**
 * @brief Why a field was refused, written whole or not at all, read whole or not at all.
 *
 * Zero names no error, so it is not a value of this type: a writer that has refused nothing holds
 * an empty optional, a read that succeeded holds its value.
 */
enum class BitError : core::u8 {
    kOverflow = 1,  /**< A write would pass the end of the destination. */
    kUnderflow,     /**< A read would pass the last valid bit of the source. */
    kInvalidWidth,  /**< A width outside 1..32 bits. */
    kInvalidRange,  /**< A quantization range that is empty, reversed, not a number or wider than a float. */
    kVarintTooLong, /**< A varint of more than five bytes, or whose value does not fit 32 bits. */
};

} // namespace lpl::net::protocol

#endif /* LPL_NET_PROTOCOL_BITERROR_HPP */
