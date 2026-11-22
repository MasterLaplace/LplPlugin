/**
 * @file BitReader.hpp
 * @brief Reads fields at any bit width from memory its caller owns, where it lies.
 *
 * @author MasterLaplace
 * @version 0.0.0
 * @date 2026-11-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_BITREADER_HPP
#    define LPL_NET_PROTOCOL_BITREADER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/net/protocol/BitError.hpp>

#    include <expected>
#    include <span>

namespace lpl::net::protocol {

/**
 * @class BitReader
 * @brief Reads, from a span its caller owns, the fields a BitWriter wrote.
 *
 * The source is read where it lies, never copied: a datagram where the network card wrote it, a
 * packet in a transport's ring. Its bytes come from anyone, so every read is checked against the
 * bits the source holds and every width against 1..32, in every build. A read is whole or refused:
 * a refused read returns its BitError and leaves the cursor where it was, so the caller can stop at
 * the first field that does not hold.
 */
class BitReader final {
public:
    /**
     * @brief Reads from the first bit of @p source.
     *
     * @param source   Bytes to read, which must outlive the reader.
     * @param bitCount Valid bits in @p source, from its first bit. A count larger than @p source holds
     *                 is cut to its bytes, so no read goes past them.
     */
    BitReader(std::span<const core::byte> source, core::u32 bitCount) noexcept;

    /**
     * @brief Reads from bit @p bitOffset of @p source.
     *
     * @param source    Bytes to read, which must outlive the reader.
     * @param bitCount  Valid bits in @p source, from its first bit, cut to the bits @p source holds.
     * @param bitOffset Bits already read. Past @p bitCount, nothing is left to read.
     */
    BitReader(std::span<const core::byte> source, core::u32 bitCount, core::u32 bitOffset) noexcept;

    /**
     * @name Fields
     * @{
     */
    /**
     * @brief Reads @p bitCount bits as an unsigned value, the first bit read at its top.
     *
     * @param bitCount Width in 1..32.
     * @return The value, BitError::kInvalidWidth for another width, or BitError::kUnderflow when
     *         the source holds fewer bits.
     */
    [[nodiscard]] std::expected<core::u32, BitError> readBits(core::u32 bitCount) noexcept;

    /**
     * @brief Reads one bit.
     */
    [[nodiscard]] std::expected<bool, BitError> readBool() noexcept;

    /**
     * @brief Reads 8 bits.
     */
    [[nodiscard]] std::expected<core::u8, BitError> readU8() noexcept;

    /**
     * @brief Reads 16 bits.
     */
    [[nodiscard]] std::expected<core::u16, BitError> readU16() noexcept;

    /**
     * @brief Reads 32 bits.
     */
    [[nodiscard]] std::expected<core::u32, BitError> readU32() noexcept;

    /**
     * @brief Reads 32 bits as a two's complement value.
     */
    [[nodiscard]] std::expected<core::i32, BitError> readI32() noexcept;

    /**
     * @brief Reads 32 bits as an IEEE 754 float, a NaN's payload included.
     */
    [[nodiscard]] std::expected<float, BitError> readFloat() noexcept;

    /**
     * @brief Reads as many bytes as @p destination holds, 8 bits each, all of them or none.
     *
     * @return Nothing, or BitError::kUnderflow, with @p destination untouched.
     */
    [[nodiscard]] std::expected<void, BitError> readBytes(std::span<core::byte> destination) noexcept;

    /**
     * @brief Reads a LEB128 varint written by BitWriter::writeVarint().
     *
     * @return The value, BitError::kVarintTooLong for more than five bytes or a value that does not
     *         fit 32 bits, or BitError::kUnderflow when the source ends inside it.
     */
    [[nodiscard]] std::expected<core::u32, BitError> readVarint() noexcept;

    /**
     * @brief Reads a value written by BitWriter::writeQuantizedFloat() with the same parameters.
     *
     * @param minimum Lower bound of the range.
     * @param maximum Upper bound of the range.
     * @param bits    Width in 1..32: a width read from the wire is checked here.
     * @return The value, BitError::kInvalidWidth or BitError::kUnderflow.
     */
    [[nodiscard]] std::expected<float, BitError> readQuantizedFloat(float minimum, float maximum,
                                                                    core::u32 bits) noexcept;

    /**
     * @brief Reads an angle written by BitWriter::writeAngle(), in [0, 2*pi].
     */
    [[nodiscard]] std::expected<float, BitError> readAngle(core::u32 bits) noexcept;
    /** @} */

    /**
     * @name State
     * @{
     */
    /**
     * @brief Bits read, from the first bit of the source.
     */
    [[nodiscard]] core::u32 bitsRead() const noexcept;

    /**
     * @brief Valid bits left to read.
     */
    [[nodiscard]] core::u32 bitsRemaining() const noexcept;
    /** @} */

private:
    [[nodiscard]] static constexpr core::u32 bitsHeld(core::usize bytes, core::u32 bitCount) noexcept;
    [[nodiscard]] bool holds(core::u32 bitCount) const noexcept;
    [[nodiscard]] static core::u32 loadTopBytes(std::span<const core::byte> rest) noexcept;
    [[nodiscard]] core::u32 take(core::u32 bitCount) noexcept;

    std::span<const core::byte> _source; /**< Where the fields are read. */
    core::u32 _bitCount;                 /**< Valid bits in the source. */
    core::u32 _cursor;                   /**< Bits read, from the first bit of the source. */
};

} // namespace lpl::net::protocol

#    include "BitReader.inl"

#endif /* LPL_NET_PROTOCOL_BITREADER_HPP */
