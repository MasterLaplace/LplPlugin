/**
 * @file BitWriter.hpp
 * @brief Packs fields at any bit width into memory its caller owns, without a heap.
 *
 * @author MasterLaplace
 * @version 0.0.0
 * @date 2026-11-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_BITWRITER_HPP
#    define LPL_NET_PROTOCOL_BITWRITER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/net/protocol/BitError.hpp>

#    include <optional>
#    include <span>

namespace lpl::net::protocol {

/**
 * @class BitWriter
 * @brief Writes fields at any width from 1 to 32 bits into a span its caller owns.
 *
 * The wire layout is the one Bitstream has always written: the first bit written is the top bit of
 * the first byte, and a value goes out from its top bit down, so a field is big-endian whatever the
 * machine. The bits after the last field, up to the end of its byte, are zero, whatever the
 * destination held before.
 *
 * A field is written whole or not at all. The first field refused (no room, a width outside 1..32, a
 * range that is not one) is kept as the writer's failure, and every write after it is refused too:
 * the caller checks failure() once, before it publishes what was written. Nothing here allocates,
 * copies the destination or halts, so it runs in ring 0 and in the tick.
 */
class BitWriter final {
public:
    /**
     * @brief Writes from the first bit of @p destination.
     *
     * @param destination Memory the fields are written into, which must outlive the writer. Its
     *                    first 2^29 - 1 bytes at most are used, so that a count of bits holds in 32
     *                    bits.
     */
    explicit BitWriter(std::span<core::byte> destination) noexcept;

    /**
     * @brief Writes from bit @p bitOffset of @p destination, keeping the bits before it.
     *
     * @param destination Memory the fields are written into, which must outlive the writer.
     * @param bitOffset   Bits already written there. Past the end of @p destination, the writer
     *                    starts failed, with BitError::kOverflow.
     */
    BitWriter(std::span<core::byte> destination, core::u32 bitOffset) noexcept;

    /**
     * @name Fields
     * @{
     */
    /**
     * @brief Writes the low @p bitCount bits of @p value, its top bit first.
     *
     * @param value    Value to write; the bits above @p bitCount are ignored.
     * @param bitCount Width in 1..32; another width is refused with BitError::kInvalidWidth.
     */
    void writeBits(core::u32 value, core::u32 bitCount) noexcept;

    /**
     * @brief Writes @p value on one bit.
     */
    void writeBool(bool value) noexcept;

    /**
     * @brief Writes @p value on 8 bits.
     */
    void writeU8(core::u8 value) noexcept;

    /**
     * @brief Writes @p value on 16 bits.
     */
    void writeU16(core::u16 value) noexcept;

    /**
     * @brief Writes @p value on 32 bits.
     */
    void writeU32(core::u32 value) noexcept;

    /**
     * @brief Writes the two's complement bits of @p value on 32 bits.
     */
    void writeI32(core::i32 value) noexcept;

    /**
     * @brief Writes the IEEE 754 bits of @p value on 32 bits, a NaN's payload included.
     */
    void writeFloat(float value) noexcept;

    /**
     * @brief Writes @p bytes, 8 bits each, as one field: all of them or none.
     */
    void writeBytes(std::span<const core::byte> bytes) noexcept;

    /**
     * @brief Writes @p value as a LEB128 varint, 7 bits a byte, low group first, as one field.
     *
     * @details A value below 128 costs one byte, a full 32-bit value five.
     */
    void writeVarint(core::u32 value) noexcept;

    /**
     * @brief Writes @p value quantized on @p bits bits over the range [@p minimum, @p maximum].
     *
     * @details See quantize(): the value is clamped to the range and takes the nearest step. A
     *          position in a 1000 m world on 16 bits keeps a step of about 15 mm.
     *
     * @param value   Value to write.
     * @param minimum Lower bound of the range.
     * @param maximum Upper bound; one that does not exceed @p minimum, or whose distance to it passes the
     *                largest float, is refused with BitError::kInvalidRange.
     * @param bits    Width in 1..32.
     */
    void writeQuantizedFloat(float value, float minimum, float maximum, core::u32 bits) noexcept;

    /**
     * @brief Writes @p radians wrapped into [0, 2*pi), quantized on @p bits bits.
     *
     * @details Ten bits give a step of about 0.35 degree.
     */
    void writeAngle(float radians, core::u32 bits) noexcept;
    /** @} */

    /**
     * @name State
     * @{
     */
    /**
     * @brief Bits written, from the first bit of the destination.
     */
    [[nodiscard]] core::u32 bitsWritten() const noexcept;

    /**
     * @brief Bits left before the end of the destination.
     */
    [[nodiscard]] core::u32 bitsRemaining() const noexcept;

    /**
     * @brief The bytes the fields cover so far, the last one completed with zero bits.
     */
    [[nodiscard]] std::span<const core::byte> bytes() const noexcept;

    /**
     * @brief The first field refused, or nothing when every field was written.
     */
    [[nodiscard]] std::optional<BitError> failure() const noexcept;
    /** @} */

private:
    [[nodiscard]] static constexpr core::u32 capacityInBits(core::usize bytes) noexcept;
    [[nodiscard]] static constexpr core::u32 varintBytes(core::u32 value) noexcept;
    [[nodiscard]] bool admit(core::u32 bitCount) noexcept;
    void refuse(BitError error) noexcept;
    void put(core::u32 value, core::u32 bitCount) noexcept;

    std::span<core::byte> _destination; /**< Where the fields go. */
    core::u32 _capacityBits;            /**< Bits the destination holds. */
    core::u32 _cursor;                  /**< Bits written, from the first bit of the destination. */
    std::optional<BitError> _failure;   /**< The first field refused. */
};

} // namespace lpl::net::protocol

#    include "BitWriter.inl"

#endif /* LPL_NET_PROTOCOL_BITWRITER_HPP */
