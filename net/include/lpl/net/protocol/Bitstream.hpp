/**
 * @file Bitstream.hpp
 * @brief Bit-level serialization stream for deterministic networking.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_BITSTREAM_HPP
#    define LPL_NET_PROTOCOL_BITSTREAM_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/NonCopyable.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/net/protocol/BitError.hpp>
#    include <lpl/std/vector.hpp>

#    include <optional>
#    include <span>

namespace lpl::net::protocol {

/**
 * @class Bitstream
 * @brief Compact bit-level read/write stream.
 *
 * Packs fields at arbitrary bit widths for minimal bandwidth. All
 * operations are deterministic and endian-safe (values stored big-endian
 * in the bit buffer).
 *
 * It owns a buffer that grows as fields are written, and leaves the packing
 * to BitWriter and BitReader, so the wire layout has one implementation, the
 * one the kernel compiles. A reader that reads a datagram where it lies, or a
 * writer into memory the caller owns, without the copy and the heap this
 * class needs, is BitReader or BitWriter itself.
 */
class Bitstream final : public core::NonCopyable<Bitstream> {
public:
    /**
     * @brief Constructs an empty writable bitstream.
     */
    Bitstream() noexcept;

    /**
     * @brief Constructs a read-only bitstream from existing data.
     * @param data   Raw bytes, copied.
     * @param bitCount Number of valid bits in @p data, cut to the bits @p data holds.
     */
    Bitstream(std::span<const core::byte> data, core::u32 bitCount) noexcept;

    ~Bitstream();

    /**
     * @name Write
     * @{
     */
    /**
     * @brief Writes @p bitCount bits from @p value.
     * @param value    Value to write (only lower @p bitCount bits are used).
     * @param bitCount Number of bits to write (1–32).
     */
    void writeBits(core::u32 value, core::u32 bitCount);

    /**
     * @brief Writes a boolean (1 bit).
     */
    void writeBool(bool value);

    /**
     * @brief Writes an unsigned 8-bit value.
     */
    void writeU8(core::u8 value);

    /**
     * @brief Writes an unsigned 16-bit value.
     */
    void writeU16(core::u16 value);

    /**
     * @brief Writes an unsigned 32-bit value.
     */
    void writeU32(core::u32 value);

    /**
     * @brief Writes a signed 32-bit value (bit-cast to u32).
     */
    void writeI32(core::i32 value);

    /**
     * @brief Writes a 32-bit float (bit-cast to u32).
     */
    void writeFloat(float value);

    /**
     * @brief Writes raw bytes (byte-aligned).
     */
    void writeBytes(std::span<const core::byte> bytes);
    /** @} */

    /**
     * @name Bit-packing / quantization (book §6.3.3)
     * @{
     */
    /**
     * @brief Writes @p value quantized into @p bits over the range [min, max].
     *
     * The value is clamped to the range, mapped to an integer in
     * [0, 2^bits - 1] (rounded to nearest, then held at the top edge when float
     * error would push it one past) and written with @p bits bits. A position
     * in a 1000 m world quantized on 16 bits keeps a resolution of ~15 mm for a
     * fraction of the 32-bit cost. Non-authoritative wire data only: the
     * quantization is lossy, so a quantized value never flows back into
     * authoritative Fixed32.
     *
     * @param value Value to encode.
     * @param min   Lower bound of the range.
     * @param max   Upper bound of the range (must exceed @p min).
     * @param bits  Bit width (1..32).
     */
    void writeQuantizedFloat(float value, float min, float max, core::u32 bits);

    /**
     * @brief Reads a value written by @ref writeQuantizedFloat with the same params.
     */
    [[nodiscard]] core::Expected<float> readQuantizedFloat(float min, float max, core::u32 bits);

    /**
     * @brief Writes an angle in radians quantized into @p bits over [0, 2*pi].
     *
     * Wraps @p radians into the circle first, so any real angle encodes. Ten
     * bits give ~0.35 degrees, enough for gameplay orientation at a third of a
     * raw float.
     */
    void writeAngle(float radians, core::u32 bits);

    /**
     * @brief Reads an angle written by @ref writeAngle, in [0, 2*pi].
     */
    [[nodiscard]] core::Expected<float> readAngle(core::u32 bits);

    /**
     * @brief Writes @p value as a LEB128 variable-length integer (7 bits/byte).
     *
     * Small ids and counts cost one byte instead of four; the top bit of each
     * byte is the continuation flag. Byte-granular, so it does not fight the
     * bit cursor beyond whole bytes.
     */
    void writeVarint(core::u32 value);

    /**
     * @brief Reads a LEB128 varint written by @ref writeVarint.
     */
    [[nodiscard]] core::Expected<core::u32> readVarint();
    /** @} */

    /**
     * @name Read
     * @{
     */
    /**
     * @brief Reads @p bitCount bits as an unsigned value.
     */
    [[nodiscard]] core::Expected<core::u32> readBits(core::u32 bitCount);

    /**
     * @brief Reads a boolean.
     */
    [[nodiscard]] core::Expected<bool> readBool();

    /**
     * @brief Reads an unsigned 8-bit value.
     */
    [[nodiscard]] core::Expected<core::u8> readU8();

    /**
     * @brief Reads an unsigned 16-bit value.
     */
    [[nodiscard]] core::Expected<core::u16> readU16();

    /**
     * @brief Reads an unsigned 32-bit value.
     */
    [[nodiscard]] core::Expected<core::u32> readU32();

    /**
     * @brief Reads a signed 32-bit value.
     */
    [[nodiscard]] core::Expected<core::i32> readI32();

    /**
     * @brief Reads a 32-bit float.
     */
    [[nodiscard]] core::Expected<float> readFloat();

    /**
     * @brief Reads @p count raw bytes.
     *
     * @details A count the stream does not hold, as a forged size from the wire asks, is refused before
     *          anything is allocated for it.
     */
    [[nodiscard]] core::Expected<lpl::pmr::vector<core::byte>> readBytes(core::u32 count);
    /** @} */

    /**
     * @name Query
     * @{
     */
    /**
     * @brief Returns the total number of written bits.
     */
    [[nodiscard]] core::u32 bitsWritten() const noexcept;

    /**
     * @brief Returns the number of bits remaining for reading.
     */
    [[nodiscard]] core::u32 bitsRemaining() const noexcept;

    /**
     * @brief Returns the underlying byte buffer.
     */
    [[nodiscard]] std::span<const core::byte> data() const noexcept;

    /**
     * @brief The first field this stream refused to write, or nothing.
     *
     * @details A field refused by the core, a width outside 1..32 or a quantization range that is not
     *          one, is logged and kept here, and every field written after it is refused too: nothing
     *          lands where the refused field should have been. reset() clears it.
     */
    [[nodiscard]] std::optional<BitError> failure() const noexcept;

    /**
     * @brief Resets read/write cursors to the beginning, and a writable stream's failure.
     */
    void reset() noexcept;
    /** @} */

private:
    template <typename Write> void writeThroughCore(core::u64 maximumBits, Write write);
    template <typename Value, typename Read> [[nodiscard]] core::Expected<Value> readThroughCore(Read read);

    lpl::pmr::vector<core::byte> _buffer; /**< The bytes written, or a copy of the ones read. */
    core::u32 _writeBit{0};               /**< Bits written. */
    core::u32 _readBit{0};                /**< Bits read. */
    core::u32 _totalBits{0};              /**< Valid bits, which a read never passes. */
    bool _readOnly{false};                /**< Whether the stream was built from bytes to read. */
    std::optional<BitError> _failure;     /**< The first field refused, which stops every later write. */
};

} // namespace lpl::net::protocol

#endif // LPL_NET_PROTOCOL_BITSTREAM_HPP
