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

#    include <cstddef>
#    include <cstring>
#    include <span>
#    include <vector>

namespace lpl::net::protocol {

/**
 * @class Bitstream
 * @brief Compact bit-level read/write stream.
 *
 * Packs fields at arbitrary bit widths for minimal bandwidth. All
 * operations are deterministic and endian-safe (values stored big-endian
 * in the bit buffer).
 */
class Bitstream final : public core::NonCopyable<Bitstream> {
public:
    /** @brief Constructs an empty writable bitstream. */
    Bitstream() noexcept;

    /**
     * @brief Constructs a read-only bitstream from existing data.
     * @param data   Raw bytes.
     * @param bitCount Number of valid bits in @p data.
     */
    Bitstream(std::span<const core::byte> data, core::u32 bitCount) noexcept;

    ~Bitstream();

    // --------------------------------------------------------------------- //
    //  Write                                                                 //
    // --------------------------------------------------------------------- //

    /**
     * @brief Writes @p bitCount bits from @p value.
     * @param value    Value to write (only lower @p bitCount bits are used).
     * @param bitCount Number of bits to write (1–32).
     */
    void writeBits(core::u32 value, core::u32 bitCount);

    /** @brief Writes a boolean (1 bit). */
    void writeBool(bool value);

    /** @brief Writes an unsigned 8-bit value. */
    void writeU8(core::u8 value);

    /** @brief Writes an unsigned 16-bit value. */
    void writeU16(core::u16 value);

    /** @brief Writes an unsigned 32-bit value. */
    void writeU32(core::u32 value);

    /** @brief Writes a signed 32-bit value (bit-cast to u32). */
    void writeI32(core::i32 value);

    /** @brief Writes a 32-bit float (bit-cast to u32). */
    void writeFloat(float value);

    /** @brief Writes raw bytes (byte-aligned). */
    void writeBytes(std::span<const core::byte> bytes);

    // --------------------------------------------------------------------- //
    //  Bit-packing / quantization (book §6.3.3)                              //
    // --------------------------------------------------------------------- //

    /**
     * @brief Writes @p value quantized into @p bits over the range [min, max].
     *
     * The value is clamped to the range, mapped to an integer in
     * [0, 2^bits - 1] and written with @p bits bits. A position in a 1000 m
     * world quantized on 16 bits keeps a resolution of ~15 mm for a fraction of
     * the 32-bit cost. Non-authoritative wire data only: the quantization is
     * lossy, so a quantized value never flows back into authoritative Fixed32.
     *
     * @param value Value to encode.
     * @param min   Lower bound of the range.
     * @param max   Upper bound of the range (must exceed @p min).
     * @param bits  Bit width (1..32).
     */
    void writeQuantizedFloat(float value, float min, float max, core::u32 bits);

    /** @brief Reads a value written by @ref writeQuantizedFloat with the same params. */
    [[nodiscard]] core::Expected<float> readQuantizedFloat(float min, float max, core::u32 bits);

    /**
     * @brief Writes an angle in radians quantized into @p bits over [0, 2*pi).
     *
     * Wraps @p radians into the circle first, so any real angle encodes. Ten
     * bits give ~0.35 degrees, enough for gameplay orientation at a third of a
     * raw float.
     */
    void writeAngle(float radians, core::u32 bits);

    /** @brief Reads an angle written by @ref writeAngle, in [0, 2*pi). */
    [[nodiscard]] core::Expected<float> readAngle(core::u32 bits);

    /**
     * @brief Writes @p value as a LEB128 variable-length integer (7 bits/byte).
     *
     * Small ids and counts cost one byte instead of four; the top bit of each
     * byte is the continuation flag. Byte-granular, so it does not fight the
     * bit cursor beyond whole bytes.
     */
    void writeVarint(core::u32 value);

    /** @brief Reads a LEB128 varint written by @ref writeVarint. */
    [[nodiscard]] core::Expected<core::u32> readVarint();

    // --------------------------------------------------------------------- //
    //  Read                                                                  //
    // --------------------------------------------------------------------- //

    /** @brief Reads @p bitCount bits as an unsigned value. */
    [[nodiscard]] core::Expected<core::u32> readBits(core::u32 bitCount);

    /** @brief Reads a boolean. */
    [[nodiscard]] core::Expected<bool> readBool();

    /** @brief Reads an unsigned 8-bit value. */
    [[nodiscard]] core::Expected<core::u8> readU8();

    /** @brief Reads an unsigned 16-bit value. */
    [[nodiscard]] core::Expected<core::u16> readU16();

    /** @brief Reads an unsigned 32-bit value. */
    [[nodiscard]] core::Expected<core::u32> readU32();

    /** @brief Reads a signed 32-bit value. */
    [[nodiscard]] core::Expected<core::i32> readI32();

    /** @brief Reads a 32-bit float. */
    [[nodiscard]] core::Expected<float> readFloat();

    /** @brief Reads @p count raw bytes. */
    [[nodiscard]] core::Expected<std::vector<core::byte>> readBytes(core::u32 count);

    // --------------------------------------------------------------------- //
    //  Query                                                                 //
    // --------------------------------------------------------------------- //

    /** @brief Returns the total number of written bits. */
    [[nodiscard]] core::u32 bitsWritten() const noexcept;

    /** @brief Returns the number of bits remaining for reading. */
    [[nodiscard]] core::u32 bitsRemaining() const noexcept;

    /** @brief Returns the underlying byte buffer. */
    [[nodiscard]] std::span<const core::byte> data() const noexcept;

    /** @brief Resets read/write cursors to the beginning. */
    void reset() noexcept;

private:
    std::vector<core::byte> _buffer;
    core::u32 _writeBit{0};
    core::u32 _readBit{0};
    core::u32 _totalBits{0};
    bool _readOnly{false};
};

} // namespace lpl::net::protocol

#endif // LPL_NET_PROTOCOL_BITSTREAM_HPP
