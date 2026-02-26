// /////////////////////////////////////////////////////////////////////////////
/// @file Bitstream.hpp
/// @brief Bit-level serialization stream for deterministic networking.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

namespace lpl::net::protocol {

// /////////////////////////////////////////////////////////////////////////////
/// @class Bitstream
/// @brief Compact bit-level read/write stream.
///
/// Packs fields at arbitrary bit widths for minimal bandwidth. All
/// operations are deterministic and endian-safe (values stored big-endian
/// in the bit buffer).
// /////////////////////////////////////////////////////////////////////////////
class Bitstream final : public core::NonCopyable<Bitstream>
{
public:
    /// @brief Constructs an empty writable bitstream.
    Bitstream() noexcept;

    /// @brief Constructs a read-only bitstream from existing data.
    /// @param data   Raw bytes.
    /// @param bitCount Number of valid bits in @p data.
    Bitstream(std::span<const core::byte> data, core::u32 bitCount) noexcept;

    ~Bitstream();

    // --------------------------------------------------------------------- //
    //  Write                                                                 //
    // --------------------------------------------------------------------- //

    /// @brief Writes @p bitCount bits from @p value.
    /// @param value    Value to write (only lower @p bitCount bits are used).
    /// @param bitCount Number of bits to write (1–32).
    void writeBits(core::u32 value, core::u32 bitCount);

    /// @brief Writes a boolean (1 bit).
    void writeBool(bool value);

    /// @brief Writes an unsigned 8-bit value.
    void writeU8(core::u8 value);

    /// @brief Writes an unsigned 16-bit value.
    void writeU16(core::u16 value);

    /// @brief Writes an unsigned 32-bit value.
    void writeU32(core::u32 value);

    /// @brief Writes raw bytes (byte-aligned).
    void writeBytes(std::span<const core::byte> bytes);

    // --------------------------------------------------------------------- //
    //  Read                                                                  //
    // --------------------------------------------------------------------- //

    /// @brief Reads @p bitCount bits as an unsigned value.
    [[nodiscard]] core::Expected<core::u32> readBits(core::u32 bitCount);

    /// @brief Reads a boolean.
    [[nodiscard]] core::Expected<bool> readBool();

    /// @brief Reads an unsigned 8-bit value.
    [[nodiscard]] core::Expected<core::u8> readU8();

    /// @brief Reads an unsigned 16-bit value.
    [[nodiscard]] core::Expected<core::u16> readU16();

    /// @brief Reads an unsigned 32-bit value.
    [[nodiscard]] core::Expected<core::u32> readU32();

    /// @brief Reads @p count raw bytes.
    [[nodiscard]] core::Expected<std::vector<core::byte>> readBytes(core::u32 count);

    // --------------------------------------------------------------------- //
    //  Query                                                                 //
    // --------------------------------------------------------------------- //

    /// @brief Returns the total number of written bits.
    [[nodiscard]] core::u32 bitsWritten() const noexcept;

    /// @brief Returns the number of bits remaining for reading.
    [[nodiscard]] core::u32 bitsRemaining() const noexcept;

    /// @brief Returns the underlying byte buffer.
    [[nodiscard]] std::span<const core::byte> data() const noexcept;

    /// @brief Resets read/write cursors to the beginning.
    void reset() noexcept;

private:
    std::vector<core::byte> buffer_;
    core::u32               writeBit_{0};
    core::u32               readBit_{0};
    core::u32               totalBits_{0};
    bool                    readOnly_{false};
};

} // namespace lpl::net::protocol
