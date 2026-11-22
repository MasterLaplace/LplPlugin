#ifndef LPL_NET_PROTOCOL_BITWRITER_INL
#define LPL_NET_PROTOCOL_BITWRITER_INL

#include <lpl/net/protocol/Quantization.hpp>

#include <bit>
#include <limits>

namespace lpl::net::protocol {

constexpr core::u32 BitWriter::capacityInBits(core::usize bytes) noexcept
{
    constexpr core::usize kLargestDestination = 0x1FFFFFFFu;

    return static_cast<core::u32>(bytes > kLargestDestination ? kLargestDestination : bytes) * 8u;
}

constexpr core::u32 BitWriter::varintBytes(core::u32 value) noexcept
{
    core::u32 count = 1u;

    while ((value >>= 7u) != 0u)
        ++count;
    return count;
}

inline BitWriter::BitWriter(std::span<core::byte> destination) noexcept : BitWriter{destination, 0u} {}

inline BitWriter::BitWriter(std::span<core::byte> destination, core::u32 bitOffset) noexcept
    : _destination{destination}, _capacityBits{capacityInBits(destination.size())}, _cursor{bitOffset}, _failure{}
{
    if (bitOffset <= _capacityBits)
        return;
    _cursor = 0u;
    refuse(BitError::kOverflow);
}

inline void BitWriter::refuse(BitError error) noexcept
{
    if (!_failure.has_value())
        _failure = error;
}

inline bool BitWriter::admit(core::u32 bitCount) noexcept
{
    if (_failure.has_value())
        return false;
    if (bitCount <= _capacityBits - _cursor)
        return true;
    refuse(BitError::kOverflow);
    return false;
}

inline void BitWriter::storeTopBytes(core::byte *first, core::usize room, core::u32 word, core::u32 count) noexcept
{
    if (room >= 4u)
    {
        first[0] = static_cast<core::byte>(static_cast<core::u8>(word >> 24u));
        first[1] = static_cast<core::byte>(static_cast<core::u8>(word >> 16u));
        first[2] = static_cast<core::byte>(static_cast<core::u8>(word >> 8u));
        first[3] = static_cast<core::byte>(static_cast<core::u8>(word));
        return;
    }
    for (core::u32 index = 0u; index < count; ++index)
        first[index] = static_cast<core::byte>(static_cast<core::u8>(word >> (24u - 8u * index)));
}

inline void BitWriter::put(core::u32 value, core::u32 bitCount) noexcept
{
    const core::u32 used = _cursor & 7u;
    const core::u32 end = used + bitCount;
    const core::usize byteIndex = _cursor >> 3u;
    core::byte *const first = _destination.data() + byteIndex;
    const core::usize room = byteIndex < _destination.size() ? _destination.size() - byteIndex : 0u;
    const core::u32 field = value & (0xFFFFFFFFu >> (32u - bitCount));
    const core::u32 kept = (static_cast<core::u32>(first[0]) & (0xFF00u >> used)) << 24u;

    _cursor += bitCount;
    if (end <= 32u)
    {
        storeTopBytes(first, room, kept | (field << (32u - end)), (end + 7u) >> 3u);
        return;
    }
    storeTopBytes(first, room, kept | (field >> (end - 32u)), 4u);
    first[4] = static_cast<core::byte>(static_cast<core::u8>(field << (40u - end)));
}

inline void BitWriter::writeBits(core::u32 value, core::u32 bitCount) noexcept
{
    if (bitCount == 0u || bitCount > 32u)
    {
        refuse(BitError::kInvalidWidth);
        return;
    }
    if (admit(bitCount))
        put(value, bitCount);
}

inline void BitWriter::writeBool(bool value) noexcept { writeBits(value ? 1u : 0u, 1u); }

inline void BitWriter::writeU8(core::u8 value) noexcept { writeBits(value, 8u); }

inline void BitWriter::writeU16(core::u16 value) noexcept { writeBits(value, 16u); }

inline void BitWriter::writeU32(core::u32 value) noexcept { writeBits(value, 32u); }

inline void BitWriter::writeI32(core::i32 value) noexcept { writeBits(std::bit_cast<core::u32>(value), 32u); }

inline void BitWriter::writeFloat(float value) noexcept { writeBits(std::bit_cast<core::u32>(value), 32u); }

inline void BitWriter::writeBytes(std::span<const core::byte> bytes) noexcept
{
    if (_failure.has_value())
        return;
    if (bytes.size() > (_capacityBits - _cursor) / 8u)
    {
        refuse(BitError::kOverflow);
        return;
    }
    for (const core::byte octet : bytes)
        put(static_cast<core::u32>(octet), 8u);
}

inline void BitWriter::writeVarint(core::u32 value) noexcept
{
    if (!admit(varintBytes(value) * 8u))
        return;
    do
    {
        const core::u32 septet = value & 0x7Fu;

        value >>= 7u;
        put(value != 0u ? (septet | 0x80u) : septet, 8u);
    } while (value != 0u);
}

inline void BitWriter::writeQuantizedFloat(float value, float minimum, float maximum, core::u32 bits) noexcept
{
    if (bits == 0u || bits > 32u)
    {
        refuse(BitError::kInvalidWidth);
        return;
    }
    if (!(maximum > minimum) || !(maximum - minimum <= std::numeric_limits<float>::max()))
    {
        refuse(BitError::kInvalidRange);
        return;
    }
    if (admit(bits))
        put(quantize(value, minimum, maximum, bits), bits);
}

inline void BitWriter::writeAngle(float radians, core::u32 bits) noexcept
{
    writeQuantizedFloat(wrapAngle(radians), 0.0f, kTwoPi, bits);
}

inline core::u32 BitWriter::bitsWritten() const noexcept { return _cursor; }

inline core::u32 BitWriter::bitsRemaining() const noexcept { return _capacityBits - _cursor; }

inline std::span<const core::byte> BitWriter::bytes() const noexcept
{
    return std::span<const core::byte>{_destination.data(), (_cursor + 7u) >> 3u};
}

inline std::optional<BitError> BitWriter::failure() const noexcept { return _failure; }

} // namespace lpl::net::protocol

#endif /* LPL_NET_PROTOCOL_BITWRITER_INL */
