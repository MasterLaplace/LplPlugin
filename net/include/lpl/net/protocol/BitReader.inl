#ifndef LPL_NET_PROTOCOL_BITREADER_INL
#define LPL_NET_PROTOCOL_BITREADER_INL

#include <lpl/net/protocol/Quantization.hpp>

#include <bit>

namespace lpl::net::protocol {

constexpr core::u32 BitReader::bitsHeld(core::usize bytes, core::u32 bitCount) noexcept
{
    const core::usize bytesNeeded = bitCount / 8u + ((bitCount & 7u) != 0u ? 1u : 0u);

    return bytes >= bytesNeeded ? bitCount : static_cast<core::u32>(bytes) * 8u;
}

inline BitReader::BitReader(std::span<const core::byte> source, core::u32 bitCount) noexcept
    : BitReader{source, bitCount, 0u}
{
}

inline BitReader::BitReader(std::span<const core::byte> source, core::u32 bitCount, core::u32 bitOffset) noexcept
    : _source{source}, _bitCount{bitsHeld(source.size(), bitCount)},
      _cursor{bitOffset < _bitCount ? bitOffset : _bitCount}
{
}

inline bool BitReader::holds(core::u32 bitCount) const noexcept { return bitCount <= _bitCount - _cursor; }

inline core::u32 BitReader::loadTopBytes(std::span<const core::byte> rest) noexcept
{
    const core::usize count = rest.size() < 4u ? rest.size() : 4u;
    core::u32 word = 0u;

    for (core::usize index = 0u; index < count; ++index)
        word = (word << 8u) | static_cast<core::u32>(rest[index]);
    return count == 0u ? 0u : word << (8u * (4u - count));
}

inline core::u32 BitReader::take(core::u32 bitCount) noexcept
{
    const core::u32 used = _cursor & 7u;
    const core::u32 end = used + bitCount;
    const core::usize byteIndex = _cursor >> 3u;
    const core::byte *const first = _source.data() + byteIndex;
    const core::u32 head = (loadTopBytes(_source.subspan(byteIndex)) << used) >> (32u - bitCount);

    _cursor += bitCount;
    if (end <= 32u)
        return head;
    return head | (static_cast<core::u32>(first[4]) >> (40u - end));
}

inline std::expected<core::u32, BitError> BitReader::readBits(core::u32 bitCount) noexcept
{
    if (bitCount == 0u || bitCount > 32u)
        return std::unexpected(BitError::kInvalidWidth);
    if (!holds(bitCount))
        return std::unexpected(BitError::kUnderflow);
    return take(bitCount);
}

inline std::expected<bool, BitError> BitReader::readBool() noexcept
{
    return readBits(1u).transform([](core::u32 bit) { return bit != 0u; });
}

inline std::expected<core::u8, BitError> BitReader::readU8() noexcept
{
    return readBits(8u).transform([](core::u32 value) { return static_cast<core::u8>(value); });
}

inline std::expected<core::u16, BitError> BitReader::readU16() noexcept
{
    return readBits(16u).transform([](core::u32 value) { return static_cast<core::u16>(value); });
}

inline std::expected<core::u32, BitError> BitReader::readU32() noexcept { return readBits(32u); }

inline std::expected<core::i32, BitError> BitReader::readI32() noexcept
{
    return readBits(32u).transform([](core::u32 value) { return std::bit_cast<core::i32>(value); });
}

inline std::expected<float, BitError> BitReader::readFloat() noexcept
{
    return readBits(32u).transform([](core::u32 value) { return std::bit_cast<float>(value); });
}

inline std::expected<void, BitError> BitReader::readBytes(std::span<core::byte> destination) noexcept
{
    if (destination.size() > (_bitCount - _cursor) / 8u)
        return std::unexpected(BitError::kUnderflow);
    for (core::byte &octet : destination)
        octet = static_cast<core::byte>(take(8u));
    return {};
}

inline std::expected<core::u32, BitError> BitReader::readVarint() noexcept
{
    constexpr core::u32 kLastGroupShift = 28u;
    constexpr core::u32 kBitsPastThirtyTwo = 0x70u;
    const core::u32 start = _cursor;
    core::u32 value = 0u;

    for (core::u32 shift = 0u; shift <= kLastGroupShift; shift += 7u)
    {
        if (!holds(8u))
        {
            _cursor = start;
            return std::unexpected(BitError::kUnderflow);
        }
        const core::u32 octet = take(8u);
        if (shift == kLastGroupShift && (octet & (0x80u | kBitsPastThirtyTwo)) != 0u)
            break;
        value |= (octet & 0x7Fu) << shift;
        if ((octet & 0x80u) == 0u)
            return value;
    }
    _cursor = start;
    return std::unexpected(BitError::kVarintTooLong);
}

inline std::expected<float, BitError> BitReader::readQuantizedFloat(float minimum, float maximum,
                                                                    core::u32 bits) noexcept
{
    return readBits(bits).transform([=](core::u32 step) { return dequantize(step, minimum, maximum, bits); });
}

inline std::expected<float, BitError> BitReader::readAngle(core::u32 bits) noexcept
{
    return readQuantizedFloat(0.0f, kTwoPi, bits);
}

inline core::u32 BitReader::bitsRead() const noexcept { return _cursor; }

inline core::u32 BitReader::bitsRemaining() const noexcept { return _bitCount - _cursor; }

} // namespace lpl::net::protocol

#endif /* LPL_NET_PROTOCOL_BITREADER_INL */
