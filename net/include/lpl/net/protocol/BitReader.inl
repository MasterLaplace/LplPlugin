#ifndef LPL_NET_PROTOCOL_BITREADER_INL
#define LPL_NET_PROTOCOL_BITREADER_INL

#include <lpl/net/protocol/Quantization.hpp>

#include <bit>

namespace lpl::net::protocol {

inline BitReader::BitReader(std::span<const core::byte> source, core::u32 bitCount) noexcept
    : BitReader{source, bitCount, 0u}
{
}

inline BitReader::BitReader(std::span<const core::byte> source, core::u32 bitCount, core::u32 bitOffset) noexcept
    : _source{source}, _bitCount{bitCount}, _cursor{bitOffset < bitCount ? bitOffset : bitCount}
{
}

inline bool BitReader::holds(core::u32 bitCount) const noexcept { return bitCount <= _bitCount - _cursor; }

inline core::u32 BitReader::take(core::u32 bitCount) noexcept
{
    core::u32 result = 0u;

    for (core::u32 i = 0u; i < bitCount; ++i)
    {
        const core::u32 byteIndex = _cursor >> 3u;
        const core::u32 bitIndex = _cursor & 7u;
        const core::u32 bit = (static_cast<core::u32>(_source[byteIndex]) >> (7u - bitIndex)) & 1u;

        result = (result << 1u) | bit;
        ++_cursor;
    }
    return result;
}

inline std::expected<core::u32, BitError> BitReader::readBits(core::u32 bitCount) noexcept
{
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
