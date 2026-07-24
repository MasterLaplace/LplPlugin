/**
 * @file Bitstream.cpp
 * @brief Bitstream implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/net/protocol/Bitstream.hpp>

namespace lpl::net::protocol {

Bitstream::Bitstream() noexcept = default;

Bitstream::Bitstream(std::span<const core::byte> data, core::u32 bitCount) noexcept
    : _buffer{data.begin(), data.end()}, _writeBit{bitCount}, _readBit{0}, _totalBits{bitCount}, _readOnly{true}
{
}

Bitstream::~Bitstream() = default;

// -------------------------------------------------------------------------- //
//  Write                                                                     //
// -------------------------------------------------------------------------- //

void Bitstream::writeBits(core::u32 value, core::u32 bitCount)
{
    LPL_ASSERT(!_readOnly);
    LPL_ASSERT(bitCount > 0 && bitCount <= 32);

    for (core::u32 i = 0; i < bitCount; ++i)
    {
        const core::u32 byteIdx = _writeBit >> 3;
        const core::u32 bitIdx = _writeBit & 7;

        if (byteIdx >= static_cast<core::u32>(_buffer.size()))
        {
            _buffer.push_back(core::byte{0});
        }

        const core::u32 bit = (value >> (bitCount - 1 - i)) & 1u;
        auto &b = _buffer[byteIdx];
        b = static_cast<core::byte>((static_cast<core::u8>(b) & ~(1u << (7 - bitIdx))) | (bit << (7 - bitIdx)));

        ++_writeBit;
    }

    _totalBits = _writeBit;
}

void Bitstream::writeBool(bool value) { writeBits(value ? 1u : 0u, 1); }
void Bitstream::writeU8(core::u8 value) { writeBits(value, 8); }
void Bitstream::writeU16(core::u16 value) { writeBits(value, 16); }
void Bitstream::writeU32(core::u32 value) { writeBits(value, 32); }

void Bitstream::writeI32(core::i32 value)
{
    core::u32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32(bits);
}

void Bitstream::writeFloat(float value)
{
    core::u32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32(bits);
}

void Bitstream::writeBytes(std::span<const core::byte> bytes)
{
    for (auto b : bytes)
    {
        writeU8(static_cast<core::u8>(b));
    }
}

// -------------------------------------------------------------------------- //
//  Bit-packing / quantization (book §6.3.3)                                  //
// -------------------------------------------------------------------------- //

namespace {

/// Largest integer representable in @p bits (the quantization ceiling).
[[nodiscard]] core::u32 quantMax(core::u32 bits) noexcept
{
    // bits is 1..32; (1u << 32) is UB, so special-case the full width.
    return (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
}

constexpr float kTwoPi = 6.28318530717958647692f;

} // namespace

void Bitstream::writeQuantizedFloat(float value, float min, float max, core::u32 bits)
{
    LPL_ASSERT(bits > 0 && bits <= 32);
    LPL_ASSERT(max > min);

    const float clamped = (value < min) ? min : (value > max ? max : value);
    const float range = max - min;
    const core::u32 maxQ = quantMax(bits);

    // Round-to-nearest, then guard the top edge against float error pushing it
    // one past the ceiling.
    const float t = (clamped - min) / range;
    core::u32 q = static_cast<core::u32>(t * static_cast<float>(maxQ) + 0.5f);
    if (q > maxQ)
        q = maxQ;

    writeBits(q, bits);
}

core::Expected<float> Bitstream::readQuantizedFloat(float min, float max, core::u32 bits)
{
    auto r = readBits(bits);
    if (!r.has_value())
        return core::makeError(r.error().code(), r.error().message());

    const core::u32 maxQ = quantMax(bits);
    const float t = static_cast<float>(r.value()) / static_cast<float>(maxQ);
    return min + t * (max - min);
}

void Bitstream::writeAngle(float radians, core::u32 bits)
{
    // Wrap into [0, 2*pi) so any real angle encodes.
    float wrapped = radians - kTwoPi * static_cast<float>(static_cast<core::i32>(radians / kTwoPi));
    if (wrapped < 0.0f)
        wrapped += kTwoPi;
    writeQuantizedFloat(wrapped, 0.0f, kTwoPi, bits);
}

core::Expected<float> Bitstream::readAngle(core::u32 bits) { return readQuantizedFloat(0.0f, kTwoPi, bits); }

void Bitstream::writeVarint(core::u32 value)
{
    // LEB128: 7 payload bits per byte, high bit = "more bytes follow".
    do
    {
        core::u8 septet = static_cast<core::u8>(value & 0x7Fu);
        value >>= 7;
        if (value != 0)
            septet |= 0x80u;
        writeU8(septet);
    } while (value != 0);
}

core::Expected<core::u32> Bitstream::readVarint()
{
    core::u32 result = 0;
    core::u32 shift = 0;
    while (shift < 32)
    {
        auto b = readU8();
        if (!b.has_value())
            return core::makeError(b.error().code(), b.error().message());
        result |= static_cast<core::u32>(b.value() & 0x7Fu) << shift;
        if ((b.value() & 0x80u) == 0)
            return result;
        shift += 7;
    }
    return core::makeError(core::ErrorCode::CorruptedData, "Varint too long");
}

// -------------------------------------------------------------------------- //
//  Read                                                                      //
// -------------------------------------------------------------------------- //

core::Expected<core::u32> Bitstream::readBits(core::u32 bitCount)
{
    LPL_ASSERT(bitCount > 0 && bitCount <= 32);

    if (_readBit + bitCount > _totalBits)
    {
        return core::makeError(core::ErrorCode::OutOfRange, "Bitstream underflow");
    }

    core::u32 result = 0;
    for (core::u32 i = 0; i < bitCount; ++i)
    {
        const core::u32 byteIdx = _readBit >> 3;
        const core::u32 bitIdx = _readBit & 7;
        const core::u32 bit = (static_cast<core::u8>(_buffer[byteIdx]) >> (7 - bitIdx)) & 1u;
        result = (result << 1) | bit;
        ++_readBit;
    }

    return result;
}

core::Expected<bool> Bitstream::readBool()
{
    auto r = readBits(1);
    if (!r.has_value())
        return core::makeError(r.error().code(), r.error().message());
    return r.value() != 0;
}

core::Expected<core::u8> Bitstream::readU8()
{
    auto r = readBits(8);
    if (!r.has_value())
        return core::makeError(r.error().code(), r.error().message());
    return static_cast<core::u8>(r.value());
}

core::Expected<core::u16> Bitstream::readU16()
{
    auto r = readBits(16);
    if (!r.has_value())
        return core::makeError(r.error().code(), r.error().message());
    return static_cast<core::u16>(r.value());
}

core::Expected<core::u32> Bitstream::readU32() { return readBits(32); }

core::Expected<core::i32> Bitstream::readI32()
{
    auto r = readU32();
    if (!r.has_value())
        return core::makeError(r.error().code(), r.error().message());
    core::i32 val;
    core::u32 bits = r.value();
    std::memcpy(&val, &bits, sizeof(val));
    return val;
}

core::Expected<float> Bitstream::readFloat()
{
    auto r = readU32();
    if (!r.has_value())
        return core::makeError(r.error().code(), r.error().message());
    float val;
    core::u32 bits = r.value();
    std::memcpy(&val, &bits, sizeof(val));
    return val;
}

core::Expected<std::vector<core::byte>> Bitstream::readBytes(core::u32 count)
{
    std::vector<core::byte> out;
    out.reserve(count);
    for (core::u32 i = 0; i < count; ++i)
    {
        auto r = readU8();
        if (!r.has_value())
            return core::makeError(r.error().code(), r.error().message());
        out.push_back(static_cast<core::byte>(r.value()));
    }
    return out;
}

// -------------------------------------------------------------------------- //
//  Query                                                                     //
// -------------------------------------------------------------------------- //

core::u32 Bitstream::bitsWritten() const noexcept { return _writeBit; }

core::u32 Bitstream::bitsRemaining() const noexcept { return (_totalBits > _readBit) ? _totalBits - _readBit : 0; }

std::span<const core::byte> Bitstream::data() const noexcept { return _buffer; }

void Bitstream::reset() noexcept
{
    _readBit = 0;
    if (!_readOnly)
    {
        _writeBit = 0;
        _totalBits = 0;
        _buffer.clear();
    }
}

} // namespace lpl::net::protocol
