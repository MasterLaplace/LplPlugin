/**
 * @file Bitstream.cpp
 * @brief Bitstream implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/core/Assert.hpp>

namespace lpl::net::protocol {

Bitstream::Bitstream() noexcept = default;

Bitstream::Bitstream(std::span<const core::byte> data, core::u32 bitCount) noexcept
    : _buffer{data.begin(), data.end()}
    , _writeBit{bitCount}
    , _readBit{0}
    , _totalBits{bitCount}
    , _readOnly{true}
{}

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
        const core::u32 bitIdx  = _writeBit & 7;

        if (byteIdx >= static_cast<core::u32>(_buffer.size()))
        {
            _buffer.push_back(core::byte{0});
        }

        const core::u32 bit = (value >> (bitCount - 1 - i)) & 1u;
        auto& b = _buffer[byteIdx];
        b = static_cast<core::byte>(
            (static_cast<core::u8>(b) & ~(1u << (7 - bitIdx))) |
            (bit << (7 - bitIdx)));

        ++_writeBit;
    }

    _totalBits = _writeBit;
}

void Bitstream::writeBool(bool value)   { writeBits(value ? 1u : 0u, 1); }
void Bitstream::writeU8(core::u8 value) { writeBits(value, 8); }
void Bitstream::writeU16(core::u16 value) { writeBits(value, 16); }
void Bitstream::writeU32(core::u32 value) { writeBits(value, 32); }

void Bitstream::writeBytes(std::span<const core::byte> bytes)
{
    for (auto b : bytes)
    {
        writeU8(static_cast<core::u8>(b));
    }
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
        const core::u32 bitIdx  = _readBit & 7;
        const core::u32 bit = (static_cast<core::u8>(_buffer[byteIdx]) >> (7 - bitIdx)) & 1u;
        result = (result << 1) | bit;
        ++_readBit;
    }

    return result;
}

core::Expected<bool> Bitstream::readBool()
{
    auto r = readBits(1);
    if (!r.has_value()) return core::makeError(r.error().code(), r.error().message());
    return r.value() != 0;
}

core::Expected<core::u8> Bitstream::readU8()
{
    auto r = readBits(8);
    if (!r.has_value()) return core::makeError(r.error().code(), r.error().message());
    return static_cast<core::u8>(r.value());
}

core::Expected<core::u16> Bitstream::readU16()
{
    auto r = readBits(16);
    if (!r.has_value()) return core::makeError(r.error().code(), r.error().message());
    return static_cast<core::u16>(r.value());
}

core::Expected<core::u32> Bitstream::readU32()
{
    return readBits(32);
}

core::Expected<std::vector<core::byte>> Bitstream::readBytes(core::u32 count)
{
    std::vector<core::byte> out;
    out.reserve(count);
    for (core::u32 i = 0; i < count; ++i)
    {
        auto r = readU8();
        if (!r.has_value()) return core::makeError(r.error().code(), r.error().message());
        out.push_back(static_cast<core::byte>(r.value()));
    }
    return out;
}

// -------------------------------------------------------------------------- //
//  Query                                                                     //
// -------------------------------------------------------------------------- //

core::u32 Bitstream::bitsWritten() const noexcept { return _writeBit; }

core::u32 Bitstream::bitsRemaining() const noexcept
{
    return (_totalBits > _readBit) ? _totalBits - _readBit : 0;
}

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
