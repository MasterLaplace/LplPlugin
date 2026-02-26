// /////////////////////////////////////////////////////////////////////////////
/// @file Bitstream.cpp
/// @brief Bitstream implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/core/Assert.hpp>

namespace lpl::net::protocol {

Bitstream::Bitstream() noexcept = default;

Bitstream::Bitstream(std::span<const core::byte> data, core::u32 bitCount) noexcept
    : buffer_{data.begin(), data.end()}
    , writeBit_{bitCount}
    , readBit_{0}
    , totalBits_{bitCount}
    , readOnly_{true}
{}

Bitstream::~Bitstream() = default;

// -------------------------------------------------------------------------- //
//  Write                                                                     //
// -------------------------------------------------------------------------- //

void Bitstream::writeBits(core::u32 value, core::u32 bitCount)
{
    LPL_ASSERT(!readOnly_);
    LPL_ASSERT(bitCount > 0 && bitCount <= 32);

    for (core::u32 i = 0; i < bitCount; ++i)
    {
        const core::u32 byteIdx = writeBit_ >> 3;
        const core::u32 bitIdx  = writeBit_ & 7;

        if (byteIdx >= static_cast<core::u32>(buffer_.size()))
        {
            buffer_.push_back(core::byte{0});
        }

        const core::u32 bit = (value >> (bitCount - 1 - i)) & 1u;
        auto& b = buffer_[byteIdx];
        b = static_cast<core::byte>(
            (static_cast<core::u8>(b) & ~(1u << (7 - bitIdx))) |
            (bit << (7 - bitIdx)));

        ++writeBit_;
    }

    totalBits_ = writeBit_;
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

    if (readBit_ + bitCount > totalBits_)
    {
        return core::makeError(core::ErrorCode::OutOfRange, "Bitstream underflow");
    }

    core::u32 result = 0;
    for (core::u32 i = 0; i < bitCount; ++i)
    {
        const core::u32 byteIdx = readBit_ >> 3;
        const core::u32 bitIdx  = readBit_ & 7;
        const core::u32 bit = (static_cast<core::u8>(buffer_[byteIdx]) >> (7 - bitIdx)) & 1u;
        result = (result << 1) | bit;
        ++readBit_;
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

core::u32 Bitstream::bitsWritten() const noexcept { return writeBit_; }

core::u32 Bitstream::bitsRemaining() const noexcept
{
    return (totalBits_ > readBit_) ? totalBits_ - readBit_ : 0;
}

std::span<const core::byte> Bitstream::data() const noexcept { return buffer_; }

void Bitstream::reset() noexcept
{
    readBit_ = 0;
    if (!readOnly_)
    {
        writeBit_ = 0;
        totalBits_ = 0;
        buffer_.clear();
    }
}

} // namespace lpl::net::protocol
