#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/net/protocol/BitReader.hpp>
#include <lpl/net/protocol/BitWriter.hpp>
#include <lpl/net/protocol/Bitstream.hpp>

namespace lpl::net::protocol {

namespace {

[[nodiscard]] core::usize bytesFor(core::u64 bits) noexcept { return static_cast<core::usize>((bits + 7u) / 8u); }

[[nodiscard]] core::u32 bitsHeld(std::span<const core::byte> data, core::u32 bitCount) noexcept
{
    return BitReader{data, bitCount}.bitsRemaining();
}

[[nodiscard]] core::Unexpected refusal(BitError error)
{
    switch (error)
    {
    case BitError::kUnderflow: return core::makeError(core::ErrorCode::OutOfRange, "Bitstream underflow");
    case BitError::kVarintTooLong: return core::makeError(core::ErrorCode::CorruptedData, "Varint too long");
    case BitError::kInvalidWidth:
        return core::makeError(core::ErrorCode::kInvalidArgument, "Bitstream width outside 1..32");
    case BitError::kInvalidRange: return core::makeError(core::ErrorCode::kInvalidArgument, "Bitstream range invalid");
    case BitError::kOverflow: return core::makeError(core::ErrorCode::kBufferOverflow, "Bitstream overflow");
    }
    return core::makeError(core::ErrorCode::kInternalError, "Bitstream refusal unnamed");
}

} // namespace

Bitstream::Bitstream() noexcept = default;

Bitstream::Bitstream(std::span<const core::byte> data, core::u32 bitCount) noexcept
    : _buffer{data.begin(), data.end()}, _writeBit{bitsHeld(data, bitCount)}, _readBit{0},
      _totalBits{bitsHeld(data, bitCount)}, _readOnly{true}
{
}

Bitstream::~Bitstream() = default;

template <typename Write> void Bitstream::writeThroughCore(core::u64 maximumBits, Write write)
{
    LPL_ASSERT(!_readOnly);

    if (_failure.has_value())
        return;
    _buffer.resize(bytesFor(_writeBit + maximumBits));
    BitWriter writer{_buffer, _writeBit};
    write(writer);
    _writeBit = writer.bitsWritten();
    _totalBits = _writeBit;
    _buffer.resize(bytesFor(_writeBit));
    if (!writer.failure().has_value())
        return;
    _failure = writer.failure();
    core::Log::error("Bitstream", "a field was refused, and every field written after it is refused too");
}

template <typename Value, typename Read> core::Expected<Value> Bitstream::readThroughCore(Read read)
{
    BitReader reader{_buffer, _totalBits, _readBit};
    const std::expected<Value, BitError> value = read(reader);

    if (!value.has_value())
        return refusal(value.error());
    _readBit = reader.bitsRead();
    return *value;
}

void Bitstream::writeBits(core::u32 value, core::u32 bitCount)
{
    writeThroughCore(bitCount, [=](BitWriter &writer) { writer.writeBits(value, bitCount); });
}

void Bitstream::writeBool(bool value) { writeBits(value ? 1u : 0u, 1); }
void Bitstream::writeU8(core::u8 value) { writeBits(value, 8); }
void Bitstream::writeU16(core::u16 value) { writeBits(value, 16); }
void Bitstream::writeU32(core::u32 value) { writeBits(value, 32); }

void Bitstream::writeI32(core::i32 value)
{
    writeThroughCore(32u, [=](BitWriter &writer) { writer.writeI32(value); });
}

void Bitstream::writeFloat(float value)
{
    writeThroughCore(32u, [=](BitWriter &writer) { writer.writeFloat(value); });
}

void Bitstream::writeBytes(std::span<const core::byte> bytes)
{
    writeThroughCore(static_cast<core::u64>(bytes.size()) * 8u, [=](BitWriter &writer) { writer.writeBytes(bytes); });
}

void Bitstream::writeQuantizedFloat(float value, float min, float max, core::u32 bits)
{
    writeThroughCore(bits, [=](BitWriter &writer) { writer.writeQuantizedFloat(value, min, max, bits); });
}

core::Expected<float> Bitstream::readQuantizedFloat(float min, float max, core::u32 bits)
{
    return readThroughCore<float>([=](BitReader &reader) { return reader.readQuantizedFloat(min, max, bits); });
}

void Bitstream::writeAngle(float radians, core::u32 bits)
{
    writeThroughCore(bits, [=](BitWriter &writer) { writer.writeAngle(radians, bits); });
}

core::Expected<float> Bitstream::readAngle(core::u32 bits)
{
    return readThroughCore<float>([=](BitReader &reader) { return reader.readAngle(bits); });
}

void Bitstream::writeVarint(core::u32 value)
{
    writeThroughCore(40u, [=](BitWriter &writer) { writer.writeVarint(value); });
}

core::Expected<core::u32> Bitstream::readVarint()
{
    return readThroughCore<core::u32>([](BitReader &reader) { return reader.readVarint(); });
}

core::Expected<core::u32> Bitstream::readBits(core::u32 bitCount)
{
    return readThroughCore<core::u32>([=](BitReader &reader) { return reader.readBits(bitCount); });
}

core::Expected<bool> Bitstream::readBool()
{
    return readThroughCore<bool>([](BitReader &reader) { return reader.readBool(); });
}

core::Expected<core::u8> Bitstream::readU8()
{
    return readThroughCore<core::u8>([](BitReader &reader) { return reader.readU8(); });
}

core::Expected<core::u16> Bitstream::readU16()
{
    return readThroughCore<core::u16>([](BitReader &reader) { return reader.readU16(); });
}

core::Expected<core::u32> Bitstream::readU32() { return readBits(32); }

core::Expected<core::i32> Bitstream::readI32()
{
    return readThroughCore<core::i32>([](BitReader &reader) { return reader.readI32(); });
}

core::Expected<float> Bitstream::readFloat()
{
    return readThroughCore<float>([](BitReader &reader) { return reader.readFloat(); });
}

core::Expected<lpl::pmr::vector<core::byte>> Bitstream::readBytes(core::u32 count)
{
    BitReader reader{_buffer, _totalBits, _readBit};

    if (count > reader.bitsRemaining() / 8u)
        return refusal(BitError::kUnderflow);
    lpl::pmr::vector<core::byte> out(count);
    if (const auto read = reader.readBytes(out); !read.has_value())
        return refusal(read.error());
    _readBit = reader.bitsRead();
    return out;
}

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
        _failure.reset();
    }
}

std::optional<BitError> Bitstream::failure() const noexcept { return _failure; }

} // namespace lpl::net::protocol
