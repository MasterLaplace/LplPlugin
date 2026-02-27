/**
 * @file StateSnapshot.cpp
 * @brief StateSnapshot implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/serial/StateSnapshot.hpp>
#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/core/Assert.hpp>

namespace lpl::serial {

StateSnapshot::StateSnapshot() = default;
StateSnapshot::~StateSnapshot() = default;

core::u64 StateSnapshot::tick() const noexcept { return _tick; }
void StateSnapshot::setTick(core::u64 tick) noexcept { _tick = tick; }

core::u64 StateSnapshot::hash() const noexcept { return _hash; }

void StateSnapshot::addEntityBlob(core::u32 entityId,
                                  const core::byte* data,
                                  core::usize size)
{
    LPL_ASSERT(data != nullptr || size == 0);

    EntityBlob blob;
    blob.entityId = entityId;
    blob.data.assign(data, data + size);
    _blobs.push_back(std::move(blob));
}

core::usize StateSnapshot::entityCount() const noexcept
{
    return _blobs.size();
}

const EntityBlob& StateSnapshot::blob(core::usize index) const
{
    LPL_ASSERT(index < _blobs.size());
    return _blobs[index];
}

void StateSnapshot::clear() noexcept
{
    _blobs.clear();
    _hash = 0;
    _tick = 0;
}

void StateSnapshot::rehash()
{
    math::StateHash hasher;
    hasher.combine(_tick);
    for (const auto& b : _blobs)
    {
        hasher.combine(b.entityId);
        if (!b.data.empty())
        {
            hasher.hashBytes({reinterpret_cast<const core::byte*>(b.data.data()),
                              b.data.size()});
        }
    }
    _hash = hasher.digest();
}

core::Expected<void> StateSnapshot::serialize(
    net::protocol::Bitstream& stream) const
{
    // Write 64-bit _tick
    stream.writeU32(static_cast<core::u32>(_tick >> 32));
    stream.writeU32(static_cast<core::u32>(_tick & 0xFFFFFFFF));

    // Write 64-bit _hash
    stream.writeU32(static_cast<core::u32>(_hash >> 32));
    stream.writeU32(static_cast<core::u32>(_hash & 0xFFFFFFFF));

    stream.writeU32(static_cast<core::u32>(_blobs.size()));

    for (const auto& blob : _blobs)
    {
        stream.writeU32(blob.entityId);
        stream.writeU32(static_cast<core::u32>(blob.data.size()));
        if (!blob.data.empty())
        {
            stream.writeBytes(blob.data);
        }
    }

    return {};
}

core::Expected<void> StateSnapshot::deserialize(
    net::protocol::Bitstream& stream)
{
    clear();

    auto tickHi = stream.readU32();
    if (!tickHi) return core::makeError(core::ErrorCode::IoError, "EOF reading tick");
    auto tickLo = stream.readU32();
    if (!tickLo) return core::makeError(core::ErrorCode::IoError, "EOF reading tick");
    _tick = (static_cast<core::u64>(*tickHi) << 32) | *tickLo;

    auto hashHi = stream.readU32();
    if (!hashHi) return core::makeError(core::ErrorCode::IoError, "EOF reading hash");
    auto hashLo = stream.readU32();
    if (!hashLo) return core::makeError(core::ErrorCode::IoError, "EOF reading hash");
    _hash = (static_cast<core::u64>(*hashHi) << 32) | *hashLo;

    auto count = stream.readU32();
    if (!count) return core::makeError(core::ErrorCode::IoError, "EOF reading count");

    _blobs.reserve(*count);

    for (core::u32 i = 0; i < *count; ++i)
    {
        EntityBlob blob;

        auto id = stream.readU32();
        if (!id) return core::makeError(core::ErrorCode::IoError, "EOF reading id");
        blob.entityId = *id;

        auto size = stream.readU32();
        if (!size) return core::makeError(core::ErrorCode::IoError, "EOF reading size");

        if (*size > 0)
        {
            auto dataBytes = stream.readBytes(*size);
            if (!dataBytes) return core::makeError(core::ErrorCode::IoError, "EOF reading data");
            blob.data = std::move(*dataBytes);
        }

        _blobs.push_back(std::move(blob));
    }

    return {};
}

} // namespace lpl::serial
