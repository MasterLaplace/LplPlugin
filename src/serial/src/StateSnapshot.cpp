// /////////////////////////////////////////////////////////////////////////////
/// @file StateSnapshot.cpp
/// @brief StateSnapshot implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/serial/StateSnapshot.hpp>
#include <lpl/core/Assert.hpp>

namespace lpl::serial {

StateSnapshot::StateSnapshot() = default;
StateSnapshot::~StateSnapshot() = default;

core::u64 StateSnapshot::tick() const noexcept { return tick_; }
void StateSnapshot::setTick(core::u64 tick) noexcept { tick_ = tick; }

core::u64 StateSnapshot::hash() const noexcept { return hash_; }

void StateSnapshot::addEntityBlob(core::u32 entityId,
                                  const core::byte* data,
                                  core::usize size)
{
    LPL_ASSERT(data != nullptr || size == 0);

    EntityBlob blob;
    blob.entityId = entityId;
    blob.data.assign(data, data + size);
    blobs_.push_back(std::move(blob));
}

core::usize StateSnapshot::entityCount() const noexcept
{
    return blobs_.size();
}

const EntityBlob& StateSnapshot::blob(core::usize index) const
{
    LPL_ASSERT(index < blobs_.size());
    return blobs_[index];
}

void StateSnapshot::clear() noexcept
{
    blobs_.clear();
    hash_ = 0;
    tick_ = 0;
}

void StateSnapshot::rehash()
{
    math::StateHash hasher;
    hasher.combine(tick_);
    for (const auto& b : blobs_)
    {
        hasher.combine(b.entityId);
        if (!b.data.empty())
        {
            hasher.hashBytes({reinterpret_cast<const core::byte*>(b.data.data()),
                              b.data.size()});
        }
    }
    hash_ = hasher.digest();
}

core::Expected<void> StateSnapshot::serialize(
    net::protocol::Bitstream& /*stream*/) const
{
    LPL_ASSERT(false && "StateSnapshot::serialize not yet implemented");
    return {};
}

core::Expected<void> StateSnapshot::deserialize(
    net::protocol::Bitstream& /*stream*/)
{
    LPL_ASSERT(false && "StateSnapshot::deserialize not yet implemented");
    return {};
}

} // namespace lpl::serial
