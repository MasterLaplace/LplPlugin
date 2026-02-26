// /////////////////////////////////////////////////////////////////////////////
/// @file PacketQueue.cpp
/// @brief PacketQueue implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/net/PacketQueue.hpp>

namespace lpl::net {

void PacketQueue::push(QueuedPacket packet)
{
    std::lock_guard<std::mutex> lock{mutex_};
    queue_.push(std::move(packet));
}

bool PacketQueue::pop(QueuedPacket& out)
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (queue_.empty())
    {
        return false;
    }
    out = std::move(const_cast<QueuedPacket&>(queue_.top()));
    queue_.pop();
    return true;
}

bool PacketQueue::empty() const noexcept
{
    std::lock_guard<std::mutex> lock{mutex_};
    return queue_.empty();
}

core::u32 PacketQueue::size() const noexcept
{
    std::lock_guard<std::mutex> lock{mutex_};
    return static_cast<core::u32>(queue_.size());
}

void PacketQueue::clear()
{
    std::lock_guard<std::mutex> lock{mutex_};
    while (!queue_.empty())
    {
        queue_.pop();
    }
}

} // namespace lpl::net
