/**
 * @file PacketQueue.cpp
 * @brief PacketQueue implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/PacketQueue.hpp>

namespace lpl::net {

void PacketQueue::push(QueuedPacket packet)
{
    std::lock_guard<std::mutex> lock{_mutex};
    _queue.push(std::move(packet));
}

bool PacketQueue::pop(QueuedPacket& out)
{
    std::lock_guard<std::mutex> lock{_mutex};
    if (_queue.empty())
    {
        return false;
    }
    out = std::move(const_cast<QueuedPacket&>(_queue.top()));
    _queue.pop();
    return true;
}

bool PacketQueue::empty() const noexcept
{
    std::lock_guard<std::mutex> lock{_mutex};
    return _queue.empty();
}

core::u32 PacketQueue::size() const noexcept
{
    std::lock_guard<std::mutex> lock{_mutex};
    return static_cast<core::u32>(_queue.size());
}

void PacketQueue::clear()
{
    std::lock_guard<std::mutex> lock{_mutex};
    while (!_queue.empty())
    {
        _queue.pop();
    }
}

} // namespace lpl::net
