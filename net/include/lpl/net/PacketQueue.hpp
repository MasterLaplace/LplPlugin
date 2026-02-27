/**
 * @file PacketQueue.hpp
 * @brief Priority-sorted, thread-safe outbound packet queue.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PACKETQUEUE_HPP
    #define LPL_NET_PACKETQUEUE_HPP

#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <mutex>
#include <queue>
#include <vector>

namespace lpl::net {

/**
 * @struct QueuedPacket
 * @brief A packet awaiting transmission, with priority metadata.
 */
struct QueuedPacket
{
    core::u8                    priority;
    protocol::PacketHeader      header;
    std::vector<core::byte>     payload;

    [[nodiscard]] bool operator<(const QueuedPacket& other) const noexcept
    {
        return priority < other.priority;
    }
};

/**
 * @class PacketQueue
 * @brief Thread-safe max-priority queue of outbound packets.
 */
class PacketQueue final : public core::NonCopyable<PacketQueue>
{
public:
    PacketQueue() = default;
    ~PacketQueue() = default;

    /** @brief Pushes a packet into the queue. */
    void push(QueuedPacket packet);

    /**
     * @brief Pops the highest-priority packet.
     * @param[out] out Filled with the packet if available.
     * @return @c true if a packet was dequeued.
     */
    bool pop(QueuedPacket& out);

    /** @brief Returns @c true if the queue is empty. */
    [[nodiscard]] bool empty() const noexcept;

    /** @brief Returns the number of queued packets. */
    [[nodiscard]] core::u32 size() const noexcept;

    /** @brief Discards all queued packets. */
    void clear();

private:
    mutable std::mutex                                   _mutex;
    std::priority_queue<QueuedPacket>                    _queue;
};

} // namespace lpl::net

#endif // LPL_NET_PACKETQUEUE_HPP
