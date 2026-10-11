/**
 * @file test_ringbuffer_parity.cpp
 * @brief Two-thread stress test of lpl::container::RingBuffer.
 *
 * A producer thread and a consumer thread move two million elements through a ring of 64 slots,
 * small enough that both ends keep running out of their cached index and reading the other's.
 * Every push and every pop call is used in turn, so each path meets each other path across the
 * threads. The consumer checks that each element arrives once and in order; for an element that
 * owns heap memory, the constructions and destructions on both threads balance at the end. The
 * single-threaded semantics are tested in tests/container/ring_buffer.cpp, on the host and in
 * ring 0; this one needs threads, so it runs on the host only, and under ThreadSanitizer it checks
 * the memory orders.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-06
 * @copyright MIT License
 */

#include <lpl/container/RingBuffer.hpp>
#include <lpl/core/Log.hpp>

#include <atomic>
#include <cstdio>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr lpl::core::u32 kElementCount = 1u << 21u;
constexpr lpl::core::usize kSlots = 64u;
constexpr lpl::core::u32 kProducerChunk = 37u;
constexpr lpl::core::u32 kConsumerChunk = 29u;
constexpr lpl::core::usize kConsumeAtMost = 41u;

int failures = 0;

void check(const char *label, bool ok)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok)
        ++failures;
}

/** @brief An element that owns heap memory, and counts its lives on whichever thread they end. */
class Owned final {
public:
    explicit Owned(lpl::core::u32 sequence) : _payload{sequence, ~sequence}
    {
        lives.fetch_add(1, std::memory_order_relaxed);
    }

    Owned(const Owned &other) : _payload(other._payload) { lives.fetch_add(1, std::memory_order_relaxed); }

    Owned(Owned &&other) noexcept : _payload(std::move(other._payload))
    {
        lives.fetch_add(1, std::memory_order_relaxed);
    }

    Owned &operator=(const Owned &other) = default;
    Owned &operator=(Owned &&other) noexcept = default;

    ~Owned() { lives.fetch_sub(1, std::memory_order_relaxed); }

    [[nodiscard]] bool carries(lpl::core::u32 sequence) const noexcept
    {
        return _payload.size() == 2u && _payload[0] == sequence && _payload[1] == ~sequence;
    }

    static inline std::atomic<lpl::core::i64> lives{0}; /**< Constructions minus destructions, on every thread. */

private:
    std::vector<lpl::core::u32> _payload;
};

template <typename Element> [[nodiscard]] Element make(lpl::core::u32 sequence)
{
    if constexpr (std::is_same_v<Element, Owned>)
        return Owned(sequence);
    else
        return sequence;
}

template <typename Element> [[nodiscard]] bool carries(const Element &element, lpl::core::u32 sequence) noexcept
{
    if constexpr (std::is_same_v<Element, Owned>)
        return element.carries(sequence);
    else
        return element == sequence;
}

template <typename Element>
void produce(lpl::container::RingBuffer<Element, kSlots> &ring, const std::atomic<bool> &consumerGone)
{
    std::vector<Element> chunk;
    lpl::core::u32 next = 0u;
    lpl::core::u32 call = 0u;

    chunk.reserve(kProducerChunk);
    while (next < kElementCount && !consumerGone.load(std::memory_order_acquire))
    {
        bool pushed = false;
        switch (call++ % 4u)
        {
        case 0u: {
            const Element element = make<Element>(next);
            pushed = ring.push(element);
            break;
        }
        case 1u: pushed = ring.push(make<Element>(next)); break;
        case 2u: pushed = ring.emplace(make<Element>(next)); break;
        default:
            chunk.clear();
            for (lpl::core::u32 i = 0u; i < kProducerChunk && next + i < kElementCount; ++i)
                chunk.push_back(make<Element>(next + i));
            next += static_cast<lpl::core::u32>(ring.pushBulk(std::span<const Element>(chunk)));
            break;
        }
        if (pushed)
            ++next;
    }
}

template <typename Element> [[nodiscard]] bool consume(lpl::container::RingBuffer<Element, kSlots> &ring)
{
    std::vector<Element> chunk;
    lpl::core::u32 expected = 0u;
    lpl::core::u32 call = 0u;
    bool inOrder = true;

    while (expected < kElementCount && inOrder)
    {
        switch (call++ % 4u)
        {
        case 0u: {
            Element element = make<Element>(0u);
            if (ring.pop(element))
                inOrder = carries(element, expected++);
            break;
        }
        case 1u:
            if (const Element *oldest = ring.front())
            {
                inOrder = carries(*oldest, expected++);
                ring.popFront();
            }
            break;
        case 2u: {
            chunk.assign(kConsumerChunk, make<Element>(0u));
            const lpl::core::usize count = ring.drain(std::span<Element>(chunk));
            for (lpl::core::usize i = 0u; i < count && inOrder; ++i)
                inOrder = carries(chunk[i], expected++);
            break;
        }
        default:
            ring.consume([&](Element &element) noexcept { inOrder = carries(element, expected++) && inOrder; },
                         kConsumeAtMost);
            break;
        }
    }
    return inOrder && expected == kElementCount;
}

template <typename Element> [[nodiscard]] bool transfer(lpl::core::u32 &rejected)
{
    lpl::container::RingBuffer<Element, kSlots> ring;
    bool consumedInOrder = false;
    std::atomic<bool> consumerGone{false};

    std::thread consumer([&] {
        consumedInOrder = consume(ring);
        consumerGone.store(true, std::memory_order_release);
    });
    produce(ring, consumerGone);
    consumer.join();
    rejected = ring.rejectedCount();
    return consumedInOrder && ring.isEmpty();
}

} // namespace

int main()
{
    lpl::core::Log::info("=== RingBuffer two-thread stress test ===");

    lpl::core::u32 rejected = 0u;
    check("two million u32 cross a 64-slot ring once each and in order", transfer<lpl::core::u32>(rejected));
    check("the producer met a full ring and read the consumer's index again", rejected > 0u);

    check("two million heap-owning elements cross it the same way", transfer<Owned>(rejected));
    check("every element built on either thread is destroyed", Owned::lives.load() == 0);

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
