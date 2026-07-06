/**
 * @file test_ringbuffer_parity.cpp
 * @brief Parity test: SPSC lpl::container::RingBuffer semantics.
 *
 * Exercises the single-producer/single-consumer ring buffer: FIFO ordering,
 * empty/full boundaries (one slot is kept empty, so a Capacity-N buffer holds
 * N-1 elements), index wraparound, drain(), and the non-trivial move path
 * added for owning element types (e.g. std::vector).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-06
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/container/RingBuffer.hpp>
#include <lpl/core/Log.hpp>

#include <utility>
#include <vector>

using namespace lpl;

static int failures = 0;

static void check(const char *label, bool ok)
{
    if (ok)
    {
        std::printf("  PASS: %s\n", label);
    }
    else
    {
        std::printf("  FAIL: %s\n", label);
        ++failures;
    }
}

int main()
{
    core::Log::info("=== RingBuffer Parity Test ===");

    // Basic push/pop and FIFO ordering (trivially-copyable T).
    {
        container::RingBuffer<int, 64> rb;
        check("fresh buffer is empty", rb.isEmpty() && rb.size() == 0);
        check("push succeeds", rb.push(1) && rb.push(2) && rb.push(3));
        check("size reflects pushes", rb.size() == 3);
        check("not empty after push", !rb.isEmpty());

        int v = 0;
        bool order = rb.pop(v) && v == 1;
        order = order && rb.pop(v) && v == 2;
        order = order && rb.pop(v) && v == 3;
        check("FIFO order preserved", order);
        check("empty after draining", rb.isEmpty());
        check("pop on empty returns false", !rb.pop(v));
    }

    // Full boundary: a Capacity-N buffer holds N-1 elements (one empty slot).
    {
        container::RingBuffer<int, 4> rb;
        bool filled = rb.push(10) && rb.push(20) && rb.push(30);
        check("holds Capacity-1 elements", filled && rb.size() == 3);
        check("isFull at Capacity-1", rb.isFull());
        check("push on full returns false", !rb.push(40));
    }

    // Wraparound: repeatedly filling and draining past the capacity must keep
    // FIFO order (indices wrap via the power-of-two mask).
    {
        container::RingBuffer<int, 8> rb;
        bool ok = true;
        int expected = 0;
        for (int round = 0; round < 100 && ok; ++round)
        {
            for (int i = 0; i < 5; ++i)
                ok = ok && rb.push(round * 5 + i);
            int v = 0;
            for (int i = 0; i < 5 && ok; ++i)
                ok = ok && rb.pop(v) && v == expected++;
        }
        check("wraparound keeps FIFO order over 500 items", ok && rb.isEmpty());
    }

    // drain(): dequeue all available into a span.
    {
        container::RingBuffer<int, 16> rb;
        for (int i = 0; i < 10; ++i)
            rb.push(i);
        int out[16] = {};
        const core::usize n = rb.drain(std::span<int>(out, 16));
        bool ok = n == 10 && rb.isEmpty();
        for (int i = 0; i < 10 && ok; ++i)
            ok = out[i] == i;
        check("drain retrieves all elements in order", ok);
    }

    // Non-trivial element type via the move path: the vector's storage is moved
    // into the slot and moved back out, and the source is left empty.
    {
        container::RingBuffer<std::vector<int>, 8> rb;
        std::vector<int> src{1, 2, 3, 4};
        check("push(T&&) succeeds for non-trivial T", rb.push(std::move(src)));
        check("moved-from source is emptied", src.empty());

        std::vector<int> out;
        bool ok = rb.pop(out) && out.size() == 4 && out[0] == 1 && out[3] == 4;
        check("pop moves the payload out intact", ok);
        check("buffer empty after pop", rb.isEmpty());
    }

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
