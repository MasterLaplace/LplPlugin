/**
 * @file TestRingBuffer.cpp
 * @brief Unit tests for bci::dsp::RingBuffer.
 */

#include <catch2/catch_test_macros.hpp>

#include "lpl/bci/dsp/RingBuffer.hpp"

namespace lpl::bci {

using namespace bci::dsp;

TEST_CASE("RingBuffer basic push and pop", "[dsp][ringbuffer]")
{
    RingBuffer<int, 64> buffer;

    REQUIRE(buffer.empty());
    REQUIRE(buffer.capacity() == 64);

    REQUIRE(buffer.push(42));
    REQUIRE_FALSE(buffer.empty());
    REQUIRE(buffer.size() == 1);

    int val = 0;
    REQUIRE(buffer.pop(val));
    REQUIRE(val == 42);
    REQUIRE(buffer.empty());
}

TEST_CASE("RingBuffer tryPop returns nullopt when empty", "[dsp][ringbuffer]")
{
    RingBuffer<float, 16> buffer;
    float val = 0.0f;
    REQUIRE_FALSE(buffer.tryPop(val));
}

TEST_CASE("RingBuffer drain retrieves all elements", "[dsp][ringbuffer]")
{
    RingBuffer<int, 128> buffer;

    for (int i = 0; i < 10; ++i)
        buffer.push(i);

    std::vector<int> drained;
    auto count = buffer.drain([&](int val) { drained.push_back(val); });
    
    REQUIRE(count == 10);
    REQUIRE(drained.size() == 10);
    REQUIRE(buffer.empty());

    for (int i = 0; i < 10; ++i)
        REQUIRE(drained[static_cast<std::size_t>(i)] == i);
}

TEST_CASE("RingBuffer reports full when capacity reached", "[dsp][ringbuffer]")
{
    RingBuffer<int, 4> buffer;

    REQUIRE(buffer.push(1));
    REQUIRE(buffer.push(2));
    REQUIRE(buffer.push(3));
    REQUIRE(buffer.push(4));
    REQUIRE_FALSE(buffer.push(5));
}



} // namespace lpl::bci
