#include <lpl/net/protocol/BitWriter.hpp>
#include <lpl/net/protocol/Bitstream.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <span>

namespace {

int g_failures = 0;

/** Whether operator new counts what it allocates. */
bool g_counting = false;

/** Allocations counted while g_counting was on. */
unsigned long g_allocations = 0;

} // namespace

/**
 * @brief Allocates with malloc, and counts the allocation while counting is on.
 *
 * @details lpl::pmr::vector is std::vector over std::allocator on the host, which allocates through this
 *          operator. The engine builds -fno-exceptions, so a failed new aborts rather than throws.
 */
void *operator new(std::size_t size)
{
    if (g_counting)
        ++g_allocations;
    void *p = std::malloc(size ? size : 1);
    if (p == nullptr)
        std::abort();
    return p;
}

void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

namespace {

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

/**
 * @brief Round-trips a quantized float through a fresh read-only stream.
 */
float roundTripQuant(float v, float lo, float hi, lpl::core::u32 bits)
{
    lpl::net::protocol::Bitstream w;
    w.writeQuantizedFloat(v, lo, hi, bits);
    lpl::net::protocol::Bitstream r{w.data(), w.bitsWritten()};
    return r.readQuantizedFloat(lo, hi, bits).value();
}

/**
 * @brief Quantized position: 16 bits over a 1000 m world ≈ 15 mm resolution.
 */
void checkSixteenBitPositionRoundTripsWithinOneStep()
{
    const float lo = 0.0f, hi = 1000.0f;
    const lpl::core::u32 bits = 16;
    const float step = (hi - lo) / static_cast<float>((1u << bits) - 1u);

    float maxErr = 0.0f;
    for (float v = 0.0f; v <= 1000.0f; v += 3.1f)
    {
        const float got = roundTripQuant(v, lo, hi, bits);
        maxErr = std::fmax(maxErr, std::fabs(got - v));
    }
    std::printf("  (16-bit step = %.4f m, observed max error = %.4f m)\n", step, maxErr);
    check(maxErr <= step, "16-bit position round-trips within one quantization step");
    check(step < 0.02f, "16-bit resolution over 1000 m is finer than 20 mm");
}

/**
 * @brief Endpoints are exact; out-of-range clamps.
 */
void checkEndpointsAreExactAndOutOfRangeClamps()
{
    check(roundTripQuant(0.0f, 0.0f, 1000.0f, 16) == 0.0f, "min endpoint is exact");
    check(std::fabs(roundTripQuant(1000.0f, 0.0f, 1000.0f, 16) - 1000.0f) < 1e-3f, "max endpoint is exact");
    check(roundTripQuant(-50.0f, 0.0f, 1000.0f, 16) == 0.0f, "below-range clamps to min");
    check(std::fabs(roundTripQuant(5000.0f, 0.0f, 1000.0f, 16) - 1000.0f) < 1e-3f, "above-range clamps to max");
}

/**
 * @brief Angle: 10 bits ≈ 0.35 deg, wraps any real angle.
 *
 * @details A negative / >2*pi angle wraps to the same bucket as its in-circle image.
 */
void checkTenBitAngleWrapsAnyRealAngle()
{
    const lpl::core::u32 bits = 10;
    const float twoPi = 6.28318530717958647692f;
    const float stepDeg = 360.0f / static_cast<float>((1u << bits) - 1u);
    check(stepDeg < 0.36f, "10-bit angle resolution is finer than 0.36 deg");

    float maxErrDeg = 0.0f;
    for (float a = 0.0f; a < twoPi; a += 0.05f)
    {
        lpl::net::protocol::Bitstream w;
        w.writeAngle(a, bits);
        lpl::net::protocol::Bitstream r{w.data(), w.bitsWritten()};
        const float got = r.readAngle(bits).value();
        maxErrDeg = std::fmax(maxErrDeg, std::fabs(got - a) * 180.0f / twoPi);
    }
    check(maxErrDeg <= stepDeg, "angle round-trips within one step across the circle");

    lpl::net::protocol::Bitstream w1, w2;
    w1.writeAngle(0.5f, bits);
    w2.writeAngle(0.5f + twoPi, bits);
    check(w1.data().size() == w2.data().size(), "wrapped angle encodes to the same width");
    lpl::net::protocol::Bitstream r1{w1.data(), w1.bitsWritten()};
    lpl::net::protocol::Bitstream r2{w2.data(), w2.bitsWritten()};
    check(std::fabs(r1.readAngle(bits).value() - r2.readAngle(bits).value()) < 1e-4f,
          "an angle and itself + 2*pi decode identically");
}

/**
 * @brief Varint: small ids are one byte, big ids grow gracefully.
 */
void checkVarintSmallIdsAreOneByteBigIdsGrowGracefully()
{
    const lpl::core::u32 samples[] = {0u, 1u, 127u, 128u, 300u, 16383u, 16384u, 0xFFFFFFFFu};
    bool allOk = true;
    for (lpl::core::u32 v : samples)
    {
        lpl::net::protocol::Bitstream w;
        w.writeVarint(v);
        lpl::net::protocol::Bitstream r{w.data(), w.bitsWritten()};
        if (r.readVarint().value() != v)
            allOk = false;
    }
    check(allOk, "varint round-trips 0..UINT32_MAX");

    lpl::net::protocol::Bitstream small;
    small.writeVarint(63);
    check(small.data().size() == 1, "an id < 128 costs a single byte (vs 4 raw)");

    lpl::net::protocol::Bitstream big;
    big.writeVarint(0xFFFFFFFFu);
    check(big.data().size() == 5, "a full 32-bit id costs 5 varint bytes");
}

/**
 * @brief A width read from the wire, as PacketDispatch passes a StateDelta's posBits, is refused.
 */
void checkAWidthOutsideOneToThirtyTwoIsRefused()
{
    const std::array<lpl::core::byte, 8> bytes{};
    lpl::net::protocol::Bitstream r{bytes, 64};

    check(!r.readQuantizedFloat(-1.0f, 1.0f, 0u).has_value(),
          "a quantized position on 0 bits is refused, not read as 0/0");
    check(!r.readBits(33u).has_value(), "33 bits are refused though the stream holds them");
}

/**
 * @brief A bit count larger than the bytes given never takes a read past them.
 */
void checkABitCountLargerThanItsBytesIsCut()
{
    std::array<lpl::core::byte, 8> memory{};
    memory.fill(lpl::core::byte{0x11u});
    lpl::net::protocol::Bitstream r{
        std::span<const lpl::core::byte>{memory.data(), 2u},
         64
    };

    check(r.bitsRemaining() == 16u, "a count of 64 bits over 2 bytes holds 16");
    check(!r.readU32().has_value(), "32 bits are refused rather than read past the copy of 2 bytes");
}

/**
 * @brief Bitstream writes, for the same fields, the bytes BitWriter writes.
 */
void checkBitstreamWritesWhatBitWriterWrites()
{
    lpl::net::protocol::Bitstream stream;
    std::array<lpl::core::byte, 32> wire{};
    lpl::net::protocol::BitWriter writer{wire};

    stream.writeBits(0b101u, 3u);
    writer.writeBits(0b101u, 3u);
    stream.writeU16(0xABCDu);
    writer.writeU16(0xABCDu);
    stream.writeBool(true);
    writer.writeBool(true);
    stream.writeVarint(300u);
    writer.writeVarint(300u);
    stream.writeFloat(-0.0f);
    writer.writeFloat(-0.0f);
    stream.writeQuantizedFloat(500.0f, 0.0f, 1000.0f, 16u);
    writer.writeQuantizedFloat(500.0f, 0.0f, 1000.0f, 16u);
    stream.writeAngle(-0.5f, 10u);
    writer.writeAngle(-0.5f, 10u);
    stream.writeI32(-2);
    writer.writeI32(-2);

    const auto streamBytes = stream.data();
    const auto writerBytes = writer.bytes();
    bool same = stream.bitsWritten() == writer.bitsWritten() && streamBytes.size() == writerBytes.size();
    for (std::size_t i = 0; same && i < streamBytes.size(); ++i)
        same = streamBytes[i] == writerBytes[i];
    check(same, "Bitstream and BitWriter write the same bytes for the same fields");
}

/**
 * @brief A count of bytes the stream does not hold is refused before anything is allocated for it.
 *
 * @details The count comes from the wire, as StateSnapshot reads a blob's size: a forged one must cost
 *          nothing. Nine bytes asked of eight show it without asking the machine for a large block, and
 *          the eight read whole show that the counter sees the vector Bitstream allocates.
 */
void checkABytesCountPastTheStreamAllocatesNothing()
{
    const std::array<lpl::core::byte, 8> eight{};
    lpl::net::protocol::Bitstream refused{eight, 64};
    lpl::net::protocol::Bitstream accepted{eight, 64};

    g_allocations = 0;
    g_counting = true;
    const bool nineRefused = !refused.readBytes(9u).has_value();
    const unsigned long allocationsRefused = g_allocations;
    g_allocations = 0;
    const bool eightRead = accepted.readBytes(8u).has_value();
    const unsigned long allocationsAccepted = g_allocations;
    g_counting = false;

    check(nineRefused, "nine bytes are refused from a stream of eight");
    check(allocationsRefused == 0, "the refused count allocates nothing");
    check(eightRead && allocationsAccepted == 1, "the eight bytes the stream holds are read into one allocation");
}

} // namespace

/**
 * @brief Bitstream quantization / bit-packing test (book §6.3.3).
 *
 * @details Proves the bit-packing helpers round-trip within their quantization error and that a coarse
 *          field genuinely costs fewer bits than a raw float. This is the foundation the network-LOD
 *          coarse-quant path and the compact-id varints build on: positions on 16 bits (~15 mm in a
 *          1000 m world), angles on 10 bits (~0.35 deg), ids as LEB128 varints.
 */
int main()
{
    std::printf("== bitstream quantization ==\n");

    checkSixteenBitPositionRoundTripsWithinOneStep();
    checkEndpointsAreExactAndOutOfRangeClamps();
    checkTenBitAngleWrapsAnyRealAngle();
    checkVarintSmallIdsAreOneByteBigIdsGrowGracefully();
    checkAWidthOutsideOneToThirtyTwoIsRefused();
    checkABitCountLargerThanItsBytesIsCut();
    checkBitstreamWritesWhatBitWriterWrites();
    checkABytesCountPastTheStreamAllocatesNothing();

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
