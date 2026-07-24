/*
** LplPlugin — Bitstream quantization / bit-packing test (book §6.3.3)
**
** Proves the bit-packing helpers round-trip within their quantization error and
** that a coarse field genuinely costs fewer bits than a raw float. This is the
** foundation the network-LOD coarse-quant path and the compact-id varints build
** on: positions on 16 bits (~15 mm in a 1000 m world), angles on 10 bits
** (~0.35 deg), ids as LEB128 varints.
*/

#include <lpl/net/protocol/Bitstream.hpp>

#include <cmath>
#include <cstdio>

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

/// Round-trips a quantized float through a fresh read-only stream.
float roundTripQuant(float v, float lo, float hi, core::u32 bits)
{
    net::protocol::Bitstream w;
    w.writeQuantizedFloat(v, lo, hi, bits);
    net::protocol::Bitstream r{w.data(), w.bitsWritten()};
    return r.readQuantizedFloat(lo, hi, bits).value();
}

} // namespace

int main()
{
    std::printf("== bitstream quantization ==\n");

    // ── Quantized position: 16 bits over a 1000 m world ≈ 15 mm resolution ──── //
    {
        const float lo = 0.0f, hi = 1000.0f;
        const core::u32 bits = 16;
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

    // ── Endpoints are exact; out-of-range clamps ───────────────────────────── //
    {
        check(roundTripQuant(0.0f, 0.0f, 1000.0f, 16) == 0.0f, "min endpoint is exact");
        check(std::fabs(roundTripQuant(1000.0f, 0.0f, 1000.0f, 16) - 1000.0f) < 1e-3f, "max endpoint is exact");
        check(roundTripQuant(-50.0f, 0.0f, 1000.0f, 16) == 0.0f, "below-range clamps to min");
        check(std::fabs(roundTripQuant(5000.0f, 0.0f, 1000.0f, 16) - 1000.0f) < 1e-3f, "above-range clamps to max");
    }

    // ── Angle: 10 bits ≈ 0.35 deg, wraps any real angle ────────────────────── //
    {
        const core::u32 bits = 10;
        const float twoPi = 6.28318530717958647692f;
        const float stepDeg = 360.0f / static_cast<float>((1u << bits) - 1u);
        check(stepDeg < 0.36f, "10-bit angle resolution is finer than 0.36 deg");

        float maxErrDeg = 0.0f;
        for (float a = 0.0f; a < twoPi; a += 0.05f)
        {
            net::protocol::Bitstream w;
            w.writeAngle(a, bits);
            net::protocol::Bitstream r{w.data(), w.bitsWritten()};
            const float got = r.readAngle(bits).value();
            maxErrDeg = std::fmax(maxErrDeg, std::fabs(got - a) * 180.0f / twoPi);
        }
        check(maxErrDeg <= stepDeg, "angle round-trips within one step across the circle");

        // A negative / >2*pi angle wraps to the same bucket as its in-circle image.
        net::protocol::Bitstream w1, w2;
        w1.writeAngle(0.5f, bits);
        w2.writeAngle(0.5f + twoPi, bits);
        check(w1.data().size() == w2.data().size(), "wrapped angle encodes to the same width");
        net::protocol::Bitstream r1{w1.data(), w1.bitsWritten()};
        net::protocol::Bitstream r2{w2.data(), w2.bitsWritten()};
        check(std::fabs(r1.readAngle(bits).value() - r2.readAngle(bits).value()) < 1e-4f,
              "an angle and itself + 2*pi decode identically");
    }

    // ── Varint: small ids are one byte, big ids grow gracefully ────────────── //
    {
        const core::u32 samples[] = {0u, 1u, 127u, 128u, 300u, 16383u, 16384u, 0xFFFFFFFFu};
        bool allOk = true;
        for (core::u32 v : samples)
        {
            net::protocol::Bitstream w;
            w.writeVarint(v);
            net::protocol::Bitstream r{w.data(), w.bitsWritten()};
            if (r.readVarint().value() != v)
                allOk = false;
        }
        check(allOk, "varint round-trips 0..UINT32_MAX");

        net::protocol::Bitstream small;
        small.writeVarint(63);
        check(small.data().size() == 1, "an id < 128 costs a single byte (vs 4 raw)");

        net::protocol::Bitstream big;
        big.writeVarint(0xFFFFFFFFu);
        check(big.data().size() == 5, "a full 32-bit id costs 5 varint bytes");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
