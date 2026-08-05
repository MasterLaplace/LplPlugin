/**
 * @file ReedSolomon.cpp
 * @brief Systematic RS(n, k) over GF(256): encode, syndromes, Berlekamp-Massey,
 *        Chien search and Forney.
 *
 * Every working array is a fixed-size automatic: the decoder runs in ring 0 on a
 * cartridge that may itself be the thing that is broken, so it allocates nothing.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/ReedSolomon.hpp>

#include <lpl/codec/GaloisField.hpp>

namespace lpl::codec {

core::u32 generatorPolynomial(core::u32 parityCount, core::u8 *out) noexcept
{
    if (out == nullptr || parityCount == 0u || parityCount > kMaxParitySymbols)
        return 0u;

    // g(x) = 1, then multiplied by (x - a^i) once per root. Over GF(256) subtraction
    // is addition is XOR, so "x - a^i" and "x + a^i" are the same polynomial — which
    // is why no sign appears anywhere below.
    out[0] = 1u;
    core::u32 degree = 0u;

    for (core::u32 root = 0u; root < parityCount; ++root)
    {
        const core::u8 alpha = gf256Pow(root);
        out[degree + 1u] = 0u;
        // In place, descending, so a coefficient is read before it is overwritten.
        for (core::u32 i = degree + 1u; i > 0u; --i)
            out[i] = gf256Add(out[i - 1u], gf256Mul(out[i], alpha));
        out[0] = gf256Mul(out[0], alpha);
        ++degree;
    }
    return degree + 1u;
}

bool reedSolomonEncode(const core::u8 *data, core::u32 dataCount, core::u32 parityCount, core::u8 *outParity) noexcept
{
    if (data == nullptr || outParity == nullptr || parityCount == 0u || parityCount > kMaxParitySymbols)
        return false;
    if (dataCount + parityCount > kMaxCodewordSymbols)
        return false;

    core::u8 generator[kMaxParitySymbols + 1u]{};
    if (generatorPolynomial(parityCount, generator) == 0u)
        return false;

    // Long division of data * x^s by g(x); the remainder IS the parity. Written as a
    // shift register rather than as an array division because the remainder is all
    // that is wanted and the quotient would be computed only to be thrown away.
    for (core::u32 i = 0u; i < parityCount; ++i)
        outParity[i] = 0u;

    for (core::u32 i = 0u; i < dataCount; ++i)
    {
        const core::u8 feedback = gf256Add(data[i], outParity[parityCount - 1u]);
        for (core::u32 j = parityCount - 1u; j > 0u; --j)
            outParity[j] = gf256Add(outParity[j - 1u], gf256Mul(feedback, generator[j]));
        outParity[0] = gf256Mul(feedback, generator[0]);
    }

    // The register holds the remainder with the highest-order symbol last; the wire
    // wants it in transmission order.
    for (core::u32 i = 0u; i < parityCount / 2u; ++i)
    {
        const core::u8 held = outParity[i];
        outParity[i] = outParity[parityCount - 1u - i];
        outParity[parityCount - 1u - i] = held;
    }
    return true;
}

bool reedSolomonCorrect(core::u8 *codeword, core::u32 symbolCount, core::u32 parityCount,
                        ReedSolomonRepair &outRepair) noexcept
{
    outRepair = ReedSolomonRepair{};

    if (codeword == nullptr || parityCount == 0u || parityCount > kMaxParitySymbols)
        return false;
    if (symbolCount <= parityCount || symbolCount > kMaxCodewordSymbols)
        return false;

    // ── Syndromes ─────────────────────────────────────────────────────────────
    //
    // S_i = C(a^i). A codeword with no errors is a multiple of g(x), whose roots are
    // exactly those points, so every syndrome is zero. Any non-zero one is proof of
    // corruption — and the cheapest proof there is, at one field multiply per symbol.
    core::u8 syndrome[kMaxParitySymbols]{};
    bool clean = true;
    for (core::u32 i = 0u; i < parityCount; ++i)
    {
        core::u8 value = 0u;
        const core::u8 alpha = gf256Pow(i);
        for (core::u32 j = 0u; j < symbolCount; ++j)
            value = gf256Add(gf256Mul(value, alpha), codeword[j]);
        syndrome[i] = value;
        clean = clean && value == 0u;
    }

    if (clean)
    {
        outRepair.clean = true;
        outRepair.corrected = true;
        return true;
    }

    // ── Berlekamp-Massey: the shortest recurrence the syndromes obey ──────────
    //
    // The error locator polynomial's roots are the inverse positions of the errors.
    // Finding it is finding the shortest linear feedback register that generates the
    // syndrome sequence, which is what this loop does — one discrepancy at a time.
    core::u8 locator[kMaxParitySymbols + 1u]{};
    core::u8 previous[kMaxParitySymbols + 1u]{};
    core::u8 scratch[kMaxParitySymbols + 1u]{};
    locator[0] = 1u;
    previous[0] = 1u;
    core::u32 locatorLength = 0u;
    core::u32 shift = 1u;
    core::u8 lastDiscrepancy = 1u;

    for (core::u32 n = 0u; n < parityCount; ++n)
    {
        core::u8 discrepancy = syndrome[n];
        for (core::u32 i = 1u; i <= locatorLength; ++i)
            discrepancy = gf256Add(discrepancy, gf256Mul(locator[i], syndrome[n - i]));

        if (discrepancy == 0u)
        {
            ++shift;
            continue;
        }

        const core::u8 scale = gf256Div(discrepancy, lastDiscrepancy);
        for (core::u32 i = 0u; i <= parityCount; ++i)
            scratch[i] = locator[i];
        for (core::u32 i = 0u; i + shift <= parityCount; ++i)
            locator[i + shift] = gf256Add(locator[i + shift], gf256Mul(scale, previous[i]));

        if (2u * locatorLength <= n)
        {
            locatorLength = n + 1u - locatorLength;
            for (core::u32 i = 0u; i <= parityCount; ++i)
                previous[i] = scratch[i];
            lastDiscrepancy = discrepancy;
            shift = 1u;
        }
        else
        {
            ++shift;
        }
    }

    outRepair.errorDegree = locatorLength;
    if (locatorLength == 0u || locatorLength > parityCount / 2u)
        return false; // more errors than the bound allows; say so rather than guess

    // ── Chien search: which positions the locator points at ───────────────────
    core::u32 position[kMaxParitySymbols]{};
    core::u32 found = 0u;
    for (core::u32 i = 0u; i < symbolCount; ++i)
    {
        // The symbol at index i has position exponent (symbolCount - 1 - i), because
        // the codeword was evaluated highest-degree-first above.
        const core::u32 exponent = symbolCount - 1u - i;
        const core::u8 point = gf256Pow((255u - (exponent % 255u)) % 255u);
        if (gf256Evaluate(locator, locatorLength + 1u, point) != 0u)
            continue;
        if (found >= kMaxParitySymbols)
            return false;
        position[found++] = i;
    }

    if (found != locatorLength)
        return false; // the locator has roots outside the codeword: not our errors

    // ── Forney: how much each of them is wrong by ─────────────────────────────
    //
    // Omega(x) = S(x) * Lambda(x) mod x^s, and the magnitude at a root is
    // Omega(X^-1) / Lambda'(X^-1). The formal derivative over a field of
    // characteristic two keeps only the odd-degree terms, which is why the loop
    // below steps by two — every even term differentiates to zero.
    core::u8 omega[kMaxParitySymbols]{};
    for (core::u32 i = 0u; i < parityCount; ++i)
    {
        core::u8 value = 0u;
        for (core::u32 j = 0u; j <= i && j <= locatorLength; ++j)
            value = gf256Add(value, gf256Mul(locator[j], syndrome[i - j]));
        omega[i] = value;
    }

    for (core::u32 e = 0u; e < found; ++e)
    {
        const core::u32 exponent = symbolCount - 1u - position[e];
        const core::u8 inverse = gf256Pow((255u - (exponent % 255u)) % 255u);

        const core::u8 numerator = gf256Evaluate(omega, parityCount, inverse);

        core::u8 derivative = 0u;
        for (core::u32 i = 1u; i <= locatorLength; i += 2u)
            derivative = gf256Add(derivative, gf256Mul(locator[i], gf256PowOf(inverse, i - 1u)));

        if (derivative == 0u)
            return false;

        // e_j = X_j^(1-b) * Omega(X_j^-1) / Lambda'(X_j^-1), and this code's first
        // consecutive root is b = 0, so the factor is X_j itself.
        //
        // Leaving it out was measured rather than reasoned about: the decoder refused
        // 194 of 200 single-error codewords and fixed 6. Three percent is one position
        // in forty, which is exactly the share where X_j happens to be one — the only
        // place a missing multiplication by X_j is invisible.
        const core::u8 locatorValue = gf256Pow(exponent % 255u);
        const core::u8 magnitude = gf256Mul(locatorValue, gf256Div(numerator, derivative));
        if (magnitude == 0u)
            return false;

        codeword[position[e]] = gf256Add(codeword[position[e]], magnitude);
        ++outRepair.errorCount;
    }

    // Verify rather than trust. A bounded-distance decoder handed more errors than it
    // can correct will happily converge on a WRONG codeword, and the syndromes are the
    // only thing that tells the two apart.
    for (core::u32 i = 0u; i < parityCount; ++i)
    {
        core::u8 value = 0u;
        const core::u8 alpha = gf256Pow(i);
        for (core::u32 j = 0u; j < symbolCount; ++j)
            value = gf256Add(gf256Mul(value, alpha), codeword[j]);
        if (value != 0u)
            return false;
    }

    outRepair.corrected = true;
    return true;
}

bool transversalEncode(const core::u8 *protectedBytesPtr, core::u32 protectedBytes, core::u32 dataShards,
                       core::u32 parityShards, core::u32 rowBytes, core::u8 *outParity) noexcept
{
    if (protectedBytesPtr == nullptr || outParity == nullptr || dataShards == 0u || parityShards == 0u ||
        rowBytes == 0u)
        return false;
    if (dataShards + parityShards > kMaxCodewordSymbols || parityShards > kMaxParitySymbols)
        return false;

    for (core::u32 column = 0u; column < rowBytes; ++column)
    {
        core::u8 symbols[kMaxCodewordSymbols]{};
        for (core::u32 row = 0u; row < dataShards; ++row)
        {
            const core::u64 index = static_cast<core::u64>(row) * rowBytes + column;
            symbols[row] = index < protectedBytes ? protectedBytesPtr[index] : core::u8{0};
        }
        core::u8 columnParity[kMaxParitySymbols]{};
        if (!reedSolomonEncode(symbols, dataShards, parityShards, columnParity))
            return false;
        for (core::u32 p = 0u; p < parityShards; ++p)
            outParity[static_cast<core::u64>(p) * rowBytes + column] = columnParity[p];
    }
    return true;
}

bool transversalRepair(core::u8 *protectedBytesPtr, core::u32 protectedBytes, core::u8 *parity, core::u32 dataShards,
                       core::u32 parityShards, core::u32 rowBytes, TransversalReport &outReport) noexcept
{
    outReport = TransversalReport{};
    if (protectedBytesPtr == nullptr || parity == nullptr || dataShards == 0u || parityShards == 0u || rowBytes == 0u)
        return false;
    if (dataShards + parityShards > kMaxCodewordSymbols || parityShards > kMaxParitySymbols)
        return false;

    outReport.codewords = rowBytes;
    const core::u32 symbolCount = dataShards + parityShards;

    for (core::u32 column = 0u; column < rowBytes; ++column)
    {
        core::u8 codeword[kMaxCodewordSymbols]{};
        for (core::u32 row = 0u; row < dataShards; ++row)
        {
            const core::u64 index = static_cast<core::u64>(row) * rowBytes + column;
            codeword[row] = index < protectedBytes ? protectedBytesPtr[index] : core::u8{0};
        }
        for (core::u32 p = 0u; p < parityShards; ++p)
            codeword[dataShards + p] = parity[static_cast<core::u64>(p) * rowBytes + column];

        ReedSolomonRepair repair{};
        if (!reedSolomonCorrect(codeword, symbolCount, parityShards, repair))
            return false;
        if (repair.clean)
            continue;

        ++outReport.damagedCodewords;
        outReport.correctedBytes += repair.errorCount;

        for (core::u32 row = 0u; row < dataShards; ++row)
        {
            const core::u64 index = static_cast<core::u64>(row) * rowBytes + column;
            if (index < protectedBytes)
                protectedBytesPtr[index] = codeword[row];
        }
        for (core::u32 p = 0u; p < parityShards; ++p)
            parity[static_cast<core::u64>(p) * rowBytes + column] = codeword[dataShards + p];
    }
    return true;
}

} // namespace lpl::codec
