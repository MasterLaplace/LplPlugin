/**
 * @file DeltaCompression.cpp
 * @brief XOR delta compression implementation with zero-run-length encoding.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/net/protocol/DeltaCompression.hpp>
#include <lpl/core/Assert.hpp>

namespace lpl::net::protocol {

core::Expected<std::vector<core::byte>> DeltaCompression::encode(
    std::span<const core::byte> baseline,
    std::span<const core::byte> current)
{
    if (baseline.size() != current.size())
    {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Baseline and current must be the same size");
    }

    std::vector<core::byte> delta;
    delta.reserve(current.size());

    core::u32 i = 0;
    const auto n = static_cast<core::u32>(current.size());

    while (i < n)
    {
        const auto xorByte = static_cast<core::byte>(
            static_cast<core::u8>(current[i]) ^ static_cast<core::u8>(baseline[i]));

        if (xorByte == core::byte{0})
        {
            core::u32 runLen = 0;
            while (i < n && runLen < 255 &&
                   (static_cast<core::u8>(current[i]) ^ static_cast<core::u8>(baseline[i])) == 0)
            {
                ++runLen;
                ++i;
            }
            delta.push_back(core::byte{0});
            delta.push_back(static_cast<core::byte>(runLen));
        }
        else
        {
            delta.push_back(xorByte);
            ++i;
        }
    }

    return delta;
}

core::Expected<std::vector<core::byte>> DeltaCompression::decode(
    std::span<const core::byte> baseline,
    std::span<const core::byte> delta)
{
    std::vector<core::byte> result;
    result.reserve(baseline.size());

    core::u32 baseIdx = 0;
    core::u32 deltaIdx = 0;
    const auto deltaSize = static_cast<core::u32>(delta.size());
    const auto baseSize  = static_cast<core::u32>(baseline.size());

    while (deltaIdx < deltaSize && baseIdx < baseSize)
    {
        if (delta[deltaIdx] == core::byte{0})
        {
            ++deltaIdx;
            if (deltaIdx >= deltaSize)
            {
                return core::makeError(core::ErrorCode::CorruptedData,
                                       "Truncated zero-run in delta");
            }

            const auto runLen = static_cast<core::u32>(static_cast<core::u8>(delta[deltaIdx]));
            ++deltaIdx;

            for (core::u32 r = 0; r < runLen && baseIdx < baseSize; ++r, ++baseIdx)
            {
                result.push_back(baseline[baseIdx]);
            }
        }
        else
        {
            result.push_back(static_cast<core::byte>(
                static_cast<core::u8>(baseline[baseIdx]) ^
                static_cast<core::u8>(delta[deltaIdx])));
            ++baseIdx;
            ++deltaIdx;
        }
    }

    while (baseIdx < baseSize)
    {
        result.push_back(baseline[baseIdx++]);
    }

    return result;
}

} // namespace lpl::net::protocol
