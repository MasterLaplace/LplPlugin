#ifndef LPL_NET_PROTOCOL_QUANTIZATION_INL
#define LPL_NET_PROTOCOL_QUANTIZATION_INL

#include <limits>

namespace lpl::net::protocol {

constexpr core::u32 quantizationCeiling(core::u32 bits) noexcept
{
    return (bits >= 32u) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
}

constexpr bool isQuantizationRange(float minimum, float maximum) noexcept
{
    return maximum > minimum && maximum - minimum <= std::numeric_limits<float>::max();
}

inline core::u32 quantize(float value, float minimum, float maximum, core::u32 bits) noexcept
{
    const float clamped = !(value >= minimum) ? minimum : (value > maximum ? maximum : value);
    const core::u32 ceiling = quantizationCeiling(bits);
    const float share = (clamped - minimum) / (maximum - minimum);
    const float scaled = share * static_cast<float>(ceiling) + 0.5f;

    return scaled < static_cast<float>(ceiling) ? static_cast<core::u32>(scaled) : ceiling;
}

inline float dequantize(core::u32 step, float minimum, float maximum, core::u32 bits) noexcept
{
    const float share = static_cast<float>(step) / static_cast<float>(quantizationCeiling(bits));

    return minimum + share * (maximum - minimum);
}

inline float wrapAngle(float radians) noexcept
{
    constexpr float kCountableTurns = 2147483648.0f;
    const float turns = radians / kTwoPi;

    if (!(turns > -kCountableTurns && turns < kCountableTurns))
        return radians;
    const float wrapped = radians - kTwoPi * static_cast<float>(static_cast<core::i32>(turns));

    return wrapped < 0.0f ? wrapped + kTwoPi : wrapped;
}

} // namespace lpl::net::protocol

#endif /* LPL_NET_PROTOCOL_QUANTIZATION_INL */
