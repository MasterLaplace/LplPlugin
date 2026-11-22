#ifndef LPL_NET_PROTOCOL_QUANTIZATION_INL
#define LPL_NET_PROTOCOL_QUANTIZATION_INL

namespace lpl::net::protocol {

constexpr core::u32 quantizationCeiling(core::u32 bits) noexcept
{
    return (bits >= 32u) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
}

inline core::u32 quantize(float value, float minimum, float maximum, core::u32 bits) noexcept
{
    const float clamped = !(value >= minimum) ? minimum : (value > maximum ? maximum : value);
    const core::u32 ceiling = quantizationCeiling(bits);
    const float share = (clamped - minimum) / (maximum - minimum);
    const core::u32 step = static_cast<core::u32>(share * static_cast<float>(ceiling) + 0.5f);

    return step > ceiling ? ceiling : step;
}

inline float dequantize(core::u32 step, float minimum, float maximum, core::u32 bits) noexcept
{
    const float share = static_cast<float>(step) / static_cast<float>(quantizationCeiling(bits));

    return minimum + share * (maximum - minimum);
}

inline float wrapAngle(float radians) noexcept
{
    const float wrapped = radians - kTwoPi * static_cast<float>(static_cast<core::i32>(radians / kTwoPi));

    return wrapped < 0.0f ? wrapped + kTwoPi : wrapped;
}

} // namespace lpl::net::protocol

#endif /* LPL_NET_PROTOCOL_QUANTIZATION_INL */
