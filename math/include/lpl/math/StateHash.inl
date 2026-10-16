#ifndef LPL_MATH_STATE_HASH_INL
#define LPL_MATH_STATE_HASH_INL

namespace lpl::math {

template <core::Blittable T> StateHash &StateHash::combine(const T &value)
{
    const auto *ptr = reinterpret_cast<const core::byte *>(&value);
    return hashBytes({ptr, sizeof(T)});
}

} // namespace lpl::math

#endif // LPL_MATH_STATE_HASH_INL
