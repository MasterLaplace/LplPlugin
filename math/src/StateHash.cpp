#include <lpl/math/StateHash.hpp>

namespace lpl::math {

StateHash &StateHash::hashBytes(std::span<const core::byte> data)
{
    for (const core::byte b : data)
    {
        _hash ^= static_cast<core::u64>(b);
        _hash *= kPrime;
    }
    return *this;
}

} // namespace lpl::math
