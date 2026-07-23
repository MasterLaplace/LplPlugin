/**
 * @file StateHash.cpp
 * @brief FNV-1a incremental hash for deterministic desync detection.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-22
 * @copyright MIT License
 */

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
