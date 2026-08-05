/**
 * @file InferenceBudget.cpp
 * @brief Implementation of where the server profile's spare capacity goes.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/InferenceBudget.hpp>

namespace lpl::engine {

InferenceBudget InferenceBudget::ofTurns(core::u32 turns) noexcept { return InferenceBudget{turns}; }

core::u32 InferenceBudget::concludeAfter() const noexcept
{
    if (_turns == 0u)
        return 0u;
    const core::u32 tail = _turns / 10u;
    return _turns - (tail == 0u ? 1u : tail);
}

} // namespace lpl::engine
