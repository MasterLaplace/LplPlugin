/**
 * @file Herd.inl
 * @brief Out-of-line definitions for ecology::Herd.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_HERD_INL
#    define LPL_ECOLOGY_HERD_INL

namespace lpl::ecology {

inline core::u32 Herd::countSpecies(core::u32 species) const noexcept
{
    core::u32 count = 0u;
    for (core::u32 i = 0u; i < _members.size(); ++i)
        count += _members[i].species == species ? 1u : 0u;
    return count;
}

inline bool Herd::removeOne(core::u32 species) noexcept
{
    for (core::u32 i = 0u; i < _members.size(); ++i)
        if (_members[i].species == species)
        {
            _members[i] = _members[_members.size() - 1u];
            _members.pop_back();
            return true;
        }
    return false;
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_HERD_INL
