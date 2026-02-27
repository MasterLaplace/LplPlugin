/**
 * @file Archetype.inl
 * @brief Inline implementations for Archetype.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_ARCHETYPE_INL
    #define LPL_ECS_ARCHETYPE_INL

namespace lpl::ecs {

inline Archetype::Archetype(std::span<const ComponentId> ids) noexcept
{
    for (auto id : ids)
    {
        _mask.set(static_cast<core::usize>(id));
    }
}

inline void Archetype::add(ComponentId id) noexcept
{
    _mask.set(static_cast<core::usize>(id));
}

inline void Archetype::remove(ComponentId id) noexcept
{
    _mask.reset(static_cast<core::usize>(id));
}

inline bool Archetype::has(ComponentId id) const noexcept
{
    return _mask.test(static_cast<core::usize>(id));
}

inline bool Archetype::contains(const Archetype& other) const noexcept
{
    return (_mask & other._mask) == other._mask;
}

inline const Archetype::Mask& Archetype::mask() const noexcept
{
    return _mask;
}

inline core::usize Archetype::count() const noexcept
{
    return _mask.count();
}

inline bool Archetype::operator==(const Archetype& other) const noexcept
{
    return _mask == other._mask;
}

} // namespace lpl::ecs

#endif // LPL_ECS_ARCHETYPE_INL
