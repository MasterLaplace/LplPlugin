/**
 * @file Mosaic.cpp
 * @brief The resident set, and the rule that the finest brick wins.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Mosaic.hpp>

namespace lpl::voxel {

namespace {

/// Mixes a key into a slot. Rotating between the three axes so that a plane of bricks -- which is
/// what a resident set mostly is -- does not collapse onto a handful of slots.
[[nodiscard]] core::u32 hashKey(const BrickKey &key) noexcept
{
    core::u32 h = 2166136261u;
    const core::u32 parts[4]{key.level, static_cast<core::u32>(key.z), static_cast<core::u32>(key.y),
                             static_cast<core::u32>(key.x)};
    for (core::u32 i = 0u; i < 4u; ++i)
    {
        h ^= parts[i];
        h *= 16777619u;
        h ^= h >> 13;
    }
    return h;
}

} // namespace

void BrickMosaic::recomputeCoarsest() noexcept
{
    core::u32 coarsest = 0u;
    core::u32 mask = 0u;
    for (core::u32 i = 0u; i < _count; ++i)
    {
        if (_bricks[i].key.level > coarsest)
            coarsest = _bricks[i].key.level;
        mask |= 1u << _bricks[i].key.level;
    }
    _coarsest = coarsest;
    _levelMask = mask;
}

void BrickMosaic::reindex() noexcept
{
    for (core::u32 i = 0u; i < kSlots; ++i)
        _index[i] = 0u;
    for (core::u32 i = 0u; i < _count; ++i)
    {
        core::u32 slot = hashKey(_bricks[i].key) & (kSlots - 1u);
        // Linear probing. The table is four times the capacity, so a probe run is short.
        for (core::u32 tries = 0u; tries < kSlots; ++tries)
        {
            if (_index[slot] == 0u)
            {
                _index[slot] = static_cast<core::u16>(i + 1u);
                break;
            }
            slot = (slot + 1u) & (kSlots - 1u);
        }
    }
}

const BrickView *BrickMosaic::lookup(const BrickKey &key) const noexcept
{
    core::u32 slot = hashKey(key) & (kSlots - 1u);
    for (core::u32 tries = 0u; tries < kSlots; ++tries)
    {
        const core::u16 entry = _index[slot];
        if (entry == 0u)
            return nullptr;
        const BrickView &b = _bricks[entry - 1u];
        if (b.key == key)
            return &b;
        slot = (slot + 1u) & (kSlots - 1u);
    }
    return nullptr;
}

bool BrickMosaic::insert(const BrickView &brick) noexcept
{
    if (!brick.valid() || _count >= kMaxResidentBricks || brick.key.level >= kMaxPyramidLevels)
        return false;

    // Replacing rather than appending: two bricks with one key would both answer lookups, and
    // which one answered would depend on insertion order -- the very thing find() exists to fix.
    for (core::u32 i = 0u; i < _count; ++i)
    {
        if (_bricks[i].key == brick.key)
        {
            _bricks[i] = brick;
            recomputeCoarsest();
            reindex();
            return true;
        }
    }

    _bricks[_count++] = brick;
    if (brick.key.level > _coarsest)
        _coarsest = brick.key.level;
    _levelMask |= 1u << brick.key.level;
    reindex();
    return true;
}

bool BrickMosaic::remove(const BrickKey &key) noexcept
{
    for (core::u32 i = 0u; i < _count; ++i)
    {
        if (!(_bricks[i].key == key))
            continue;
        _bricks[i] = _bricks[_count - 1u];
        --_count;
        recomputeCoarsest();
        reindex();
        return true;
    }
    return false;
}

const BrickView *BrickMosaic::findCoarserThan(core::u32 level, core::i64 bz, core::i64 by, core::i64 bx) const noexcept
{
    for (core::u32 l = level + 1u; l < kMaxPyramidLevels; ++l)
    {
        if ((_levelMask & (1u << l)) == 0u)
            continue;
        const BrickKey key{l, brickIndexOfBase(bz, l), brickIndexOfBase(by, l), brickIndexOfBase(bx, l)};
        const BrickView *hit = lookup(key);
        if (hit != nullptr)
            return hit;
    }
    return nullptr;
}

bool BrickMosaic::contains(const BrickKey &key) const noexcept
{
    for (core::u32 i = 0u; i < _count; ++i)
    {
        if (_bricks[i].key == key)
            return true;
    }
    return false;
}

const BrickView *BrickMosaic::find(core::i64 bz, core::i64 by, core::i64 bx) const noexcept
{
    // Finest first, and the first hit wins: levels overlap on purpose -- that is what lets a fine
    // brick be evicted without leaving a hole -- so asking in order of level is what makes the
    // answer the best detail available rather than whichever brick arrived first.
    for (core::u32 level = 0u; level < kMaxPyramidLevels; ++level)
    {
        if ((_levelMask & (1u << level)) == 0u)
            continue;
        const BrickKey key{level, brickIndexOfBase(bz, level), brickIndexOfBase(by, level),
                           brickIndexOfBase(bx, level)};
        const BrickView *hit = lookup(key);
        if (hit != nullptr)
            return hit;
    }
    return nullptr;
}

} // namespace lpl::voxel
