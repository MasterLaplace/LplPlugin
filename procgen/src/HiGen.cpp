/**
 * @file HiGen.cpp
 * @brief Implementation of the hierarchical schedule and its cache.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/HiGen.hpp>

namespace lpl::procgen {

bool HiGenSchedule::addLevel(core::u32 cellSize)
{
    if (levelCount >= kMaxGridLevels || cellSize == 0u)
        return false;

    // Coarsest first, strictly. Equal sizes are refused too: two levels of the
    // same resolution are one level, and allowing them would make "which level is
    // coarser" ambiguous — which is the exact question the cascade rule asks.
    if (levelCount > 0u && !levels[levelCount - 1u].unbounded && cellSize >= levels[levelCount - 1u].cellSize)
        return false;

    levels[levelCount].cellSize = cellSize;
    levels[levelCount].unbounded = false;
    ++levelCount;
    return true;
}

bool HiGenSchedule::addUnbounded()
{
    // Only as the first entry: an unbounded pass runs over the whole domain, so
    // by definition nothing is coarser than it.
    if (levelCount != 0u)
        return false;
    levels[0].cellSize = 0u;
    levels[0].unbounded = true;
    levelCount = 1u;
    return true;
}

CascadeViolation checkCascade(const HiGenSchedule &schedule, core::u32 passLevel, core::u32 inputLevel) noexcept
{
    // The unbounded level sees the whole domain, so it may read anything and
    // anything may read it. It is the one level with no resolution to compare.
    if (passLevel == kUnboundedLevel || inputLevel == kUnboundedLevel)
    {
        if (passLevel != kUnboundedLevel && passLevel >= schedule.levelCount)
            return CascadeViolation::UnknownLevel;
        if (inputLevel != kUnboundedLevel && inputLevel >= schedule.levelCount)
            return CascadeViolation::UnknownLevel;
        return CascadeViolation::None;
    }

    if (passLevel >= schedule.levelCount || inputLevel >= schedule.levelCount)
        return CascadeViolation::UnknownLevel;

    // Levels are stored coarsest-first, so a LOWER index is coarser. A pass may
    // read its own level and anything coarser; reading a higher index means
    // reading finer, which is the forbidden direction.
    return inputLevel <= passLevel ? CascadeViolation::None : CascadeViolation::ReadsFiner;
}

ChunkCoord levelCellOf(const HiGenSchedule &schedule, core::u32 level, core::i32 worldX, core::i32 worldZ) noexcept
{
    if (level >= schedule.levelCount || schedule.levels[level].unbounded)
        return ChunkCoord{0, 0};

    const core::i32 size = static_cast<core::i32>(schedule.levels[level].cellSize);
    if (size <= 0)
        return ChunkCoord{0, 0};

    // Floor division, not truncation: -1 / 8 is 0 in C++ and -1 here, so the
    // cells left of the origin would all collapse onto cell 0 and a world would
    // be mirrored about its own axis.
    const core::i32 x = worldX >= 0 ? worldX / size : -(((-worldX) + size - 1) / size);
    const core::i32 z = worldZ >= 0 ? worldZ / size : -(((-worldZ) + size - 1) / size);
    return ChunkCoord{x, z};
}

bool HiGenCache::lookup(core::u32 level, ChunkCoord coord, core::u32 &out)
{
    for (core::u32 i = 0u; i < _entries.size(); ++i)
    {
        const Entry &entry = _entries[i];
        if (entry.level == level && entry.x == coord.x && entry.z == coord.z)
        {
            out = entry.value;
            ++_hits;
            return true;
        }
    }
    ++_misses;
    return false;
}

void HiGenCache::store(core::u32 level, ChunkCoord coord, core::u32 value)
{
    for (core::u32 i = 0u; i < _entries.size(); ++i)
    {
        Entry &entry = _entries[i];
        if (entry.level == level && entry.x == coord.x && entry.z == coord.z)
        {
            entry.value = value;
            return;
        }
    }
    _entries.push_back(Entry{level, coord.x, coord.z, value});
}

} // namespace lpl::procgen
