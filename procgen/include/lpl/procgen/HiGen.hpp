/**
 * @file HiGen.hpp
 * @brief Running a pipeline at several resolutions at once, in the right order.
 *
 * Evaluating an entire world on one grid is wasteful in both directions at the
 * same time. A forest's macro-layout wants a coarse grid — computing it per blade
 * of grass repeats the same answer thousands of times. Grass wants a fine one —
 * computing it on a continental cell means holding a continent's worth of blades
 * in memory. UE's answer is *hierarchical generation*: one pipeline, several grid
 * sizes, with a strict rule about which way information may flow.
 *
 * The rule is the whole idea, and it is worth stating plainly: **a coarse pass
 * may constrain a fine one, never the reverse.** A forest decides where the trees
 * go; the trees do not decide where the forest is. Coarse results are cached and
 * read by every fine cell inside them, so the coarse work happens once.
 *
 * There is a third case that is not a resolution at all: passes that make no
 * sense per cell. A river crosses the whole map — asking each cell "is there a
 * river here" and stitching the answers gives a different river at every seam.
 * Those run @ref GridLevel::unbounded: once, over everything.
 *
 * What this file adds beyond a comment is that the rule is **mechanical**. A pass
 * declaring a level and reading a finer one is a detected error, not a silently
 * wrong world — which is the only version of a layering rule that survives
 * contact with a growing pipeline.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_HIGEN_HPP
#    define LPL_PROCGEN_HIGEN_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/// Most levels a hierarchy may declare.
inline constexpr core::u32 kMaxGridLevels = 4u;

/// Level index meaning "not bound to a grid at all": runs once, over everything.
inline constexpr core::u32 kUnboundedLevel = 0xFFFFFFFFu;

/**
 * @struct GridLevel
 * @brief One resolution in the hierarchy.
 */
struct GridLevel {
    core::u32 cellSize{1u}; ///< World cells per level cell; larger is coarser.
    bool unbounded{false};  ///< Runs once for the whole domain rather than per cell.
};

/**
 * @struct HiGenSchedule
 * @brief The levels, coarsest first.
 *
 * Coarsest first is not a convention, it is the execution order: a level may only
 * read what a coarser one already produced, so running them in any other order
 * would mean reading a cache that is not filled yet.
 */
struct HiGenSchedule {
    GridLevel levels[kMaxGridLevels]{};
    core::u32 levelCount{0u};

    /**
     * @brief Appends a level, keeping the coarsest-first ordering.
     * @param cellSize World cells per level cell.
     * @return false when the level would break the ordering or the array is full.
     */
    bool addLevel(core::u32 cellSize);

    /// @brief Appends the unbounded level. At most one may exist, and it runs first.
    bool addUnbounded();
};

/**
 * @enum CascadeViolation
 * @brief What is wrong with a pass's declared dependencies.
 */
enum class CascadeViolation : core::u8 {
    None = 0,    ///< The dependency is legal.
    ReadsFiner,  ///< A pass reads a level finer than its own — the forbidden direction.
    UnknownLevel ///< A level index outside the schedule.
};

/**
 * @brief Checks one dependency against the cascade rule.
 *
 * Separate from the schedule so a caller can validate a whole pipeline before
 * running any of it. A world half-generated against a broken dependency graph is
 * worse than one that never started, because it looks finished.
 *
 * @param schedule   The hierarchy.
 * @param passLevel  Level the pass runs at (or @ref kUnboundedLevel).
 * @param inputLevel Level the pass wants to read.
 * @return What is wrong, or @c None.
 */
[[nodiscard]] CascadeViolation checkCascade(const HiGenSchedule &schedule, core::u32 passLevel,
                                            core::u32 inputLevel) noexcept;

/**
 * @brief Which cell of @p level covers a world cell.
 * @param schedule The hierarchy.
 * @param level    Level index.
 * @param worldX   Absolute world column.
 * @param worldZ   Absolute world row.
 * @return The coarse cell's coordinates, as a chunk coordinate.
 */
[[nodiscard]] ChunkCoord levelCellOf(const HiGenSchedule &schedule, core::u32 level, core::i32 worldX,
                                     core::i32 worldZ) noexcept;

/**
 * @class HiGenCache
 * @brief Coarse results, kept so every fine cell reads them instead of redoing them.
 *
 * The saving IS the point of the hierarchy: without a cache, running a coarse
 * pass "per level cell" degenerates into running it once per fine cell, which is
 * the flat pipeline with extra bookkeeping.
 *
 * Keyed by (level, coordinate). Deliberately a small linear structure rather than
 * a hash map: the number of live coarse cells is bounded by the streaming radius,
 * and a linear scan over a few dozen entries beats a map that allocates.
 */
class HiGenCache {
public:
    /// @brief Forgets everything. Called when the world's parameters change.
    void clear() noexcept { _entries.clear(); }

    /// @return Number of cached results.
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_entries.size()); }

    /// @return How many lookups were served from the cache.
    [[nodiscard]] core::u32 hits() const noexcept { return _hits; }

    /// @return How many lookups had to be computed.
    [[nodiscard]] core::u32 misses() const noexcept { return _misses; }

    /**
     * @brief Looks up a coarse result, recording whether it was there.
     * @param level Level index.
     * @param coord Level cell.
     * @param out   Receives the value on a hit.
     * @return true on a hit.
     */
    bool lookup(core::u32 level, ChunkCoord coord, core::u32 &out);

    /**
     * @brief Stores a coarse result.
     * @param level Level index.
     * @param coord Level cell.
     * @param value Result to keep.
     */
    void store(core::u32 level, ChunkCoord coord, core::u32 value);

private:
    struct Entry {
        core::u32 level;
        core::i32 x;
        core::i32 z;
        core::u32 value;
    };
    lpl::pmr::vector<Entry> _entries;
    core::u32 _hits{0u};
    core::u32 _misses{0u};
};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_HIGEN_HPP
