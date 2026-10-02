/**
 * @file Mosaic.hpp
 * @brief The bricks that are in memory right now, and how a point finds the best one.
 *
 * @warning **A lookup returns the FINEST resident brick covering the point, not the first one.**
 * Levels overlap by design -- that is what lets a fine brick be evicted without leaving a hole,
 * because the coarse brick behind it still answers. Without an explicit level comparison the brick
 * that answers is whichever the loader happened to insert first, so what the scroll looks like
 * would depend on network timing. This project has already paid that exact bug once, on relief
 * tiles, and the fix is the same.
 *
 * @warning **A slot owns nothing and outlives nothing.** The bytes are a mapping; if the mapping is
 * released while its brick is still listed, every ray reads unmapped memory, and nothing about a
 * dangling brick looks wrong until the picture is already false. Whoever releases a mapping removes
 * its brick first. Tests must assert the **counts**, never a value read back through the mosaic:
 * freed memory lies convincingly, and a probe that reads through it passes by luck.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_MOSAIC_HPP
#    define LPL_VOXEL_MOSAIC_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/voxel/Brick.hpp>

namespace lpl::voxel {

/// Bricks a mosaic can hold. At 2 MiB each this caps the resident set at 1 GiB.
inline constexpr core::u32 kMaxResidentBricks = 512u;

/**
 * @class BrickMosaic
 * @brief A bounded, non-owning set of resident bricks.
 */
class BrickMosaic final {
public:
    /// @brief Forgets every brick. Does not touch the bytes; the owner of the mappings does that.
    void clear() noexcept
    {
        _count = 0u;
        _coarsest = 0u;
        _levelMask = 0u;
        for (core::u32 i = 0u; i < kSlots; ++i)
            _index[i] = 0u;
    }

    /**
     * @brief Lists a brick. Ignored if the set is full or the brick is invalid.
     * @return Whether it was listed.
     */
    bool insert(const BrickView &brick) noexcept;

    /**
     * @brief Unlists the brick with @p key, if present.
     * @return Whether one was removed.
     */
    bool remove(const BrickKey &key) noexcept;

    /**
     * @brief Finest resident brick containing the level-0 sample (@p bz, @p by, @p bx).
     * @return Pointer into the set, or nullptr when no resident brick covers the point.
     */
    [[nodiscard]] const BrickView *find(core::i64 bz, core::i64 by, core::i64 bx) const noexcept;

    /**
     * @brief Finest resident brick strictly coarser than @p level covering the point.
     *
     * The other half of the overlap: @ref find answers with the best detail available, and this
     * answers with what is behind it. A level of detail that switches without one is a visible
     * seam -- the boundary of the fine ring is a cube, and a cube-shaped brightness step in the
     * middle of a scan reads as structure that is not there.
     */
    [[nodiscard]] const BrickView *findCoarserThan(core::u32 level, core::i64 bz, core::i64 by,
                                                   core::i64 bx) const noexcept;

    /// @brief Whether a brick with exactly this key is listed.
    [[nodiscard]] bool contains(const BrickKey &key) const noexcept;

    [[nodiscard]] core::u32 count() const noexcept { return _count; }

    /**
     * @brief Coarsest level currently listed.
     *
     * A ray that finds nothing at a point has to advance somehow, and advancing by the finest
     * brick span in a mostly-empty subject costs thousands of wasted lookups. The coarsest
     * resident level is the largest step that cannot skip over a brick the mosaic actually holds.
     */
    [[nodiscard]] core::u32 coarsestLevel() const noexcept { return _coarsest; }

    /// Bit per level that has at least one resident brick. A lookup skips the empty ones.
    [[nodiscard]] core::u32 levelMask() const noexcept { return _levelMask; }

    /**
     * @brief Finest level with any resident brick.
     *
     * A brick at this level cannot be shadowed by a finer one, which is what makes it safe to
     * reuse as a hint without asking the index again.
     */
    [[nodiscard]] core::u32 finestLevel() const noexcept
    {
        for (core::u32 level = 0u; level < kMaxPyramidLevels; ++level)
        {
            if ((_levelMask & (1u << level)) != 0u)
                return level;
        }
        return 0u;
    }
    [[nodiscard]] const BrickView &at(core::u32 i) const noexcept { return _bricks[i]; }

private:
    /**
     * Open-addressed index from key to slot.
     *
     * @warning **A linear scan is what made this the hot spot.** Every sample asks which brick
     * covers a point, and the gradient asks six more; over five hundred resident bricks that is
     * hundreds of comparisons per sample. The index turns a lookup into at most one probe per
     * RESIDENT LEVEL -- at most eight, usually three or four -- which is a different order of
     * magnitude for the same answer.
     *
     * @warning It is rebuilt whenever the set changes, never patched incrementally: a stale entry
     * would return a brick that is no longer there, and the bytes behind it are freed memory that
     * reads back convincingly.
     */
    static constexpr core::u32 kSlots = 2048u; ///< Power of two, comfortably over kMaxResidentBricks.

    void recomputeCoarsest() noexcept;
    void reindex() noexcept;
    [[nodiscard]] const BrickView *lookup(const BrickKey &key) const noexcept;

    BrickView _bricks[kMaxResidentBricks]{};
    core::u16 _index[kSlots]{}; ///< Slot + 1, so zero means empty.
    core::u32 _count{0u};
    core::u32 _coarsest{0u};
    core::u32 _levelMask{0u};
};

} // namespace lpl::voxel

#endif // LPL_VOXEL_MOSAIC_HPP
