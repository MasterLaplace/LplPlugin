/**
 * @file Streaming.hpp
 * @brief Deciding which chunks should exist right now, and which should not.
 *
 * A world larger than memory is generated around whoever is looking at it. That
 * needs a policy — what to build, what to release, in what order — and the policy
 * is the part worth isolating: it is pure, it is deterministic, and it can be
 * tested without a renderer, a player, or a frame.
 *
 * So this file decides nothing about *how* a chunk is built. It answers one
 * question: given where the sources are and what is already resident, what should
 * happen next?
 *
 * Three decisions shape the answer, and each has a failure it exists to prevent.
 *
 * **The release radius is larger than the generate radius.** Without that gap a
 * source standing on a boundary generates and releases the same chunk every tick,
 * forever. Hysteresis is not a refinement here, it is the difference between a
 * streaming manager and a thrashing one.
 *
 * **Priority is weighted by heading.** What a player is walking toward has to
 * exist before they arrive; what is behind them can wait. Distance alone
 * schedules the world symmetrically around someone who is not moving
 * symmetrically.
 *
 * **The budget is counted in chunks, never in milliseconds.** UE's runtime PCG
 * spends a millisecond budget per frame, and it is right to — its results are not
 * required to be identical on two machines. Ours are: a wall clock in the
 * authoritative path means two clients disagree about what exists, which is the
 * one thing a seed-shipping architecture cannot survive.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_STREAMING_HPP
#    define LPL_PROCGEN_STREAMING_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/**
 * @struct GenerationSource
 * @brief Something the world should exist around.
 *
 * A player, a camera, a server's area of interest. Position is in CHUNK units so
 * the policy never has to know a world's cell size, and the heading is a plain
 * direction — normalised or not, only its sign structure matters.
 */
struct GenerationSource {
    math::Fixed32 x{};        ///< Position along X, in chunks.
    math::Fixed32 z{};        ///< Position along Z, in chunks.
    math::Fixed32 headingX{}; ///< Facing along X; zero means "no preference".
    math::Fixed32 headingZ{}; ///< Facing along Z.
};

/**
 * @struct StreamingParams
 * @brief Radii, weighting and per-tick budget.
 */
struct StreamingParams {
    core::u32 generateRadius{3u}; ///< Chunks around a source that should exist.

    /**
     * @brief Release radius as a multiple of @ref generateRadius, in sixteenths.
     *
     * Expressed as an integer ratio rather than a float because it must be
     * exactly reproducible, and stored in sixteenths so 1.5x is representable.
     * **Must exceed 16**, or the two radii coincide and a source on the boundary
     * thrashes.
     */
    core::u32 releaseRatio16{24u};

    /**
     * @brief How much the heading biases priority, in sixteenths.
     *
     * 0 schedules purely by distance. 16 makes a chunk directly ahead count as
     * one chunk nearer than one directly behind.
     */
    core::u32 directionWeight16{12u};

    core::u32 maxGeneratePerTick{2u}; ///< Chunks scheduled per tick; 0 means no limit.
    core::u32 maxReleasePerTick{4u};  ///< Chunks released per tick; 0 means no limit.
};

/**
 * @struct StreamingRequest
 * @brief One chunk the plan wants built, and how badly.
 */
struct StreamingRequest {
    ChunkCoord coord{};     ///< Which chunk.
    core::u32 priority{0u}; ///< Lower is more urgent.
};

/**
 * @struct StreamingPlan
 * @brief What should happen this tick.
 */
struct StreamingPlan {
    lpl::pmr::vector<StreamingRequest> toGenerate; ///< Most urgent first.
    lpl::pmr::vector<ChunkCoord> toRelease;        ///< Furthest first.
    core::u32 wanted{0u};                          ///< Chunks that should exist, before the budget applied.
    core::u32 resident{0u};                        ///< Chunks that already exist.
};

/**
 * @brief Decides what to build and what to drop.
 *
 * Pure: the same sources and the same resident set always give the same plan, on
 * any machine, in any order. That is what makes a streaming world reproducible
 * rather than merely convincing.
 *
 * @param sources      Where the world should exist.
 * @param sourceCount  Entries in @p sources.
 * @param resident     Chunks that currently exist.
 * @param residentCount Entries in @p resident.
 * @param params       Radii and budget.
 * @return The plan.
 */
[[nodiscard]] StreamingPlan planStreaming(const GenerationSource *sources, core::u32 sourceCount,
                                          const ChunkCoord *resident, core::u32 residentCount,
                                          const StreamingParams &params);

/**
 * @class ChunkPool
 * @brief Recycled chunk slots, so streaming does not allocate in the tick.
 *
 * A manager that allocates a chunk's storage when it appears and frees it when it
 * goes gives the allocator a job every few frames forever — and this project has
 * already measured what that costs, and built a real-time guard that fails a tick
 * for doing it. Slots are taken and returned instead; the peak count is the
 * streaming radius, which is bounded by construction.
 */
class ChunkPool {
public:
    /**
     * @brief Pre-creates @p count slots.
     * @param count Slots to hold.
     */
    void reserve(core::u32 count);

    /**
     * @brief Takes a free slot.
     * @return Slot index, or @c kNoSlot when the pool is exhausted.
     */
    [[nodiscard]] core::u32 acquire();

    /**
     * @brief Returns a slot to the pool.
     * @param slot Slot to release; out-of-range values are ignored.
     */
    void release(core::u32 slot);

    [[nodiscard]] core::u32 capacity() const noexcept { return static_cast<core::u32>(_free.size()) + _liveCount; }
    [[nodiscard]] core::u32 live() const noexcept { return _liveCount; }
    /// @return How many acquisitions were served by a recycled slot.
    [[nodiscard]] core::u32 recycled() const noexcept { return _recycled; }

    /// Returned when no slot is available.
    static constexpr core::u32 kNoSlot = 0xFFFFFFFFu;

private:
    lpl::pmr::vector<core::u32> _free;
    core::u32 _liveCount{0u};
    core::u32 _recycled{0u};
    core::u32 _created{0u};
};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_STREAMING_HPP
