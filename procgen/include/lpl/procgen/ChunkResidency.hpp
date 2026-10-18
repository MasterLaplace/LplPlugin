/**
 * @file ChunkResidency.hpp
 * @brief The set of chunks a streamed world keeps loaded, and how it changes.
 *
 * @ref Streaming.hpp decides WHAT should be resident: given sources and a policy
 * it returns a plan — chunks to generate, chunks to release. Nothing owned the
 * other half, so every world that streams wrote it again: hold the records, apply
 * the plan, respect a memory ceiling, find the record covering a world cell, and
 * answer "how high is the ground here" from the resident field rather than from
 * the noise.
 *
 * That last one is not a convenience. A streamed world is DRAWN from the resident
 * field (eroded, carved, whatever its passes did), so anything standing on the
 * ground has to read the same field or it floats — which is exactly the bug that
 * put a herd in mid-air, in a world where the sampler and the renderer disagreed
 * by the amount erosion had moved.
 *
 * The payload is a template parameter because what a game keeps per chunk is the
 * game's business (vegetation, a shadow mask, spawn points, an entity list) while
 * the residency policy is not.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_CHUNK_RESIDENCY_HPP
#    define LPL_PROCGEN_CHUNK_RESIDENCY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Streaming.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/**
 * @brief Floor division, so a chunk index is monotonic across the origin.
 */
[[nodiscard]] inline core::i32 floorDivChunk(core::i32 value, core::i32 divisor) noexcept;

/**
 * @class ChunkResidency
 * @brief Holds the resident chunks of a streamed world and applies streaming plans.
 *
 * @tparam Chunk A record with a @c coord field of type @ref ChunkCoord. Everything
 *               else in it belongs to the caller; this class only moves records in
 *               and out and hands them back.
 */
template <typename Chunk> class ChunkResidency {
public:
    /**
     * @brief Sets the world parameters, the streaming policy and the ceiling.
     *
     * @param maxResident Hard cap on records, whatever the policy's radii ask for.
     *                    Zero means no cap, which on a bounded heap is a decision
     *                    rather than a default.
     */
    void configure(const ChunkParams &chunkParams, const StreamingParams &streamParams, core::u32 maxResident) noexcept;

    [[nodiscard]] const ChunkParams &chunkParams() const noexcept { return _chunkParams; }
    [[nodiscard]] const StreamingParams &streamParams() const noexcept { return _streamParams; }

    void clear() noexcept;

    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_chunks.size()); }
    [[nodiscard]] bool empty() const noexcept { return _chunks.empty(); }
    [[nodiscard]] Chunk &at(core::u32 index) noexcept { return _chunks[index]; }
    [[nodiscard]] const Chunk &at(core::u32 index) const noexcept { return _chunks[index]; }
    [[nodiscard]] core::u32 generatedCount() const noexcept { return _generated; }
    [[nodiscard]] core::u32 releasedCount() const noexcept { return _released; }
    [[nodiscard]] core::u32 maxResident() const noexcept { return _maxResident; }

    /**
     * @brief The record covering a chunk coordinate, or nullptr.
     */
    [[nodiscard]] const Chunk *find(core::i32 chunkX, core::i32 chunkZ) const noexcept;

    /**
     * @brief The record covering a WORLD cell, or nullptr.
     */
    [[nodiscard]] const Chunk *findByCell(core::i32 worldX, core::i32 worldZ) const noexcept;

    /**
     * @brief Local cell of a world cell inside its chunk.
     */
    void localCell(core::i32 worldX, core::i32 worldZ, core::u32 &outX, core::u32 &outZ) const noexcept;

    /**
     * @brief Applies one streaming plan around a focus.
     *
     * A released record is swapped with the last one rather than erased: the order
     * of the resident set has no meaning, and erase would move every record after it.
     *
     * @param focusCellX  Focus in WORLD cells (converted to chunk units here, so a
     *                    caller never has to remember which unit the policy takes).
     * @param headingX    Where the focus is looking, for the direction weighting.
     * @param build       Called for each chunk to generate: (ChunkCoord) -> Chunk.
     * @param onGenerated Called with the freshly inserted record, for whatever the
     *                    game wants to do once per new chunk (a shadow mask, an
     *                    entity spawn).
     */
    template <typename Build, typename OnGenerated>
    void stream(core::f32 focusCellX, core::f32 focusCellZ, core::f32 headingX, core::f32 headingZ, Build &&build,
                OnGenerated &&onGenerated);

    /**
     * @brief Ground height at a world cell — the resident field first.
     *
     * Outside residency the noise still answers, which is what a creature walking
     * towards the horizon needs; inside it, the answer is the field that is actually
     * DRAWN. Two answers to this question is a bug with a picture.
     *
     * @tparam HeightOf Called as @c heightOf(chunk, localX, localZ) -> f32.
     */
    template <typename HeightOf>
    [[nodiscard]] core::f32 groundAt(core::i32 worldX, core::i32 worldZ, HeightOf &&heightOf) const;

    /**
     * @brief The same lookup, in Fixed32, for a caller whose result is AUTHORITATIVE.
     *
     * Not a convenience overload. @ref groundAt hands back a float because that is
     * what shading and projection want, and a float is exactly what a walking body
     * may not derive its position from — the determinism contract says authoritative
     * state is Fixed32 and bit-identical across targets, and a position that came
     * from a rounded height would diverge without anything looking wrong.
     *
     * The underlying data was Fixed32 all along; only the accessor was lossy.
     *
     * @tparam HeightOf Called as @c heightOf(chunk, localX, localZ) -> math::Fixed32.
     */
    template <typename HeightOf>
    [[nodiscard]] math::Fixed32 groundFixedAt(core::i32 worldX, core::i32 worldZ, HeightOf &&heightOf) const;

private:
    ChunkParams _chunkParams{};
    StreamingParams _streamParams{};
    core::u32 _maxResident{0u};
    core::u32 _generated{0u};
    core::u32 _released{0u};
    lpl::pmr::vector<Chunk> _chunks;
    lpl::pmr::vector<ChunkCoord> _scratch;
};

} // namespace lpl::procgen

#    include "ChunkResidency.inl"

#endif // LPL_PROCGEN_CHUNK_RESIDENCY_HPP
