/**
 * @file Prng.hpp
 * @brief The deterministic generator whose seed replaces a payload.
 *
 * The keystone of the whole codec: a droplet does not carry the list of source
 * packets it combines, it carries a 32-bit seed, and the decoder re-runs this
 * generator to rebuild that list. Storing the generator instead of the result —
 * the same move as a world recipe or a command journal. It must therefore be
 * bit-identical across targets; a libc rand() here would be a silent bug.
 *
 * The stream itself is `lpl::math::Random`, the project's xorshift32, rather than a
 * second generator written here. What this file adds is the POLICY on top of it: how
 * a seed becomes a degree and a set of source indices. Two generators would be two
 * chances to disagree about what a seed means, and a seed that means two things is a
 * payload that decodes two ways.
 *
 * Everything below is integer. The robust soliton distribution is written with a
 * logarithm and a square root, and both come from `lpl::math` in fixed point, never
 * from libm: the distribution decides which droplets exist, so it is authoritative
 * state and a float would put the oracle and ring 0 on different sides of a rounding.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_PRNG_HPP
#    define LPL_LPL_CODEC_PRNG_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Random.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::codec {

/**
 * @struct SolitonParams
 * @brief The two knobs of the robust soliton distribution, plus the block count.
 *
 * @c c and @c delta are Fixed32 rather than float for the reason the whole module is
 * integer: they feed the weights that decide which droplets exist, so they are part
 * of the authoritative state and have to mean the same bits on both targets.
 */
struct SolitonParams {
    core::u32 sourceBlocks{0u};                                  ///< K, the packets the payload was cut into.
    math::Fixed32 c{math::Fixed32::fromRaw(2621)};               ///< ~0.04, the usual tuning constant.
    math::Fixed32 delta{math::Fixed32::fromRaw(3277)};           ///< ~0.05, the tolerated failure probability.
};

/**
 * @class SolitonTable
 * @brief The degree distribution, precomputed as an integer cumulative table.
 *
 * A table rather than a formula evaluated per droplet, and a cumulative one rather
 * than a probability one: sampling then costs a draw and a binary search, both exact
 * in integers. Evaluating the density per draw would need a division per degree and
 * would put the rounding inside the hot loop instead of once, at build time.
 *
 * Weights are held in Q32 inside a 64-bit accumulator. Q16.16 is not enough: the
 * ideal soliton's tail is 1/(d(d-1)), which at d = 256 is already smaller than the
 * Q16.16 quantum, so half the distribution would round to zero and the degrees that
 * make a fountain cover its source would simply never be drawn.
 */
class SolitonTable {
public:
    SolitonTable() noexcept = default;

    /**
     * @brief Builds the distribution for @p params.
     * @param params K and the two tuning constants.
     */
    void build(const SolitonParams &params);

    /**
     * @brief Draws a degree in [1, K].
     * @param stream The droplet's own generator, already seeded.
     * @return The number of source packets this droplet combines.
     */
    [[nodiscard]] core::u32 drawDegree(math::Random &stream) const noexcept;

    /**
     * @brief The spike position, K/R, where the robust component concentrates.
     * @return The degree carrying the tau spike, or 0 when the table is empty.
     */
    [[nodiscard]] core::u32 spikeDegree() const noexcept { return _spikeDegree; }

    /**
     * @brief R = c * ln(K/delta) * sqrt(K), in Q16.16.
     * @return The robust component's scale, as computed without libm.
     */
    [[nodiscard]] math::Fixed32 robustScale() const noexcept { return _robustScale; }

    /**
     * @brief Degrees the table can produce.
     * @return K, or 0 before @ref build.
     */
    [[nodiscard]] core::u32 degrees() const noexcept { return _sourceBlocks; }

    /**
     * @brief FNV-1a over the cumulative weights.
     *
     * The distribution is authoritative state, so it gets folded like any other: a
     * host and a kernel that disagree about one weight disagree about which droplets
     * exist, and that has to fail a gate rather than show up as an occasional
     * undecodable payload.
     *
     * @param seed Fold seed.
     * @return The signature.
     */
    [[nodiscard]] core::u32 fold(core::u32 seed) const noexcept;

private:
    lpl::pmr::vector<core::u64> _cumulative{}; ///< Q32 running total, one entry per degree.
    core::u64 _total{0u};
    core::u32 _sourceBlocks{0u};
    core::u32 _spikeDegree{0u};
    math::Fixed32 _robustScale{};
};

/**
 * @struct DropletPlan
 * @brief What a seed expands to: a degree and the source packets it combines.
 */
struct DropletPlan {
    core::u32 seed{0u};                    ///< The 32-bit word actually stored on the wire.
    core::u32 degree{0u};                  ///< Source packets combined.
    lpl::pmr::vector<core::u32> indices{}; ///< The packets, ascending, distinct.
};

/**
 * @brief Expands @p seed into the droplet it names.
 *
 * The encoder calls this to know what to XOR; the decoder calls it to know what it
 * received. One function, two callers, no list on the wire — which is the third
 * density lever of DNA Fountain and the reason a strand's header is negligible.
 *
 * Indices are returned in ascending order. Not cosmetic: the decoder builds a matrix
 * row from them and the peeling loop removes them one by one, so a stable order is
 * what makes two runs of the same seed produce the same matrix and therefore the
 * same fold.
 *
 * @param seed  The droplet's seed.
 * @param table The distribution, already built.
 * @param out   Receives the plan; its vector is reused rather than reallocated.
 */
void expandDroplet(core::u32 seed, const SolitonTable &table, DropletPlan &out);

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_PRNG_HPP
