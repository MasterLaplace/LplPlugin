/**
 * @file Fountain.hpp
 * @brief LT codes: a rateless stream of droplets.
 *
 * A fountain never runs dry, which is what makes constraint filtering free: a
 * droplet that violates a biological or wire constraint is discarded at zero cost
 * in bits, because the next one is already available. A fixed-rate code would have
 * to pay redundancy for the same guarantee. The robust soliton distribution is what
 * makes decoding terminate.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_FOUNTAIN_HPP
#    define LPL_LPL_CODEC_FOUNTAIN_HPP

#    include <lpl/codec/Prng.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::codec {

/**
 * @struct SourceView
 * @brief The payload, seen as K fixed-size blocks.
 *
 * A view rather than a copy: the encoder reads it and never owns it, so a caller can
 * hand it a cartridge image, a mapped file or a network buffer without the module
 * needing to know which.
 */
struct SourceView {
    const core::u8 *bytes{nullptr}; ///< First byte of block 0.
    core::u32 blockBytes{0u};       ///< Bytes per block; every block is this size.
    core::u32 blockCount{0u};       ///< K.
};

/**
 * @struct Droplet
 * @brief One emitted combination: a seed, and the XOR it names.
 *
 * The seed IS the metadata. It expands to the degree and the list of source blocks,
 * so a droplet's header is four bytes whatever its degree — the third density lever
 * of DNA Fountain, and the reason a strand spends its bases on payload.
 */
struct Droplet {
    core::u32 seed{0u};
    lpl::pmr::vector<core::u8> payload{};
};

/**
 * @struct BiologicalLimits
 * @brief What a strand has to satisfy to be synthesisable.
 *
 * Two failure modes, both chemical rather than informational: a long run of one base
 * makes the polymerase slip, and a GC fraction far from half makes the strand melt at
 * the wrong temperature. Neither is a property the code can repair, which is exactly
 * why they are checked BEFORE the droplet is kept rather than corrected after.
 *
 * Permille rather than a fraction: this decides which droplets exist, so it is
 * authoritative and a float would put two targets on different sides of a comparison.
 */
struct BiologicalLimits {
    core::u32 maxHomopolymer{3u};  ///< Longest run of one base tolerated.
    core::u32 minGcPermille{450u}; ///< Lowest GC share, in thousandths.
    core::u32 maxGcPermille{550u}; ///< Highest GC share, in thousandths.
};

/**
 * @brief Would @p bytes synthesise?
 *
 * The two-bit code is A=0, C=1, G=2, T=3, most significant pair first. The mapping is
 * arbitrary in the chemistry and is NOT arbitrary here: it decides which droplets are
 * rejected, so it is part of the format the way the field modulus is.
 *
 * @param bytes  Droplet payload.
 * @param count  Bytes available.
 * @param limits What to enforce.
 * @return true when the strand is within every limit.
 */
[[nodiscard]] bool satisfiesBiologicalLimits(const core::u8 *bytes, core::u32 count,
                                             const BiologicalLimits &limits) noexcept;

/**
 * @class Fountain
 * @brief Emits droplets from a source, for as long as it is asked.
 */
class Fountain {
public:
    Fountain() noexcept = default;

    /**
     * @brief Binds a source and builds its degree distribution.
     * @param source  The payload, already cut into equal blocks.
     * @param tuning  Soliton constants; @c sourceBlocks is taken from @p source.
     */
    Fountain(const SourceView &source, const SolitonParams &tuning);

    /**
     * @brief Emits the droplet named by @p seed.
     * @param seed The droplet's identity.
     * @param out  Receives it; its buffer is reused rather than reallocated.
     */
    void emit(core::u32 seed, Droplet &out) const;

    /**
     * @brief Emits @p count droplets, skipping any the limits reject.
     *
     * This is SIM-091, the move the whole scheme is built around: rejecting a droplet
     * costs nothing, because the fountain has already produced the next one. A
     * fixed-rate code would have to reserve parity in advance to survive the same
     * rejections, which is fifteen to thirty percent of the bases spent on a problem
     * that can instead be sidestepped.
     *
     * @param count       Valid droplets wanted.
     * @param limits      Constraint every kept droplet satisfies.
     * @param firstSeed   Seed to start from; seeds increase by one.
     * @param out         Receives the droplets, in emission order.
     * @return Droplets examined, valid and rejected together. The ratio is the
     *         measurement the DNA work needs and is not otherwise recoverable.
     */
    core::u32 emitValid(core::u32 count, const BiologicalLimits &limits, core::u32 firstSeed,
                        lpl::pmr::vector<Droplet> &out) const;

    /**
     * @brief The distribution this fountain draws from.
     * @return The table, for a decoder that has to expand the same seeds.
     */
    [[nodiscard]] const SolitonTable &table() const noexcept { return _table; }

    /**
     * @brief Blocks the source was cut into.
     * @return K.
     */
    [[nodiscard]] core::u32 blockCount() const noexcept { return _source.blockCount; }

    /**
     * @brief Bytes in one block, and therefore in one droplet payload.
     * @return The block size.
     */
    [[nodiscard]] core::u32 blockBytes() const noexcept { return _source.blockBytes; }

private:
    SourceView _source{};
    SolitonTable _table{};
};

/**
 * @struct DecayParams
 * @brief A millennium of molecular decay, as three rates.
 *
 * Per-thousand rather than per-unit for the reason everything else here is integer:
 * the same run has to be reproducible from a seed, and a float rate compared against
 * a float draw is a comparison two builds can land on either side of.
 */
struct DecayParams {
    core::u32 substitutionPerMillionPerCentury{40u}; ///< A base read as a different base.
    core::u32 breakPerMillionPerCentury{15u};        ///< A strand cut, losing it entirely.
    core::u32 dropoutPerMillionPerCentury{25u};      ///< A strand never sequenced at all.
};

/**
 * @struct DecayReport
 * @brief What a simulated span of years did to a pool of strands.
 */
struct DecayReport {
    core::u32 strands{0u};       ///< Strands the pool started with.
    core::u32 intact{0u};        ///< Strands that came back readable.
    core::u32 lost{0u};          ///< Strands broken or never sequenced.
    core::u32 substitutions{0u}; ///< Bases that came back wrong inside surviving strands.
};

/**
 * @brief Ages a pool of droplets by @p years, in silico.
 *
 * Three failure modes, because they are not the same failure and the codes that
 * answer them are not the same code. A strand that BREAKS or is never sequenced is an
 * erasure: it is simply absent, and the fountain replaces it with another. A base read
 * as a different base is an ERROR: the strand arrives, looks fine, and is wrong —
 * which no fountain can see and only Reed-Solomon within the strand can catch.
 *
 * Simulating both is the point of the instrument: density and redundancy are the two
 * levers worth pulling, and neither can be tuned without a way to measure survival.
 *
 * @param pool      Strands, aged in place.
 * @param years     How long they sat.
 * @param params    The three rates.
 * @param stream    Determinism anchor; the same seed ages the same pool identically.
 * @param outReport Receives the tally.
 */
void simulateDecay(lpl::pmr::vector<Droplet> &pool, core::u32 years, const DecayParams &params, math::Random &stream,
                   DecayReport &outReport);

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_FOUNTAIN_HPP
