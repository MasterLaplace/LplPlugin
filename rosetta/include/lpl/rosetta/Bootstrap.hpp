/**
 * @file Bootstrap.hpp
 * @brief The five levels, from physical anchor to payload.
 *
 * Physical constants for calibration, optical read-out, pictogram dictionary and
 * type primitives, the ISA and its decompressor, then the payload. Each level is
 * readable using only the levels above it — that is the whole design constraint.
 *
 * Level 0 carries no information a reader has to be told: it is a set of ratios that
 * are the same everywhere in the universe, so a finder can check their instrument
 * against the plate rather than the other way round. Everything after it is only
 * meaningful once the previous level has been read, and that ordering is the artifact.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ROSETTA_BOOTSTRAP_HPP
#    define LPL_LPL_ROSETTA_BOOTSTRAP_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::rosetta {

/**
 * @enum BootstrapLevel
 * @brief The five, in the order they must be read.
 */
enum class BootstrapLevel : core::u32 {
    Calibration = 0u, ///< Ratios a finder can verify against physics, not against us.
    ReadOut = 1u,     ///< How a mark becomes a bit: cell pitch, orientation, row order.
    Primitives = 2u,  ///< Pictograms for the few types the levels below use.
    Machine = 3u,     ///< The instruction set, as SelfDescribing writes it.
    Payload = 4u,     ///< What the artifact is actually for.
    Count = 5u
};

/**
 * @brief The word a plate spells @p level with.
 * @param level The level.
 * @return Its name.
 */
[[nodiscard]] constexpr const char *bootstrapLevelName(BootstrapLevel level) noexcept
{
    switch (level)
    {
    case BootstrapLevel::Calibration: return "CALIBRATION";
    case BootstrapLevel::ReadOut: return "READOUT";
    case BootstrapLevel::Primitives: return "PRIMITIVES";
    case BootstrapLevel::Machine: return "MACHINE";
    case BootstrapLevel::Payload: return "PAYLOAD";
    case BootstrapLevel::Count: break;
    }
    return "?";
}

/**
 * @struct Bootstrap
 * @brief The four levels that precede the payload, as bytes.
 *
 * The payload is not here: a bootstrap is what a reader needs BEFORE the payload
 * means anything, and putting the two in one object would let a caller engrave a
 * plate whose instructions describe a payload it does not carry.
 */
struct Bootstrap {
    lpl::pmr::vector<core::u8> level[static_cast<core::u32>(BootstrapLevel::Count) - 1u];

    /**
     * @brief Total bytes across every level.
     * @return The sum.
     */
    [[nodiscard]] core::u32 totalBytes() const noexcept;
};

/**
 * @brief Builds the canonical bootstrap of this build.
 *
 * Level 3 is @ref emitSpecification verbatim — the same bytes an interpreter can be
 * rebuilt from, not a prose restatement of them. The three above it are short and
 * fixed; their content is a design exercise rather than a technical one, and what
 * matters here is that they exist, are ordered, and are engraved together.
 *
 * @return The four levels.
 */
[[nodiscard]] Bootstrap standardBootstrap();

} // namespace lpl::rosetta

#endif // LPL_LPL_ROSETTA_BOOTSTRAP_HPP
