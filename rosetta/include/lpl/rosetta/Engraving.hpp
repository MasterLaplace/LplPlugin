/**
 * @file Engraving.hpp
 * @brief Physical layout: redundancy, corners, error correction.
 *
 * The bootstrap sequence is duplicated at the four corners and the centre so the
 * artifact stays decodable after losing half its surface; a fifth of the area is
 * parity.
 *
 * Two kinds of redundancy, because they answer two different accidents. The parity
 * answers wear — scattered damage a Reed-Solomon column absorbs. The five copies of
 * the bootstrap answer AMPUTATION: a plate broken in half has lost an arbitrary
 * contiguous region, and no per-column code survives that. Corners and centre is the
 * cheapest arrangement where every straight cut leaves at least one copy whole.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ROSETTA_ENGRAVING_HPP
#    define LPL_LPL_ROSETTA_ENGRAVING_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/rosetta/Bootstrap.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::rosetta {

/**
 * @brief Copies of the bootstrap a plate carries.
 *
 * Five: four corners and the centre. Any straight cut through a rectangle leaves at
 * least one of those five untouched, which is the property the number is chosen for
 * rather than a round figure.
 */
inline constexpr core::u32 kBootstrapCopies = 5u;

/**
 * @brief The four bytes a replica starts with, so one can be found without a table.
 */
inline constexpr core::u8 kReplicaMagic[4] = {'L', 'P', 'L', 'R'};

/**
 * @enum Medium
 * @brief What the plate is made of.
 *
 * Carried as a label and nothing else: the layout does not change with the substrate,
 * and pretending it does would be inventing a rule a finder has to be told.
 */
enum class Medium : core::u32 {
    FusedQuartz = 0u,
    NickelPlate = 1u,
    Synthetic = 2u,
};

/**
 * @struct EngravingReport
 * @brief What reading a plate had to do.
 */
struct EngravingReport {
    bool bootstrapFound{false};    ///< At least one replica survived intact.
    bool payloadRecovered{false};  ///< The payload came back whole.
    core::u32 replicasIntact{0u};  ///< Replicas whose fold checked out.
    core::u32 repairedColumns{0u}; ///< Columns the parity had to correct.
    core::u32 repairedBytes{0u};   ///< Symbols the parity changed.
};

/**
 * @class Engraving
 * @brief Lays out a plate, and reads one back.
 */
class Engraving {
public:
    Engraving() noexcept = default;

    /**
     * @brief Names the substrate. Affects the label, never the layout.
     * @param medium What the plate is made of.
     */
    void setMedium(Medium medium) noexcept { _medium = medium; }

    /**
     * @brief Sets the parity share, in thousandths of the coded area.
     * @param permille Share; clamped so at least one parity row exists.
     */
    void setParityShare(core::u32 permille) noexcept { _parityPermille = permille; }

    /**
     * @brief Builds the plate image.
     * @param bootstrap   The four levels a reader needs first.
     * @param payload     What the artifact is for.
     * @param payloadSize Its length.
     * @return false when the payload is empty or the parity share is unusable.
     */
    [[nodiscard]] bool engrave(const Bootstrap &bootstrap, const core::u8 *payload, core::u32 payloadSize);

    /**
     * @brief The plate, as bytes.
     * @return The image; empty before @ref engrave.
     */
    [[nodiscard]] const lpl::pmr::vector<core::u8> &image() const noexcept { return _image; }

    /**
     * @brief Reads a plate: repairs what the parity can, then finds a live replica.
     *
     * Static, and it takes the image rather than reading a member, because the whole
     * point of the artifact is that reading it must not need the object that wrote it.
     *
     * @param bytes         The plate, modified where the parity corrected it.
     * @param size          Its length.
     * @param outSpec       Receives the engraved machine specification, level 3.
     * @param outPayload    Receives the payload.
     * @param outReport     Receives what had to be done.
     * @return true when the payload came back whole.
     */
    [[nodiscard]] static bool read(core::u8 *bytes, core::u32 size, lpl::pmr::vector<core::u8> &outSpec,
                                   lpl::pmr::vector<core::u8> &outPayload, EngravingReport &outReport);

private:
    lpl::pmr::vector<core::u8> _image{};
    Medium _medium{Medium::FusedQuartz};
    core::u32 _parityPermille{200u}; ///< A fifth of the coded area, as the file comment says.
};

} // namespace lpl::rosetta

#endif // LPL_LPL_ROSETTA_ENGRAVING_HPP
