/**
 * @file ShapeGrammar.hpp
 * @brief Turning a footprint into a building, by splitting a box.
 *
 * A plot is a rectangle. Raising it to a fixed height gives a prism, and a town
 * of prisms reads as a bar chart. What separates a building from a prism is
 * *articulation*: a base that differs from the floors above it, floors that
 * repeat, and a roof that terminates. Those are not three special cases, they are
 * one operation — split a volume along an axis, and recurse.
 *
 * That is the CGA-shape family (Müller et al.), reduced here to the subset a
 * deterministic generator can afford: split with absolute, relative and
 * repeating sizes; a weighted stochastic choice; and a terminal that fills.
 *
 * Two things make this worth having beyond the shapes it draws.
 *
 * **It is the surface an author writes.** A grammar is a short string, and a
 * short string is what a language model, a `.lplscene` document or a command line
 * can carry. A C++ builder API cannot cross any of those boundaries.
 *
 * **The same grammar applies along a line.** UE's PCG uses exactly this notation
 * for fences and guardrails following a spline: `{[A,P]:2,[BL,P]:1,[BS,P]:1}*,[G,P]`
 * reads as "mostly intact panels, sometimes broken ones, repeated to fill, then a
 * gate". One parser, two geometries.
 *
 * @warning Every stochastic choice draws from a stream keyed by the plot, never
 *          from a running generator. Two buildings must not depend on the order
 *          they were raised in, or a town becomes a different town when a plot is
 *          added at its edge.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_SHAPEGRAMMAR_HPP
#    define LPL_PROCGEN_SHAPEGRAMMAR_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Extrusion.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Settlement.hpp>

namespace lpl::procgen {

/// Longest grammar string accepted. Kernel-safe: parsing is bounded up front.
inline constexpr core::u32 kMaxGrammarLength = 256u;

/// Most modules one grammar may name.
inline constexpr core::u32 kMaxGrammarModules = 16u;

/**
 * @brief The material a grammar symbol names.
 *
 * `A` is 1, `B` is 2, and so on. Part of the notation's contract rather than an
 * implementation detail: an author writing `[A,P]` in a document and a test
 * checking what came out have to agree, and they can only do that against one
 * definition.
 *
 * @param letter Symbol letter.
 * @return The material id, or 0 when the character is not a symbol.
 */
[[nodiscard]] constexpr core::u8 materialForSymbol(char letter) noexcept
{
    if (letter >= 'A' && letter <= 'Z')
        return static_cast<core::u8>(letter - 'A' + 1);
    if (letter >= 'a' && letter <= 'z')
        return static_cast<core::u8>(letter - 'a' + 1);
    return 0u;
}

/**
 * @enum GrammarAxis
 * @brief Which way a split cuts.
 */
enum class GrammarAxis : core::u8 {
    X = 0, ///< Across the plot's width.
    Y,     ///< Vertically: base, floors, roof.
    Z      ///< Across the plot's depth.
};

/**
 * @struct GrammarModule
 * @brief One alternative in a probabilistic block: a symbol and its weight.
 */
struct GrammarModule {
    core::u8 material{1u}; ///< Material id emitted; 0 leaves the slot empty.
    core::u8 height{1u};   ///< Levels this module occupies.
    core::u16 weight{1u};  ///< Relative probability inside its block.
};

/**
 * @struct SequenceGrammar
 * @brief A repeated probabilistic block, then a fixed terminator.
 *
 * The wire form of `{[A,P]:2,[BL,P]:1,[BS,P]:1}*,[G,P]`: a set of weighted
 * alternatives that repeats to fill the available run, followed by a module that
 * is always placed last.
 */
struct SequenceGrammar {
    GrammarModule alternatives[kMaxGrammarModules]{}; ///< The `{...}` block.
    core::u32 alternativeCount{0u};                   ///< Entries in @c alternatives.
    GrammarModule terminator{};                       ///< The `,[G,P]` tail.
    bool hasTerminator{false};                        ///< Whether a tail was given.
    core::u32 totalWeight{0u};                        ///< Sum of the alternatives' weights.
};

/**
 * @struct BuildingGrammarParams
 * @brief How a plot becomes a volume.
 *
 * A deliberately small vocabulary: a base course, a repeated storey, and a roof.
 * Everything a town needs to stop reading as a bar chart, and nothing that would
 * make the grammar a second geometry engine.
 */
struct BuildingGrammarParams {
    core::u32 seed{0u};         ///< 0 derives a stream from the world seed.
    core::u32 minFloors{1u};    ///< Fewest storeys any building gets.
    core::u32 maxFloors{4u};    ///< Most storeys any building gets.
    core::u32 baseHeight{1u};   ///< Levels of ground-floor course.
    core::u32 floorHeight{1u};  ///< Levels per repeated storey.
    core::u32 roofHeight{1u};   ///< Levels of roof; 0 leaves the top flat.
    core::u32 inset{0u};        ///< Cells the walls step in from the plot edge.
    core::f32 roofTaper{0.5f};  ///< Share of the roof that steps inward, in [0, 1].
    core::u8 baseMaterial{2u};  ///< Material of the base course.
    core::u8 wallMaterial{1u};  ///< Material of the storeys.
    core::u8 roofMaterial{3u};  ///< Material of the roof.
    bool hollow{true};          ///< Leave the interior empty above the base.
};

/**
 * @brief Parses the report's grammar string into a @ref SequenceGrammar.
 *
 * Grammar accepted, in full:
 *
 * @code
 *   sequence := '{' alt (',' alt)* '}' '*' (',' module)?
 *   alt      := module (':' weight)?
 *   module   := '[' letters (',' letters)* ']'
 * @endcode
 *
 * A module's symbol is mapped to a material by @p symbols: the first letter of
 * the symbol indexes it, which keeps the notation the report uses (`A`, `BL`,
 * `BS`, `G`) without inventing a symbol table format nobody wrote down.
 *
 * A malformed string is REFUSED outright rather than parsed as far as it goes. A
 * half-applied grammar produces a building that looks plausible and is wrong,
 * which is worse than one that does not appear — and an author fixing a typo
 * needs to be told there was one.
 *
 * @param text    Grammar string, NUL-terminated.
 * @param out     Receives the parsed grammar on success.
 * @return true when the whole string was consumed and the grammar is usable.
 */
[[nodiscard]] bool parseSequenceGrammar(const char *text, SequenceGrammar &out);

/**
 * @brief Fills a run of slots by repeating a sequence grammar.
 *
 * @param grammar The parsed grammar.
 * @param length  Slots to fill.
 * @param seed    Stream for the weighted choices.
 * @param out     Receives one module per slot; sized to @p length.
 * @return Number of slots written.
 */
core::u32 applySequence(const SequenceGrammar &grammar, core::u32 length, core::u32 seed,
                        lpl::pmr::vector<GrammarModule> &out);

/**
 * @brief Raises one plot into a volume by splitting it.
 *
 * The volume covers the plot's own footprint, so a caller composites it into a
 * larger one at the plot's offset rather than the grammar having to know where
 * the town is.
 *
 * @param plot   Footprint to raise.
 * @param params Grammar parameters.
 * @param seed   Stream for this plot's choices.
 * @return The building's volume.
 */
[[nodiscard]] VoxelVolume buildingVolume(const BuildingPlot &plot, const BuildingGrammarParams &params,
                                         core::u32 seed);

/**
 * @brief Raises every plot of a settlement into one volume.
 *
 * @warning Only cells the settlement map still calls @c Plot are raised. A plot
 *          is a rectangle proposed BEFORE the streets were cut, so part of it is
 *          usually road by the time this runs — and raising the rectangle puts
 *          buildings in the middle of their own street. Measured once already:
 *          the tallest road column came out exactly as tall as the tallest
 *          building.
 *
 * @param settlement The layout, for the Plot test.
 * @param plots      Footprints to raise.
 * @param params     Grammar parameters.
 * @param worldSeed  Stream the per-plot seeds derive from.
 * @param levels     Height of the output volume.
 * @return The town's volume, of the settlement's dimensions.
 */
[[nodiscard]] VoxelVolume buildTown(const SettlementMap &settlement, const lpl::pmr::vector<BuildingPlot> &plots,
                                    const BuildingGrammarParams &params, core::u32 worldSeed, core::u32 levels);

/**
 * @brief Places modules along a marked path — fences, lamps, guardrails.
 *
 * The linear application of the same grammar. Cells are visited in scan order so
 * the result does not depend on how the path was traced.
 *
 * @param path      Non-zero marks a cell of the line.
 * @param grammar   Parsed sequence grammar.
 * @param seed      Stream for the weighted choices.
 * @param levels    Height of the output volume.
 * @param outCount  Receives how many modules were placed.
 * @return A volume holding the placed modules.
 */
[[nodiscard]] VoxelVolume decoratePath(const Grid<core::u8> &path, const SequenceGrammar &grammar, core::u32 seed,
                                       core::u32 levels, core::u32 &outCount);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_SHAPEGRAMMAR_HPP
