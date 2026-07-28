/**
 * @file LSystem.hpp
 * @brief Grammars that grow: rewriting a string into a branching structure.
 *
 * An L-system is a handful of rewrite rules applied in parallel, over and over.
 * `F -> F[+F]F[-F]` says "a segment becomes a segment that sprouts two branches
 * and continues" — and after five rounds that sentence is a tree. The same
 * machinery draws road networks, river deltas, lightning, cracks, and vein
 * structures, because all of them are recursive branching seen from different
 * angles.
 *
 * What makes it worth having next to noise and Voronoi is that it produces
 * CONNECTED structures. A road network from noise is a pattern of road-coloured
 * cells; a road network from an L-system is a graph you can actually walk from
 * end to end, because it was grown from a single stem.
 *
 * Interpretation is a turtle on the grid: it carries a position and a heading,
 * the alphabet moves and turns it, and brackets push and pop its state. Headings
 * are quantised to 16 compass directions so the turtle never needs a sine — the
 * step table is a compile-time constant, which keeps the whole thing exact.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_LSYSTEM_HPP
#    define LPL_PROCGEN_LSYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/std/string.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/// Compass directions the turtle can face; a full turn is 16 steps.
inline constexpr core::u32 kTurtleDirections = 16u;

/**
 * @struct LRule
 * @brief One rewrite: every @c symbol becomes @c replacement.
 *
 * @c weight makes the rule one alternative among several for the same symbol,
 * chosen in proportion to its weight. Without alternatives an L-system is a
 * function: one grammar, one shape, so every tree in a forest is the same tree
 * and every district the same district. Weighted alternatives are how the shape
 * grammars in the literature express `{[A,P]:2, [BL,P]:1, [BS,P]:1}` — two chances
 * in four of an intact wall, one each of the two broken ones.
 */
struct LRule {
    char symbol{'F'};                ///< Symbol to replace.
    lpl::pmr::string replacement;    ///< What it becomes.
    core::u32 weight{1u};            ///< Relative chance among alternatives for this symbol.
};

/**
 * @struct LSystemParams
 * @brief A grammar and how many times to apply it.
 *
 * The alphabet the turtle understands:
 *  - `F` draw a segment forward
 *  - `f` move forward without drawing
 *  - `+` turn right, `-` turn left
 *  - `[` push state, `]` pop state
 *  - any other symbol is inert (useful as a rewrite-only variable)
 */
struct LSystemParams {
    lpl::pmr::string axiom{"F"};        ///< Starting string.
    lpl::pmr::vector<LRule> rules;      ///< Rewrite rules.
    core::u32 iterations{4u};           ///< Rewrite rounds.
    core::u32 maxLength{16384u};        ///< Cap on the expanded string.
    core::u32 seed{0x15A5u};            ///< Determinism anchor for weighted alternatives.
};

/**
 * @struct TurtleParams
 * @brief How the expanded string is drawn.
 */
struct TurtleParams {
    core::u32 startX{32u};       ///< Starting column.
    core::u32 startZ{60u};       ///< Starting row.
    core::u32 startDirection{4u};///< Starting heading, in 1/16 turns (4 = north).
    core::u32 stepLength{4u};    ///< Cells drawn per `F`.
    core::u32 turnAmount{2u};    ///< Heading steps per `+` or `-`.
    core::u32 thickness{0u};     ///< Extra radius around each drawn cell.
    core::f32 stepDecay{1.0f};   ///< Step length multiplier per branch depth, in (0, 1].
};

/// Which family of field lines a region of a tensor field carries.
enum class FieldPattern : core::u8 {
    Grid = 0, ///< Parallel field lines at a fixed bearing: an orthogonal street plan.
    Radial    ///< Field lines converging on a centre: a hub-and-spoke old town.
};

/**
 * @struct FieldRegion
 * @brief One influence in a tensor field.
 */
struct FieldRegion {
    FieldPattern pattern{FieldPattern::Grid}; ///< Grid or radial.
    core::u32 centerX{0u};   ///< Where this influence is anchored.
    core::u32 centerZ{0u};   ///< Where this influence is anchored.
    core::u32 bearing{0u};   ///< For Grid: the heading its lines follow, in 1/16 turns.
    core::f32 strength{1.0f};///< Weight at the centre.
    core::f32 falloff{0.04f};///< Weight lost per cell of distance from the centre.
};

/**
 * @brief A tensor field baked to a grid: the heading a road should follow per cell.
 *
 * Values are headings in 1/16 turns, matching @ref kTurtleDirections.
 */
using HeadingField = Grid<core::u8>;

/**
 * @brief Bakes a set of influences into a per-cell heading field.
 *
 * This is the half of the Parish and Müller road model that a bare L-system is
 * missing. Their grammar is not a closed rewrite system: it is a rewrite system
 * under tension between **global goals**, which pull growth toward where the
 * designer wants density, and **local constraints**, which snap a proposed segment
 * onto what is already built. A pure grammar has neither, so it produces the same
 * self-similar fan whatever the map looks like, and no amount of tuning makes it
 * respond to a coastline or to a town centre.
 *
 * The field carries the global goal. A radial region makes streets converge on a
 * historic centre; a grid region imposes a bearing on a planned district; and
 * between them the weights blend, so a city can go from a spider's web downtown to
 * an American grid at its edges without either being drawn separately.
 *
 * Baking to a grid rather than evaluating the field per query is the optimisation
 * the literature specifically calls for: the turtle then reads a cell instead of
 * summing every influence at every step.
 *
 * @param width   Cells along X.
 * @param depth   Cells along Z.
 * @param regions Influences to blend.
 * @return A heading per cell (0 where no region has any weight).
 */
[[nodiscard]] HeadingField bakeHeadingField(core::u32 width, core::u32 depth,
                                            const lpl::pmr::vector<FieldRegion> &regions);

/**
 * @brief Draws an expanded string, steering the turtle by a heading field.
 *
 * The turtle's own heading becomes an offset from what the field says rather than
 * an absolute bearing: `+` and `-` still turn it, but a straight run curves along
 * the field lines. That is what makes one grammar produce a web downtown and a grid
 * in the suburbs.
 *
 * @param expanded The string from @ref expandLSystem.
 * @param params   Start position, step size and turn angle.
 * @param field    Per-cell headings from @ref bakeHeadingField; must match @p canvas.
 * @param conform  How strongly the field overrides the turtle, in [0, 1].
 * @param canvas   Grid to draw on; drawn cells are set to 1.
 * @return Number of cells drawn.
 */
core::u32 drawTurtleInField(const lpl::pmr::string &expanded, const TurtleParams &params, const HeadingField &field,
                            core::f32 conform, Grid<core::u8> &canvas);

/**
 * @brief Applies the rewrite rules @c iterations times.
 *
 * Stops early if the string would exceed @c maxLength: an L-system's growth is
 * exponential, and a rule set that looks harmless can produce megabytes in eight
 * rounds. Truncating is what keeps this usable in a kernel with a 4 MiB heap.
 *
 * @param params Grammar and iteration count.
 * @return The expanded string.
 */
[[nodiscard]] lpl::pmr::string expandLSystem(const LSystemParams &params);

/**
 * @brief Draws an expanded string onto a grid with a turtle.
 * @param expanded The string from @ref expandLSystem.
 * @param params   Start position, step size and turn angle.
 * @param canvas   Grid to draw on; drawn cells are set to 1.
 * @return Number of cells drawn.
 */
core::u32 drawTurtle(const lpl::pmr::string &expanded, const TurtleParams &params, Grid<core::u8> &canvas);

/**
 * @brief A ready-made branching grammar, suitable for trees or river deltas.
 * @return `F -> FF+[+F-F-F]-[-F+F+F]`, the classic bushy rule.
 */
[[nodiscard]] LSystemParams makeBranchingGrammar();

/**
 * @brief A ready-made grammar for road networks: mostly straight, forking.
 * @return A grammar whose segments run long and branch at right angles.
 */
[[nodiscard]] LSystemParams makeRoadGrammar();

} // namespace lpl::procgen

#endif // LPL_PROCGEN_LSYSTEM_HPP
