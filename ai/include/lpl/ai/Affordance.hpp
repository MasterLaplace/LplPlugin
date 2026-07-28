/**
 * @file Affordance.hpp
 * @brief The world advertises what can be done with it.
 *
 * The usual arrangement has the AI ask questions: is there food nearby? a place
 * to hide? a corpse? Every new kind of object adds a question, so the behaviour
 * tree grows by one branch per content addition and the two become impossible to
 * change independently.
 *
 * UE's Smart Objects invert it. The corpse broadcasts "eat me". The town
 * broadcasts "dangerous — unless you are starving". The den broadcasts "shelter
 * here". An agent does not know what a corpse is; it knows it is hungry and that
 * something nearby offers `Eat`. Adding a new kind of object then adds **no
 * branch anywhere** — which is the whole reason to do it this way.
 *
 * The condition is what makes it more than a tag list. "Dangerous unless
 * starving" is exactly the mechanic that turns an overpopulated forest into a
 * raid on a village, and it lives in the object's advertisement rather than in
 * the raider.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_AFFORDANCE_HPP
#    define LPL_AI_AFFORDANCE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/**
 * @enum AffordanceKind
 * @brief What an object offers. Combine as a mask.
 */
enum class AffordanceKind : core::u16 {
    None = 0,
    Eat = 1u << 0,     ///< Food: a plant, a corpse, a prey animal.
    Drink = 1u << 1,   ///< Water.
    Shelter = 1u << 2, ///< A den, a burrow, cover from weather.
    Nest = 1u << 3,    ///< Somewhere to breed.
    Danger = 1u << 4,  ///< A place to avoid — a town, a predator's territory.
    Perch = 1u << 5,   ///< High ground, a vantage point.
    Block = 1u << 6    ///< Impassable.
};

[[nodiscard]] constexpr core::u16 operator|(AffordanceKind a, AffordanceKind b) noexcept
{
    return static_cast<core::u16>(static_cast<core::u16>(a) | static_cast<core::u16>(b));
}

/**
 * @struct Affordance
 * @brief One advertisement, at one place.
 */
struct Affordance {
    core::u32 cell{0u};    ///< Flat cell index it applies at.
    core::u16 kinds{0u};   ///< Mask of @ref AffordanceKind.
    core::u16 radius{1u};  ///< Cells it reaches.
    math::Fixed32 value{}; ///< How good an offer it is; ordering only.

    /**
     * @brief Need below which this offer is ignored.
     *
     * The "dangerous unless starving" clause, generalised. An affordance may
     * require the seeker to want it badly enough — which is how a village becomes
     * attractive to a starving pack without a single line of code that mentions
     * villages or raids.
     */
    math::Fixed32 requiredNeed{};
};

/**
 * @class AffordanceRegistry
 * @brief What the world currently offers, and where.
 *
 * A flat list scanned linearly. That is the right structure here and not a
 * concession: the count is bounded by what is realised at any moment, and a
 * spatial index over a few dozen entries costs more to maintain than to skip.
 * When the count grows, the existing @c WorldPartition is the index to use —
 * writing a second broad-phase in this file would be the duplication this project
 * keeps having to undo.
 */
class AffordanceRegistry {
public:
    void clear() noexcept { _offers.clear(); }
    void add(const Affordance &offer) { _offers.push_back(offer); }
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_offers.size()); }
    [[nodiscard]] const Affordance &operator[](core::u32 i) const { return _offers[i]; }

    /**
     * @brief The best offer of a kind near a cell, for a seeker with a given need.
     *
     * @param kinds    Mask of what the seeker is looking for.
     * @param x        Seeker column.
     * @param z        Seeker row.
     * @param width    Grid width, to unpack cell indices.
     * @param need     How badly the seeker wants it, in [0, 1].
     * @param outIndex Receives the winning offer's index.
     * @return true when something was found.
     */
    [[nodiscard]] bool best(core::u16 kinds, core::u32 x, core::u32 z, core::u32 width, math::Fixed32 need,
                            core::u32 &outIndex) const
    {
        bool found = false;
        math::Fixed32 bestValue{};
        for (core::u32 i = 0u; i < _offers.size(); ++i)
        {
            const Affordance &offer = _offers[i];
            if ((offer.kinds & kinds) == 0u)
                continue;
            if (need < offer.requiredNeed)
                continue; // Not desperate enough to accept this one.

            const core::u32 ox = offer.cell % width;
            const core::u32 oz = offer.cell / width;
            const core::u32 dx = ox > x ? ox - x : x - ox;
            const core::u32 dz = oz > z ? oz - z : z - oz;
            if ((dx > dz ? dx : dz) > offer.radius)
                continue;

            // Strictly greater, so ties go to the earlier entry — a total order
            // that does not depend on how the registry was filled.
            if (!found || offer.value > bestValue)
            {
                found = true;
                bestValue = offer.value;
                outIndex = i;
            }
        }
        return found;
    }

private:
    lpl::pmr::vector<Affordance> _offers;
};

} // namespace lpl::ai

#endif // LPL_AI_AFFORDANCE_HPP
