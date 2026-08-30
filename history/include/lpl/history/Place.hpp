/**
 * @file Place.hpp
 * @brief Where a named place is, and the seam that answers it.
 *
 * @warning **The seam exists because the dependency runs the other way.** A gazetteer of the ancient
 * world is a corpus, and corpora live in LplKnowledge -- which depends on LplPlugin and not the
 * reverse. So this module cannot reach for `knowledge::GazetteerEntryV1`; it declares the
 * question it needs answered and lets whoever holds the corpus answer it. The same remedy as
 * `net::Endpoint` and `agent::Decision`: extract the shared concept into a bounded form both
 * sides can express, and let the rich side adapt.
 *
 * @warning **Coordinates are Fixed32 and therefore authoritative.** Where a place is decides where a
 * body walks and how long it takes to arrive, so two targets disagreeing in the last bit would
 * run two different journeys from one corpus.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_PLACE_HPP
#    define LPL_LPL_HISTORY_PLACE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::history {

/**
 * @struct Place
 * @brief A named place, positioned and dated.
 */
struct Place {
    core::u32 id{0u};     ///< The gazetteer's own identifier. Never a hash of a name.
    math::Fixed32 x{};    ///< World position. What the corpus calls a longitude, projected.
    math::Fixed32 z{};    ///< World position.
    core::i32 minYear{0}; ///< First year attested.
    core::i32 maxYear{0}; ///< Last year attested.
    bool located{false};  ///< Whether the coordinates mean anything. See below.
};

/**
 * @class IPlaceResolver
 * @brief Answers where a place identifier is.
 *
 * @warning A resolver may legitimately answer "I know this place and I do not know where it is" --
 * `located` false. Thousands of ancient places are known only from texts and have never been
 * found on the ground; a resolver that invented coordinates for them would turn "nobody knows
 * where Cimmeria was" into a position a body walks to.
 */
class IPlaceResolver {
public:
    virtual ~IPlaceResolver() = default;

    /**
     * @brief Looks a place up.
     *
     * @param id  The identifier.
     * @param out Receives it.
     * @return false when the gazetteer does not carry that identifier at all, which is
     *         different from carrying it without a position.
     */
    [[nodiscard]] virtual bool resolve(core::u32 id, Place &out) const = 0;

    /**
     * @brief Collects the places a corpus says this one is connected to.
     *
     * @warning **The "facts first" input, and the reason it is on this interface rather than inferred.**
     * A corpus states which places were linked -- a road, a route, a sea lane somebody sailed -- and
     * a simulation has no business inventing that. What it MAY invent is where the road runs
     * across the relief, which belongs to whatever owns the terrain.
     *
     * @warning A default of zero, deliberately: a resolver that knows positions and no links is a
     * legitimate resolver, and the walk simply falls back to geometry. Making this pure would
     * force every implementer to state that it has no links, which is how an interface acquires
     * methods that only ever return nothing.
     *
     * @param id       The place.
     * @param out      Receives the neighbours.
     * @param capacity Room in @p out.
     * @return How many were written.
     */
    [[nodiscard]] virtual core::u32 linkedPlaces(core::u32 id, core::u32 *out, core::u32 capacity) const
    {
        (void) id;
        (void) out;
        (void) capacity;
        return 0u;
    }
};

/**
 * @brief Whether a place was there in a given year.
 *
 * @warning An UNDATED place answers true for every year, and that is deliberate: absence of a window is
 * absence of knowledge, not a claim that the place never existed. Refusing it would silently
 * erase every place whose dating nobody has settled.
 *
 * @param place The place.
 * @param year  The year.
 * @return true when the place may be used in that year.
 */
[[nodiscard]] constexpr bool existsInYear(const Place &place, core::i32 year) noexcept
{
    if (place.minYear == 0 && place.maxYear == 0)
        return true;
    return year >= place.minYear && year <= place.maxYear;
}

/**
 * @struct RouteLeg
 * @brief One waypoint on the way from one place to another.
 */
struct RouteLeg {
    math::Fixed32 x{};
    math::Fixed32 z{};
};

/**
 * @class IRouteResolver
 * @brief Answers HOW to get from one place to another.
 *
 * @warning **The other half of "facts first, deterministic fill".** @ref IPlaceResolver::linkedPlaces
 * answers which places a corpus says were connected -- that is evidence, and no simulation may
 * invent it. This answers where the road actually runs across the relief, which no source
 * records and which a generator is entitled to decide: `procgen::routeLeastCost` already does it,
 * paying a cost per cell so a road follows valleys instead of cutting straight, and reusing
 * existing road so a network grows along its own trunk.
 *
 * @warning **Waypoints, not cells.** A continental route is thousands of cells and a body cannot carry
 * that; simplifying the path is the resolver's job, because only it knows what detail its own
 * terrain justified. A resolver that returned every cell would make the bound here a truncation
 * of the road rather than a summary of it.
 *
 * @warning Water is a COST in that model, not a wall -- which is the honest answer to "what if he took a
 * ship". A crossing that a corpus attests is a crossing that happened; making the sea impassable
 * would contradict the evidence rather than model it.
 *
 * @warning A default of zero, as with `linkedPlaces`: a resolver that knows positions and no terrain is
 * legitimate, and the walk then goes straight. Making this pure would force every implementer to
 * declare it has no roads.
 */
class IRouteResolver {
public:
    virtual ~IRouteResolver() = default;

    /**
     * @brief Plans the way from one place to another.
     *
     * @param fromPlace Where the body is.
     * @param toPlace   Where it is going.
     * @param out       Receives the waypoints, in order, excluding the start.
     * @param capacity  Room in @p out.
     * @return How many waypoints were written; zero means "go straight".
     */
    [[nodiscard]] virtual core::u32 route(core::u32 fromPlace, core::u32 toPlace, RouteLeg *out,
                                          core::u32 capacity) const
    {
        (void) fromPlace;
        (void) toPlace;
        (void) out;
        (void) capacity;
        return 0u;
    }
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_PLACE_HPP
