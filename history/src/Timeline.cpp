/**
 * @file Timeline.cpp
 * @brief The canonical order, and the fold that depends on it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/Timeline.hpp>

namespace lpl::history {

namespace {

constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Total order over constraints: year, then contents, then source.
 *
 * A total order, not merely chronological. Two constraints in the same year have to
 * be ordered by something, and "whichever the curator typed first" would make the
 * fold a property of a text file.
 *
 * @param a First.
 * @param b Second.
 * @return true when @p a sorts before @p b.
 */
[[nodiscard]] bool sortsBefore(const Constraint &a, const Constraint &b) noexcept
{
    if (a.fact.fromYear != b.fact.fromYear)
        return a.fact.fromYear < b.fact.fromYear;
    if (a.fact.subject != b.fact.subject)
        return a.fact.subject < b.fact.subject;
    if (a.fact.predicate != b.fact.predicate)
        return a.fact.predicate < b.fact.predicate;
    if (a.fact.object != b.fact.object)
        return a.fact.object < b.fact.object;
    return a.fact.source < b.fact.source;
}

} // namespace

void Timeline::add(const Constraint &constraint) { _constraints.push_back(constraint); }

void Timeline::finalise()
{
    // Insertion sort: a timeline is hundreds of constraints, not millions, and this is
    // stable and obvious. A faster sort here would be a faster sort nobody measured.
    for (core::usize i = 1u; i < _constraints.size(); ++i)
    {
        const Constraint held = _constraints[i];
        core::usize j = i;
        while (j > 0u && sortsBefore(held, _constraints[j - 1u]))
        {
            _constraints[j] = _constraints[j - 1u];
            --j;
        }
        _constraints[j] = held;
    }
}

const Constraint &Timeline::at(core::u32 index) const noexcept
{
    static const Constraint kEmpty{};
    return index < _constraints.size() ? _constraints[index] : kEmpty;
}

bool Timeline::constraintsOfYear(core::i32 year, core::u32 &outFirst, core::u32 &outCount) const noexcept
{
    outFirst = 0u;
    outCount = 0u;

    for (core::usize i = 0u; i < _constraints.size(); ++i)
    {
        if (_constraints[i].fact.fromYear != year)
        {
            if (outCount != 0u)
                break; // sorted, so the run is over
            continue;
        }
        if (outCount == 0u)
            outFirst = static_cast<core::u32>(i);
        ++outCount;
    }
    return outCount != 0u;
}

core::u32 Timeline::fold(core::u32 seed) const noexcept
{
    core::u32 hash = seed;
    const auto absorb = [&hash](core::u32 word) { hash = (hash ^ word) * kFnv1aPrime; };

    for (core::usize i = 0u; i < _constraints.size(); ++i)
    {
        const Constraint &c = _constraints[i];
        absorb(c.fact.subject);
        absorb(c.fact.predicate);
        absorb(c.fact.object);
        absorb(static_cast<core::u32>(c.fact.fromYear));
        absorb(static_cast<core::u32>(c.fact.toYear));
        absorb(c.fact.source);
        absorb(static_cast<core::u32>(c.fact.sigma.raw()));
        absorb(static_cast<core::u32>(c.kind));
        absorb(static_cast<core::u32>(c.confidence.raw()));
    }
    return hash;
}

bool Timeline::span(core::i32 &outFirst, core::i32 &outLast) const noexcept
{
    if (_constraints.empty())
        return false;
    outFirst = _constraints[0].fact.fromYear;
    outLast = _constraints[0].fact.toYear;
    for (core::usize i = 1u; i < _constraints.size(); ++i)
    {
        if (_constraints[i].fact.fromYear < outFirst)
            outFirst = _constraints[i].fact.fromYear;
        if (_constraints[i].fact.toYear > outLast)
            outLast = _constraints[i].fact.toYear;
    }
    return true;
}

} // namespace lpl::history
