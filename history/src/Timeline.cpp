/**
 * @file Timeline.cpp
 * @brief The canonical order, and the fold that depends on it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/history/Fold.hpp>
#include <lpl/history/Timeline.hpp>

namespace lpl::history {

namespace {

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
    if (a.fact.fromDay != b.fact.fromDay)
        return a.fact.fromDay < b.fact.fromDay;
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

bool Timeline::constraintsStartingIn(core::i32 fromDay, core::i32 toDay, core::u32 &outFirst,
                                     core::u32 &outCount) const noexcept
{
    outFirst = 0u;
    outCount = 0u;

    for (core::usize i = 0u; i < _constraints.size(); ++i)
    {
        const core::i32 start = _constraints[i].fact.fromDay;
        // @warning A RANGE, not an equality, and that is the whole repair. Asking for the constraints
        // of one exact day is correct only while the clock advances one day at a time; the moment
        // an era is geared to cross a silent century in a step -- which is what a gearing is FOR --
        // every constraint inside the step is silently never applied, and the chronicle looks
        // complete. A tick covers a span, and this returns what starts inside it.
        if (start < fromDay)
            continue;
        if (start > toDay)
            break; // sorted by start, so nothing later can qualify
        if (outCount == 0u)
            outFirst = static_cast<core::u32>(i);
        ++outCount;
    }
    return outCount != 0u;
}

core::u32 Timeline::fold(core::u32 seed) const noexcept
{
    core::u32 hash = seed;
    // @warning Through the shared fold, not a local copy of the constant: a signature exists to
    // be the same number on two machines, so the function producing it is the last thing
    // that should exist in several versions. This file and Chronicle.cpp each carried
    // their own until a third consumer was about to make it three.
    const auto absorb = [&hash](core::u32 word) { hash = foldWord(hash, word); };

    for (core::usize i = 0u; i < _constraints.size(); ++i)
    {
        const Constraint &c = _constraints[i];
        absorb(c.fact.subject);
        absorb(c.fact.predicate);
        absorb(c.fact.object);
        absorb(static_cast<core::u32>(c.fact.fromDay));
        absorb(static_cast<core::u32>(c.fact.toDay));
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
    outFirst = _constraints[0].fact.fromDay;
    outLast = _constraints[0].fact.toDay;
    for (core::usize i = 1u; i < _constraints.size(); ++i)
    {
        if (_constraints[i].fact.fromDay < outFirst)
            outFirst = _constraints[i].fact.fromDay;
        if (_constraints[i].fact.toDay > outLast)
            outLast = _constraints[i].fact.toDay;
    }
    return true;
}

} // namespace lpl::history
