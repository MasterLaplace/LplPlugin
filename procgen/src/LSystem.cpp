/**
 * @file LSystem.cpp
 * @brief Implementation of L-system rewriting and turtle interpretation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/LSystem.hpp>

#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Random.hpp>

namespace lpl::procgen {

namespace {

/**
 * @brief Unit step per compass direction, as Q16.16 raw values.
 *
 * Sixteen directions, precomputed. The turtle therefore never evaluates a sine:
 * a heading is an index, and moving is a table lookup. That is both exact and
 * free of any transcendental, which is what lets this run in the kernel.
 *
 * Values are round(cos/sin(k * 2pi/16) * 65536).
 */
constexpr core::i32 kStepX[kTurtleDirections] = {65536,  60547,  46341,  25080,  0, -25080, -46341, -60547,
                                                 -65536, -60547, -46341, -25080, 0, 25080,  46341,  60547};
constexpr core::i32 kStepZ[kTurtleDirections] = {0, 25080,  46341,  60547,  65536,  60547,  46341,  25080,
                                                 0, -25080, -46341, -60547, -65536, -60547, -46341, -25080};

/// Turtle state, saved and restored by brackets.
struct TurtleState {
    math::Fixed32 x{};
    math::Fixed32 z{};
    core::u32 direction{0u};
    core::u32 depth{0u};
    math::Fixed32 step{};
};

/// Paints a cell and its thickness halo.
void paint(Grid<core::u8> &canvas, core::i32 x, core::i32 z, core::u32 thickness, core::u32 &outDrawn)
{
    const core::i32 r = static_cast<core::i32>(thickness);
    for (core::i32 dz = -r; dz <= r; ++dz)
    {
        for (core::i32 dx = -r; dx <= r; ++dx)
        {
            const core::i32 cx = x + dx;
            const core::i32 cz = z + dz;
            if (!canvas.contains(cx, cz))
                continue;
            core::u8 &cell = canvas.at(static_cast<core::u32>(cx), static_cast<core::u32>(cz));
            if (cell == 0u)
            {
                cell = 1u;
                ++outDrawn;
            }
        }
    }
}

/**
 * @brief Which of the sixteen headings a vector points closest to.
 *
 * Picks by largest dot product against the step table. Sixteen multiply-adds and
 * no arctangent, which is both exact in Fixed32 and free of any transcendental —
 * an atan2 here would have been the one place this file needed libm.
 */
core::u32 nearestHeading(math::Fixed32 vx, math::Fixed32 vz)
{
    if (vx.raw() == 0 && vz.raw() == 0)
        return 0u;

    core::u32 best = 0u;
    math::Fixed32 bestDot = math::Fixed32::min();
    for (core::u32 d = 0u; d < kTurtleDirections; ++d)
    {
        const math::Fixed32 dot = vx * math::Fixed32::fromRaw(kStepX[d]) + vz * math::Fixed32::fromRaw(kStepZ[d]);
        // Strictly greater: ties go to the lowest heading, so the field is the
        // same on every target.
        if (dot > bestDot)
        {
            bestDot = dot;
            best = d;
        }
    }
    return best;
}

/// The heading pointing outward from a radial centre, for an offset of (dx, dz).
core::u32 radialBearing(core::i32 dx, core::i32 dz)
{
    return nearestHeading(math::Fixed32::fromInt(dx), math::Fixed32::fromInt(dz));
}

/**
 * @brief Would this step leave the canvas?
 *
 * A turtle that walks off the grid does not come back: its heading only changes
 * on `+` and `-`, so it keeps stepping further out while every cell it "draws"
 * is silently clipped. Measured on the road grammar, that made growth run
 * BACKWARDS — 173 draw symbols painted 43 cells, 4156 painted 32 — because the
 * longer expansion reached the edge sooner and spent the rest of its length
 * outside. Refusing the step instead keeps the walk in the world, and the next
 * turn in the string redirects it, exactly as the drunkard's walk confines its
 * diggers inside a margin.
 */
bool leavesCanvas(const Grid<core::u8> &canvas, math::Fixed32 x, math::Fixed32 z)
{
    return !canvas.contains(x.toInt(), z.toInt());
}

/// Draws a straight run between two points, painting every cell it crosses.
void drawSegment(Grid<core::u8> &canvas, math::Fixed32 fromX, math::Fixed32 fromZ, math::Fixed32 toX, math::Fixed32 toZ,
                 core::u32 thickness, core::u32 &outDrawn)
{
    // Step along the longer axis so no cell is skipped — the integer-only
    // equivalent of a DDA, with the division done once.
    const math::Fixed32 dx = toX - fromX;
    const math::Fixed32 dz = toZ - fromZ;
    const core::i32 spanX = dx.abs().toInt();
    const core::i32 spanZ = dz.abs().toInt();
    core::i32 steps = spanX > spanZ ? spanX : spanZ;
    if (steps <= 0)
        steps = 1;

    const math::Fixed32 stepCount = math::Fixed32::fromInt(steps);
    const math::Fixed32 incX = dx / stepCount;
    const math::Fixed32 incZ = dz / stepCount;

    math::Fixed32 x = fromX;
    math::Fixed32 z = fromZ;
    for (core::i32 i = 0; i <= steps; ++i)
    {
        paint(canvas, x.toInt(), z.toInt(), thickness, outDrawn);
        x = x + incX;
        z = z + incZ;
    }
}

} // namespace

lpl::pmr::string expandLSystem(const LSystemParams &params)
{
    lpl::pmr::string current = params.axiom;
    math::Random random = math::deriveStream(params.seed, 0x15A5u);

    for (core::u32 round = 0u; round < params.iterations; ++round)
    {
        lpl::pmr::string next;
        bool truncated = false;

        for (core::usize i = 0u; i < current.size(); ++i)
        {
            const char symbol = current[i];

            // Weighted choice among the rules for this symbol. One pass to total
            // the weights, one to pick — the alternative, collecting matches into a
            // vector first, would allocate once per symbol of a string that is
            // already the largest thing here.
            core::u32 total = 0u;
            for (core::usize r = 0u; r < params.rules.size(); ++r)
                if (params.rules[r].symbol == symbol)
                    total += params.rules[r].weight;

            const lpl::pmr::string *replacement = nullptr;
            if (total != 0u)
            {
                core::u32 roll = random.below(total);
                for (core::usize r = 0u; r < params.rules.size(); ++r)
                {
                    if (params.rules[r].symbol != symbol)
                        continue;
                    if (roll < params.rules[r].weight)
                    {
                        replacement = &params.rules[r].replacement;
                        break;
                    }
                    roll -= params.rules[r].weight;
                }
            }

            // Growth is exponential: a harmless-looking rule set reaches
            // megabytes in a few rounds. Stop at the cap rather than let a
            // kernel heap decide how this ends.
            if (next.size() + (replacement != nullptr ? replacement->size() : 1u) > params.maxLength)
            {
                truncated = true;
                break;
            }

            if (replacement != nullptr)
                next += *replacement;
            else
                next += symbol;
        }

        current = next;
        if (truncated)
            break;
    }
    return current;
}

HeadingField bakeHeadingField(core::u32 width, core::u32 depth, const lpl::pmr::vector<FieldRegion> &regions)
{
    HeadingField field{width, depth, 0u};
    if (field.empty() || regions.empty())
        return field;

    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            // Blend the influences as vectors, not as headings. Averaging heading
            // *numbers* is meaningless on a circle: bearing 15 and bearing 1 are
            // neighbours, and their mean is 8, which points the opposite way.
            math::Fixed32 sumX = math::Fixed32::zero();
            math::Fixed32 sumZ = math::Fixed32::zero();

            for (core::usize r = 0u; r < regions.size(); ++r)
            {
                const FieldRegion &region = regions[r];
                const core::i32 dx = static_cast<core::i32>(x) - static_cast<core::i32>(region.centerX);
                const core::i32 dz = static_cast<core::i32>(z) - static_cast<core::i32>(region.centerZ);

                // Chebyshev distance: no root needed, and the resulting square
                // falloff contours are invisible once several regions overlap.
                const core::u32 adx = static_cast<core::u32>(dx < 0 ? -dx : dx);
                const core::u32 adz = static_cast<core::u32>(dz < 0 ? -dz : dz);
                const core::u32 distance = adx > adz ? adx : adz;

                math::Fixed32 weight =
                    math::Fixed32::fromFloat(region.strength) -
                    math::Fixed32::fromFloat(region.falloff) * math::Fixed32::fromInt(static_cast<core::i32>(distance));
                if (weight.raw() <= 0)
                    continue;

                core::u32 bearing = region.bearing % kTurtleDirections;
                if (region.pattern == FieldPattern::Radial)
                {
                    // A radial field's lines run along the ray from the centre, so
                    // the bearing is the octant the offset falls in. Sixteen
                    // directions and a Chebyshev comparison decide it without any
                    // arctangent.
                    if (distance == 0u)
                        continue;
                    bearing = radialBearing(dx, dz);
                }

                sumX = sumX + math::Fixed32::fromRaw(kStepX[bearing]) * weight;
                sumZ = sumZ + math::Fixed32::fromRaw(kStepZ[bearing]) * weight;
            }

            field.at(x, z) = static_cast<core::u8>(nearestHeading(sumX, sumZ));
        }
    }
    return field;
}

core::u32 drawTurtle(const lpl::pmr::string &expanded, const TurtleParams &params, Grid<core::u8> &canvas)
{
    if (canvas.empty())
        return 0u;

    core::u32 drawn = 0u;
    const math::Fixed32 decay = math::Fixed32::fromFloat(params.stepDecay <= 0.0f ? 1.0f : params.stepDecay);

    TurtleState state;
    state.x = math::Fixed32::fromInt(static_cast<core::i32>(params.startX));
    state.z = math::Fixed32::fromInt(static_cast<core::i32>(params.startZ));
    state.direction = params.startDirection % kTurtleDirections;
    state.step = math::Fixed32::fromInt(static_cast<core::i32>(params.stepLength));

    lpl::pmr::vector<TurtleState> saved;

    for (core::usize i = 0u; i < expanded.size(); ++i)
    {
        switch (expanded[i])
        {
        case 'F':
        case 'f': {
            const math::Fixed32 stepX = math::Fixed32::fromRaw(kStepX[state.direction]) * state.step;
            const math::Fixed32 stepZ = math::Fixed32::fromRaw(kStepZ[state.direction]) * state.step;
            const math::Fixed32 nextX = state.x + stepX;
            const math::Fixed32 nextZ = state.z + stepZ;
            if (leavesCanvas(canvas, nextX, nextZ))
                break; // refused: stay put and let the next turn redirect us
            if (expanded[i] == 'F')
                drawSegment(canvas, state.x, state.z, nextX, nextZ, params.thickness, drawn);
            state.x = nextX;
            state.z = nextZ;
            break;
        }
        case '+': state.direction = (state.direction + params.turnAmount) % kTurtleDirections; break;
        case '-':
            // Add the complement rather than subtract: the heading is unsigned,
            // and wrapping through zero the other way would land far away.
            state.direction =
                (state.direction + kTurtleDirections - (params.turnAmount % kTurtleDirections)) % kTurtleDirections;
            break;
        case '[': {
            TurtleState branch = state;
            ++branch.depth;
            branch.step = state.step * decay; // branches taper
            saved.push_back(state);
            state = branch;
            break;
        }
        case ']':
            if (!saved.empty())
            {
                state = saved[saved.size() - 1u];
                saved.pop_back();
            }
            break;
        default: break; // inert: a rewrite-only variable
        }
    }
    return drawn;
}

core::u32 drawTurtleInField(const lpl::pmr::string &expanded, const TurtleParams &params, const HeadingField &field,
                            core::f32 conform, Grid<core::u8> &canvas)
{
    if (canvas.empty())
        return 0u;
    if (field.width() != canvas.width() || field.depth() != canvas.depth())
        return drawTurtle(expanded, params, canvas);

    core::u32 drawn = 0u;
    const math::Fixed32 decay = math::Fixed32::fromFloat(params.stepDecay <= 0.0f ? 1.0f : params.stepDecay);
    const math::Fixed32 conformity =
        math::Fixed32::fromFloat(conform < 0.0f ? 0.0f : (conform > 1.0f ? 1.0f : conform));

    TurtleState state;
    state.x = math::Fixed32::fromInt(static_cast<core::i32>(params.startX));
    state.z = math::Fixed32::fromInt(static_cast<core::i32>(params.startZ));
    state.direction = params.startDirection % kTurtleDirections;
    state.step = math::Fixed32::fromInt(static_cast<core::i32>(params.stepLength));

    lpl::pmr::vector<TurtleState> saved;
    bool justTurned = false;

    for (core::usize i = 0u; i < expanded.size(); ++i)
    {
        switch (expanded[i])
        {
        case 'F':
        case 'f': {
            // The grammar says which way to turn; the field says which way the
            // ground runs. Blend the two as vectors, weighted by conformity, so a
            // straight run follows the field lines while `+` and `-` still steer.
            //
            // "Still steer" has to be literal, and it was not: blending on the
            // step that FOLLOWS a turn pulls the new bearing straight back onto
            // the streamline, so at high conformity every branch out of a node
            // retraces the same line and the network stops growing. Measured, it
            // stopped dead — 16 cells at conformity 1 whatever the grammar did,
            // against 106 for the same grammar with the field off. So a turn gets
            // one unconformed step to establish itself, and the field takes over
            // from the step after.
            core::u32 heading = state.direction;
            if (justTurned)
                justTurned = false;
            else if (conformity.raw() > 0 && field.contains(state.x.toInt(), state.z.toInt()))
            {
                const core::u32 wanted =
                    field.at(static_cast<core::u32>(state.x.toInt()), static_cast<core::u32>(state.z.toInt()));
                const math::Fixed32 own = math::Fixed32::one() - conformity;
                heading = nearestHeading(math::Fixed32::fromRaw(kStepX[state.direction]) * own +
                                             math::Fixed32::fromRaw(kStepX[wanted]) * conformity,
                                         math::Fixed32::fromRaw(kStepZ[state.direction]) * own +
                                             math::Fixed32::fromRaw(kStepZ[wanted]) * conformity);
            }

            const math::Fixed32 stepX = math::Fixed32::fromRaw(kStepX[heading]) * state.step;
            const math::Fixed32 stepZ = math::Fixed32::fromRaw(kStepZ[heading]) * state.step;
            const math::Fixed32 nextX = state.x + stepX;
            const math::Fixed32 nextZ = state.z + stepZ;
            if (leavesCanvas(canvas, nextX, nextZ))
                break; // refused: stay put and let the next turn redirect us
            if (expanded[i] == 'F')
                drawSegment(canvas, state.x, state.z, nextX, nextZ, params.thickness, drawn);
            state.x = nextX;
            state.z = nextZ;
            // Adopt the blended heading: otherwise the field would be re-applied
            // to the original bearing at every step and the run would never curve.
            state.direction = heading;
            break;
        }
        case '+':
            state.direction = (state.direction + params.turnAmount) % kTurtleDirections;
            justTurned = true;
            break;
        case '-':
            state.direction =
                (state.direction + kTurtleDirections - (params.turnAmount % kTurtleDirections)) % kTurtleDirections;
            justTurned = true;
            break;
        case '[': {
            TurtleState branch = state;
            ++branch.depth;
            branch.step = state.step * decay;
            saved.push_back(state);
            state = branch;
            break;
        }
        case ']':
            if (!saved.empty())
            {
                state = saved[saved.size() - 1u];
                saved.pop_back();
            }
            break;
        default: break;
        }
    }
    return drawn;
}

LSystemParams makeBranchingGrammar()
{
    LSystemParams params;
    params.axiom = "F";
    // Three alternatives for the same symbol, so a stand of trees grown from one
    // grammar is a stand of different trees.
    params.rules.push_back(LRule{'F', lpl::pmr::string{"FF+[+F-F-F]-[-F+F+F]"}, 3u});
    params.rules.push_back(LRule{'F', lpl::pmr::string{"FF-[-F+F]+[+F-F+F]"}, 2u});
    params.rules.push_back(LRule{'F', lpl::pmr::string{"F[+F]F[-F]F"}, 2u});
    params.iterations = 3u;
    return params;
}

LSystemParams makeRoadGrammar()
{
    LSystemParams params;
    // A stem that keeps going and throws off perpendicular side streets, which
    // is what a road network looks like from above.
    params.axiom = "X";
    params.rules.push_back(LRule{'X', lpl::pmr::string{"F[+X]F[-X]FX"}, 3u});
    params.rules.push_back(LRule{'X', lpl::pmr::string{"F[+X]FX"}, 1u});
    params.rules.push_back(LRule{'X', lpl::pmr::string{"F[-X]FX"}, 1u});
    params.rules.push_back(LRule{'F', lpl::pmr::string{"FF"}});
    params.iterations = 4u;
    return params;
}

} // namespace lpl::procgen
