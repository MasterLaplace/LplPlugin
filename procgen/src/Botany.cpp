/**
 * @file Botany.cpp
 * @brief Implementation of the 3D turtle that grows a tree from a grammar.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Botany.hpp>

#include <lpl/math/Cordic.hpp>
#include <lpl/procgen/Random.hpp>

namespace lpl::procgen {

namespace {

/**
 * @struct Frame
 * @brief The turtle's orientation: three orthonormal axes in Q16.16.
 *
 * Carrying a frame rather than two angles is what makes roll mean anything. With
 * yaw and pitch alone, a branch's children all fan out in the same plane and the
 * tree is flat — the classic mistake, and it looks like a cardboard cut-out. Roll
 * turns that plane around the branch, so successive children leave in different
 * directions, which is what a real tree does and why the golden angle is the
 * default roll.
 */
struct Frame {
    math::Fixed32 hx{}, hy{}, hz{}; ///< Heading: where the turtle moves.
    math::Fixed32 lx{}, ly{}, lz{}; ///< Left.
    math::Fixed32 ux{}, uy{}, uz{}; ///< Up, relative to the heading.
};

struct TurtleState {
    math::Fixed32 x{}, y{}, z{};
    Frame frame{};
    math::Fixed32 length{};
    math::Fixed32 radius{};
    core::u8 depth{0u};
    bool leafed{false}; ///< Did anything inside this branch place a leaf?
};

/// Rotates @p a and @p b about the axis they span, by @p angle radians.
void rotatePair(math::Fixed32 &ax, math::Fixed32 &ay, math::Fixed32 &az, math::Fixed32 &bx, math::Fixed32 &by,
                math::Fixed32 &bz, math::Fixed32 angle)
{
    math::Fixed32 s{};
    math::Fixed32 c{};
    math::Cordic::sincos(angle, s, c);
    const math::Fixed32 nax = ax * c + bx * s;
    const math::Fixed32 nay = ay * c + by * s;
    const math::Fixed32 naz = az * c + bz * s;
    const math::Fixed32 nbx = bx * c - ax * s;
    const math::Fixed32 nby = by * c - ay * s;
    const math::Fixed32 nbz = bz * c - az * s;
    ax = nax;
    ay = nay;
    az = naz;
    bx = nbx;
    by = nby;
    bz = nbz;
}

/// Yaw: heading turns towards left.
void yaw(Frame &f, math::Fixed32 angle)
{
    rotatePair(f.hx, f.hy, f.hz, f.lx, f.ly, f.lz, angle);
}

/// Pitch: heading turns towards up.
void pitch(Frame &f, math::Fixed32 angle)
{
    rotatePair(f.hx, f.hy, f.hz, f.ux, f.uy, f.uz, angle);
}

/// Roll: left and up turn about the heading, which the heading does not feel.
void roll(Frame &f, math::Fixed32 angle)
{
    rotatePair(f.lx, f.ly, f.lz, f.ux, f.uy, f.uz, angle);
}

[[nodiscard]] math::Fixed32 absOf(math::Fixed32 value)
{
    return value.raw() < 0 ? math::Fixed32{} - value : value;
}

} // namespace

LSystemParams makeTreeGrammar(TreeSpecies species, core::u32 seed)
{
    LSystemParams params;
    params.seed = seed;
    params.maxLength = 4096u;

    switch (species)
    {
    case TreeSpecies::Conifer:
        // A dominant stem: the rule always continues with F, and the whorls it
        // throws off are short and pitched down. That, not a width factor, is
        // what makes the silhouette a cone.
        params.axiom = "A";
        params.rules.push_back(LRule{'A', lpl::pmr::string{"F[&/A L][&\\\\A L][&+A L]FA"}, 3u});
        params.rules.push_back(LRule{'A', lpl::pmr::string{"F[&\\A L][&/A L]FA"}, 2u});
        params.iterations = 4u;
        break;

    case TreeSpecies::Broadleaf:
        // No dominant stem past the first fork: A becomes two A's, so the crown
        // opens out and the trunk stops being the tallest thing in the tree.
        params.axiom = "FA";
        params.rules.push_back(LRule{'A', lpl::pmr::string{"F[+/A L][-\\A L]"}, 3u});
        params.rules.push_back(LRule{'A', lpl::pmr::string{"F[&/A L][^\\A L][+A L]"}, 2u});
        params.rules.push_back(LRule{'A', lpl::pmr::string{"FF[+A L][-A L]"}, 2u});
        params.iterations = 4u;
        break;

    case TreeSpecies::Shrub:
    case TreeSpecies::Count:
        params.axiom = "A";
        params.rules.push_back(LRule{'A', lpl::pmr::string{"[+FA L][-FA L][&FA L]"}, 2u});
        params.rules.push_back(LRule{'A', lpl::pmr::string{"[/FA L][\\FA L]"}, 1u});
        params.iterations = 3u;
        break;
    }
    return params;
}

TreeSkeleton growTree(const TreeParams &params)
{
    TreeSkeleton skeleton;
    const lpl::pmr::string expanded = expandLSystem(makeTreeGrammar(params.species, params.seed));

    // Growth wobbles per tree, so a stand grown from one grammar is a stand of
    // different trees rather than one tree stamped many times.
    Random jitter{params.seed ^ 0x5B7Eu};

    TurtleState state;
    state.frame.hy = math::Fixed32::one(); // straight up
    state.frame.lx = math::Fixed32::one();
    state.frame.uz = math::Fixed32::one();
    state.length = params.segmentLength;
    state.radius = params.radius;

    lpl::pmr::vector<TurtleState> stack;

    for (core::u32 i = 0u; i < expanded.size(); ++i)
    {
        if (skeleton.branches.size() >= params.maxSegments)
            break;

        switch (expanded[i])
        {
        case 'F':
        {
            // A wobble of up to a twentieth of a radian, per segment. Perfectly
            // straight segments read as a diagram of a tree, not as a tree.
            const math::Fixed32 wobble =
                math::Fixed32::fromRaw(static_cast<core::i32>(jitter.below(6554u)) - 3277);
            yaw(state.frame, wobble);
            pitch(state.frame, math::Fixed32::fromRaw(static_cast<core::i32>(jitter.below(6554u)) - 3277));

            TreeBranch branch;
            branch.x0 = state.x;
            branch.y0 = state.y;
            branch.z0 = state.z;
            branch.x1 = state.x + state.frame.hx * state.length;
            branch.y1 = state.y + state.frame.hy * state.length;
            branch.z1 = state.z + state.frame.hz * state.length;
            branch.radius0 = state.radius;
            branch.radius1 = state.radius * params.radiusDecay;
            branch.depth = state.depth;
            skeleton.branches.push_back(branch);

            state.x = branch.x1;
            state.y = branch.y1;
            state.z = branch.z1;
            state.radius = branch.radius1;

            if (state.y > skeleton.height)
                skeleton.height = state.y;
            const math::Fixed32 reach = absOf(state.x) > absOf(state.z) ? absOf(state.x) : absOf(state.z);
            if (reach > skeleton.spread)
                skeleton.spread = reach;
            break;
        }

        case 'L':
        {
            TreeLeaf leaf;
            leaf.x = state.x;
            leaf.y = state.y;
            leaf.z = state.z;
            // Leaves nearer the tips are smaller, which is what gives a crown an
            // edge instead of a wall.
            leaf.size = params.leafSize * (math::Fixed32::one() - math::Fixed32::fromFloat(0.12f) *
                                                                     math::Fixed32::fromInt(state.depth));
            leaf.depth = state.depth;
            if (leaf.size.raw() > 0)
                skeleton.leaves.push_back(leaf);
            state.leafed = true;
            if (!stack.empty())
                stack[stack.size() - 1u].leafed = true;
            break;
        }

        case '+': yaw(state.frame, params.branchAngle); break;
        case '-': yaw(state.frame, math::Fixed32{} - params.branchAngle); break;
        case '&': pitch(state.frame, math::Fixed32{} - params.branchAngle); break;
        case '^': pitch(state.frame, params.branchAngle); break;
        case '\\': roll(state.frame, params.rollAngle); break;
        case '/': roll(state.frame, math::Fixed32{} - params.rollAngle); break;

        case '[':
            stack.push_back(state);
            state.depth = static_cast<core::u8>(state.depth + 1u);
            state.length = state.length * params.lengthDecay;
            state.leafed = false;
            break;

        case ']':
        {
            // The tip of an unleafed branch gets foliage anyway: a grammar that
            // forgot to say "leaf" would grow bare sticks, and a bare stick is
            // not a shape any tree has.
            if (!state.leafed && state.depth != 0u)
            {
                TreeLeaf leaf;
                leaf.x = state.x;
                leaf.y = state.y;
                leaf.z = state.z;
                leaf.size = params.leafSize * math::Fixed32::fromFloat(0.8f);
                leaf.depth = state.depth;
                skeleton.leaves.push_back(leaf);
            }
            if (!stack.empty())
            {
                state = stack[stack.size() - 1u];
                stack.pop_back();
            }
            break;
        }

        default: break; // inert: a rewrite-only variable
        }
    }

    return skeleton;
}

core::u32 foldTreeSkeleton(const TreeSkeleton &skeleton)
{
    core::u32 hash = 0x811C9DC5u;
    const auto fold = [&hash](core::i32 raw) {
        const core::u32 value = static_cast<core::u32>(raw);
        for (core::u32 byte = 0u; byte < 4u; ++byte)
        {
            hash ^= (value >> (byte * 8u)) & 0xFFu;
            hash *= 0x01000193u;
        }
    };

    fold(static_cast<core::i32>(skeleton.branches.size()));
    fold(static_cast<core::i32>(skeleton.leaves.size()));
    for (core::u32 i = 0u; i < skeleton.branches.size(); ++i)
    {
        const TreeBranch &b = skeleton.branches[i];
        fold(b.x0.raw());
        fold(b.y0.raw());
        fold(b.z0.raw());
        fold(b.x1.raw());
        fold(b.y1.raw());
        fold(b.z1.raw());
        fold(b.radius0.raw());
        fold(b.radius1.raw());
        fold(static_cast<core::i32>(b.depth));
    }
    for (core::u32 i = 0u; i < skeleton.leaves.size(); ++i)
    {
        const TreeLeaf &l = skeleton.leaves[i];
        fold(l.x.raw());
        fold(l.y.raw());
        fold(l.z.raw());
        fold(l.size.raw());
    }
    fold(skeleton.height.raw());
    fold(skeleton.spread.raw());
    return hash;
}

TreeParams parityTreeParams(TreeSpecies species)
{
    TreeParams params;
    params.species = species;
    params.seed = 0x7A3Eu + static_cast<core::u32>(species) * 0x9E37u;
    switch (species)
    {
    case TreeSpecies::Conifer:
        params.segmentLength = math::Fixed32::fromFloat(0.85f);
        params.branchAngle = math::Fixed32::fromFloat(0.62f);
        params.radius = math::Fixed32::fromFloat(0.11f);
        params.leafSize = math::Fixed32::fromFloat(0.26f);
        break;
    case TreeSpecies::Broadleaf:
        params.segmentLength = math::Fixed32::fromFloat(0.95f);
        params.branchAngle = math::Fixed32::fromFloat(0.44f);
        params.radius = math::Fixed32::fromFloat(0.16f);
        params.leafSize = math::Fixed32::fromFloat(0.32f);
        break;
    case TreeSpecies::Shrub:
    case TreeSpecies::Count:
        params.segmentLength = math::Fixed32::fromFloat(0.5f);
        params.branchAngle = math::Fixed32::fromFloat(0.5f);
        params.radius = math::Fixed32::fromFloat(0.07f);
        params.leafSize = math::Fixed32::fromFloat(0.18f);
        params.iterations = 3u;
        break;
    }
    return params;
}

} // namespace lpl::procgen
