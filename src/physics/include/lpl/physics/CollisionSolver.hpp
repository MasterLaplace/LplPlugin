// /////////////////////////////////////////////////////////////////////////////
/// @file CollisionSolver.hpp
/// @brief Iterative impulse-based collision solver.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/physics/CollisionDetector.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>

#include <span>

namespace lpl::physics {

// /////////////////////////////////////////////////////////////////////////////
/// @struct BodyState
/// @brief Minimal state required by the solver for impulse resolution.
// /////////////////////////////////////////////////////////////////////////////
struct BodyState
{
    math::Vec3<math::Fixed32> position;
    math::Vec3<math::Fixed32> velocity;
    math::Fixed32             inverseMass;
    math::Fixed32             restitution;
};

// /////////////////////////////////////////////////////////////////////////////
/// @struct CollisionPair
/// @brief A pair of body indices + the contact between them.
// /////////////////////////////////////////////////////////////////////////////
struct CollisionPair
{
    core::u32       indexA;
    core::u32       indexB;
    ContactPoint    contact;
};

// /////////////////////////////////////////////////////////////////////////////
/// @class CollisionSolver
/// @brief Sequential impulse solver (Erin Catto style).
///
/// Runs @c kIterations passes over all collision pairs, applying impulses
/// that respect restitution coefficients. Fully deterministic (Fixed32).
// /////////////////////////////////////////////////////////////////////////////
class CollisionSolver
{
public:
    static constexpr core::u32 kIterations = 4;

    /// @brief Resolves all collision pairs over the body state array.
    /// @param bodies  Mutable span of body states.
    /// @param pairs   Collision pairs to resolve.
    static void solve(std::span<BodyState> bodies,
                      std::span<const CollisionPair> pairs) noexcept;
};

} // namespace lpl::physics
