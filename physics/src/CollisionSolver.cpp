/**
 * @file CollisionSolver.cpp
 * @brief Iterative impulse collision solver implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/physics/CollisionSolver.hpp>

namespace lpl::physics {

void CollisionSolver::solve(std::span<BodyState> bodies,
                            std::span<const CollisionPair> pairs) noexcept
{
    for (core::u32 iter = 0; iter < kIterations; ++iter)
    {
        for (const auto& pair : pairs)
        {
            auto& a = bodies[pair.indexA];
            auto& b = bodies[pair.indexB];

            const auto& n = pair.contact.normal;

            const auto relVel = b.velocity - a.velocity;
            const auto velAlongNormal = relVel.dot(n);

            if (velAlongNormal > math::Fixed32{0})
            {
                continue;
            }

            const auto e = std::min(a.restitution, b.restitution);
            const auto invMassSum = a.inverseMass + b.inverseMass;

            if (invMassSum == math::Fixed32{0})
            {
                continue;
            }

            const auto j = -(math::Fixed32{1} + e) * velAlongNormal / invMassSum;

            const auto impulse = n * j;

            a.velocity = a.velocity - impulse * a.inverseMass;
            b.velocity = b.velocity + impulse * b.inverseMass;
        }
    }
}

} // namespace lpl::physics
