// /////////////////////////////////////////////////////////////////////////////
/// @file Octree.hpp
/// @brief Morton-sorted linear octree for broad-phase collision.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/physics/ISpatialIndex.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::physics {

// /////////////////////////////////////////////////////////////////////////////
/// @class Octree
/// @brief Flat, Morton-sorted radix octree. Objects are sorted by their
///        Morton code each rebuild pass, enabling cache-friendly traversal
///        and O(n log n) broad-phase.
// /////////////////////////////////////////////////////////////////////////////
class Octree final : public ISpatialIndex,
                     public core::NonCopyable<Octree>
{
public:
    /// @brief Constructs an octree covering the given world-space bounds.
    /// @param worldBounds Total world AABB.
    explicit Octree(const math::AABB<math::Fixed32>& worldBounds);
    ~Octree() override;

    void insert(core::u32 objectId,
                const math::AABB<math::Fixed32>& aabb) override;

    void update(core::u32 objectId,
                const math::AABB<math::Fixed32>& aabb) override;

    void remove(core::u32 objectId) override;

    void query(const math::AABB<math::Fixed32>& region,
               const std::function<void(core::u32)>& callback) const override;

    void rebuild() override;

    [[nodiscard]] core::u32 count() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::physics
