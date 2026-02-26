// /////////////////////////////////////////////////////////////////////////////
/// @file SpatialGrid.hpp
/// @brief Uniform spatial hash grid for broad-phase collision.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/physics/ISpatialIndex.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::physics {

// /////////////////////////////////////////////////////////////////////////////
/// @class SpatialGrid
/// @brief Fixed-cell-size hash grid. Best for uniformly distributed
///        objects of similar size. O(1) insert / remove, O(k) query.
// /////////////////////////////////////////////////////////////////////////////
class SpatialGrid final : public ISpatialIndex,
                          public core::NonCopyable<SpatialGrid>
{
public:
    /// @brief Constructs a grid with the given cell size.
    /// @param cellSize Side length of each cubic cell.
    explicit SpatialGrid(math::Fixed32 cellSize);
    ~SpatialGrid() override;

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
