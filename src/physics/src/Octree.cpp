// /////////////////////////////////////////////////////////////////////////////
/// @file Octree.cpp
/// @brief Morton-sorted linear octree implementation stub.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/physics/Octree.hpp>
#include <lpl/math/Morton.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace lpl::physics {

struct Octree::Impl
{
    struct Entry
    {
        core::u32                   objectId;
        core::u64                   morton;
        math::AABB<math::Fixed32>  aabb;
    };

    math::AABB<math::Fixed32>                worldBounds;
    std::vector<Entry>                        entries;
    std::unordered_map<core::u32, core::u32>  idToIndex;
    bool                                      dirty{false};

    explicit Impl(const math::AABB<math::Fixed32>& wb)
        : worldBounds{wb}
    {}

    [[nodiscard]] core::u64 computeMorton(const math::AABB<math::Fixed32>& aabb) const
    {
        const auto c = aabb.center();
        const auto toGrid = [](math::Fixed32 v) -> core::i32 {
            return v.toInt() + static_cast<core::i32>(core::kMortonBias);
        };
        return math::morton::encode3D(toGrid(c.x), toGrid(c.y), toGrid(c.z));
    }
};

Octree::Octree(const math::AABB<math::Fixed32>& worldBounds)
    : impl_{std::make_unique<Impl>(worldBounds)}
{}

Octree::~Octree() = default;

void Octree::insert(core::u32 objectId, const math::AABB<math::Fixed32>& aabb)
{
    const core::u32 idx = static_cast<core::u32>(impl_->entries.size());
    impl_->entries.push_back({objectId, impl_->computeMorton(aabb), aabb});
    impl_->idToIndex[objectId] = idx;
    impl_->dirty = true;
}

void Octree::update(core::u32 objectId, const math::AABB<math::Fixed32>& aabb)
{
    auto it = impl_->idToIndex.find(objectId);
    LPL_ASSERT(it != impl_->idToIndex.end());
    auto& entry = impl_->entries[it->second];
    entry.aabb = aabb;
    entry.morton = impl_->computeMorton(aabb);
    impl_->dirty = true;
}

void Octree::remove(core::u32 objectId)
{
    auto it = impl_->idToIndex.find(objectId);
    LPL_ASSERT(it != impl_->idToIndex.end());

    const core::u32 idx = it->second;
    const core::u32 last = static_cast<core::u32>(impl_->entries.size()) - 1;

    if (idx != last)
    {
        impl_->entries[idx] = impl_->entries[last];
        impl_->idToIndex[impl_->entries[idx].objectId] = idx;
    }
    impl_->entries.pop_back();
    impl_->idToIndex.erase(it);
    impl_->dirty = true;
}

void Octree::query(const math::AABB<math::Fixed32>& region,
                   const std::function<void(core::u32)>& callback) const
{
    for (const auto& entry : impl_->entries)
    {
        if (entry.aabb.intersects(region))
        {
            callback(entry.objectId);
        }
    }
}

void Octree::rebuild()
{
    if (!impl_->dirty)
    {
        return;
    }

    std::sort(impl_->entries.begin(), impl_->entries.end(),
              [](const Impl::Entry& a, const Impl::Entry& b) {
                  return a.morton < b.morton;
              });

    for (core::u32 i = 0; i < static_cast<core::u32>(impl_->entries.size()); ++i)
    {
        impl_->idToIndex[impl_->entries[i].objectId] = i;
    }

    impl_->dirty = false;
}

core::u32 Octree::count() const noexcept
{
    return static_cast<core::u32>(impl_->entries.size());
}

} // namespace lpl::physics
