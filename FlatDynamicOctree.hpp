#pragma once

#include "Math.hpp"
#include "Morton.hpp"
#include <algorithm>

class FlatDynamicOctree {
private:
    struct FlatNode {
        BoundaryBox bound;
        int firstChild = -1;
        uint32_t entityStart = 0;
        uint32_t entityCount = 0;
        uint8_t childCount = 0;
    };

    struct EntityRef {
        BoundaryBox bbox;
        uint64_t mortonKey;
        uint32_t index;
    };

public:
    FlatDynamicOctree(const BoundaryBox &worldBounds, uint8_t maxDepth = 8u, uint32_t leafCapacity = 32u)
        : _WORLD_BOUND(worldBounds), _WORLD_BOUND_SIZE(_WORLD_BOUND.max - _WORLD_BOUND.min), _MAX_DEPTH(maxDepth), _LEAF_CAPACITY(leafCapacity) {};
    ~FlatDynamicOctree() = default;

    FlatDynamicOctree(FlatDynamicOctree&&) = default;
    FlatDynamicOctree& operator=(FlatDynamicOctree&&) = default;

    FlatDynamicOctree(const FlatDynamicOctree&) = delete;
    FlatDynamicOctree& operator=(const FlatDynamicOctree&) = delete;

    template <typename GetBoundCallBack>
    void rebuild(const uint32_t count, GetBoundCallBack &&func)
    {
        _nodes.clear();
        _sortedRefs.clear();

        for (uint32_t index = 0u; index < count; ++index)
        {
            BoundaryBox bound = func(index);
            Vec3 center = bound.min + (bound.max - bound.min) * 0.5f;
            Vec3 normalized = (center - _WORLD_BOUND.min) / _WORLD_BOUND_SIZE;
            constexpr float maxVal = 2097151.0f;
            uint32_t x = static_cast<uint32_t>(std::clamp(normalized.x, 0.0f, 1.0f) * maxVal);
            uint32_t y = static_cast<uint32_t>(std::clamp(normalized.y, 0.0f, 1.0f) * maxVal);
            uint32_t z = static_cast<uint32_t>(std::clamp(normalized.z, 0.0f, 1.0f) * maxVal);

            _sortedRefs.push_back({bound, Morton::encode3D(x, y, z), index});
        }

        runRadixSort(count);

        _nodes.emplace_back();
        FlatNode &root = _nodes.back();
        root.bound = _WORLD_BOUND;
        root.entityStart = 0u;
        root.entityCount = count;
        recurseBuild(0, 0, count, 0, 0);
    }

    template <typename QueryCallback>
    void query(const BoundaryBox &searchArea, QueryCallback &&func) const
    {
        if (_nodes.empty())
            return;
        queryRecurse(0, searchArea, func);
    }

private:
    void runRadixSort(const uint32_t count)
    {
        if (_tempRefs.size() < _sortedRefs.size())
            _tempRefs.resize(_sortedRefs.size());

        EntityRef *src = _sortedRefs.data();
        EntityRef *dst = _tempRefs.data();

        for (uint8_t pass = 0u; pass < 4u; ++pass)
        {
            uint32_t counts[65536u] = {0u};
            uint64_t shift = pass * 16u;

            for (uint32_t index = 0u; index < count; ++index)
            {
                uint16_t bucket = (src[index].mortonKey >> shift) & 0xFFFF;
                ++counts[bucket];
            }

            uint32_t offsets[65536u];
            offsets[0u] = 0u;
            for (uint32_t index = 1u; index < 65536u; ++index)
                offsets[index] = offsets[index - 1u] + counts[index -1u];

            for (uint32_t index = 0u; index < count; ++index)
            {
                uint16_t bucket = (src[index].mortonKey >> shift) & 0xFFFF;
                dst[offsets[bucket]++] = src[index];
            }

            std::swap(src, dst);
        }
    }

    void recurseBuild(int nodeIdx, uint32_t start, uint32_t end, uint8_t depth, uint64_t code)
    {
        uint32_t count = end - start;

        if (count <= _LEAF_CAPACITY || depth >= _MAX_DEPTH)
        {
            FlatNode &node = _nodes[nodeIdx];
            node.entityStart = start;
            node.entityCount = count;
            return;
        }

        std::size_t firstChildIdx = _nodes.size();
        _nodes.resize(firstChildIdx + 8u);
        FlatNode &node = _nodes[nodeIdx];
        node.firstChild = firstChildIdx;

        Vec3 min = node.bound.min;
        Vec3 max = node.bound.max;
        Vec3 center = min + (max - min) * 0.5f;;

        uint64_t shift = (20u - depth) * 3u;
        uint32_t current = start;
        for (uint8_t index = 0u; index < 8u; ++index)
        {
            Vec3 childMin, childMax;

            // Axe X (Bit 0) : Si 0 -> Gauche (Min à Center), Si 1 -> Droite (Center à Max)
            if (index & 1) { childMin.x = center.x; childMax.x = max.x; }
            else           { childMin.x = min.x;    childMax.x = center.x; }

            // Axe Y (Bit 1) : Si 0 -> Bas, Si 1 -> Haut
            if (index & 2) { childMin.y = center.y; childMax.y = max.y; }
            else           { childMin.y = min.y;    childMax.y = center.y; }

            // Axe Z (Bit 2) : Si 0 -> Devant, Si 1 -> Fond
            if (index & 4) { childMin.z = center.z; childMax.z = max.z; }
            else           { childMin.z = min.z;    childMax.z = center.z; }

            _nodes[firstChildIdx + index].bound = {childMin, childMax};

            uint32_t childStart = current;
            for (; current < end; ++current)
            {
                uint8_t octant = (_sortedRefs[current].mortonKey >> shift) & 7u;
                if (octant != index)
                    break;
            }
            recurseBuild(firstChildIdx + index, childStart, current, depth + 1u, 0u);
        }
    }

    template <typename QueryCallback>
    void queryRecurse(int nodeIdx, const BoundaryBox &searchArea, QueryCallback &&func) const
    {
        const FlatNode &node = _nodes[nodeIdx];

        if (searchArea.max.x < node.bound.min.x || searchArea.min.x > node.bound.max.x ||
            searchArea.max.y < node.bound.min.y || searchArea.min.y > node.bound.max.y ||
            searchArea.max.z < node.bound.min.z || searchArea.min.z > node.bound.max.z)
        {
            return;
        }

        for (uint32_t index = 0u; index < node.entityCount; ++index)
            func(_sortedRefs[node.entityStart + index].index);

        if (node.firstChild != -1)
        {
            for (int index = 0; index < 8; ++index)
                queryRecurse(node.firstChild + index, searchArea, func);
        }
    }

private:
    std::vector<FlatNode> _nodes;
    std::vector<EntityRef> _sortedRefs;
    std::vector<EntityRef> _tempRefs;
    BoundaryBox _WORLD_BOUND;
    Vec3 _WORLD_BOUND_SIZE;
    uint8_t _MAX_DEPTH;
    uint32_t _LEAF_CAPACITY;
};
