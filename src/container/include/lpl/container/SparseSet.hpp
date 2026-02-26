/**
 * @file SparseSet.hpp
 * @brief Generational sparse set with O(1) lookup and swap-and-pop removal.
 *
 * Maps public entity IDs to dense storage indices.  Each ID has a
 * generation counter to detect stale references.  The dense array is
 * always compact, enabling cache-friendly iteration.
 *
 * @tparam T Payload type stored in the dense array.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CONTAINER_SPARSE_SET_HPP
    #define LPL_CONTAINER_SPARSE_SET_HPP

    #include <lpl/core/Constants.hpp>
    #include <lpl/core/Types.hpp>

    #include <vector>
    #include <optional>

namespace lpl::container {

/**
 * @brief Generational sparse set.
 * @tparam T Dense-stored payload type.
 */
template <typename T>
class SparseSet final {
public:
    /**
     * @brief Construct a sparse set with a given sparse-array capacity.
     * @param sparseCapacity Maximum raw ID value supported.
     */
    explicit SparseSet(core::u32 sparseCapacity = core::kMaxEntityIdSpace);

    /**
     * @brief Insert an element associated with a raw ID.
     * @param id  Raw (slot) ID.
     * @param val Value to store.
     * @return True on success, false if id is out of range.
     */
    bool insert(core::u32 id, T val);

    /**
     * @brief Remove the element associated with a raw ID (swap-and-pop).
     * @param id Raw (slot) ID.
     * @return True if found and removed.
     */
    bool remove(core::u32 id);

    /**
     * @brief Lookup by raw ID.
     * @param id Raw (slot) ID.
     * @return Pointer to the dense value, or nullptr.
     */
    [[nodiscard]] T       *find(core::u32 id);
    [[nodiscard]] const T *find(core::u32 id) const;

    /**
     * @brief Check whether a raw ID is currently live.
     */
    [[nodiscard]] bool contains(core::u32 id) const;

    [[nodiscard]] core::u32    size()  const { return static_cast<core::u32>(_dense.size()); }
    [[nodiscard]] const auto & dense() const { return _dense; }

private:
    static constexpr core::u32 kInvalid = ~core::u32{0};

    std::vector<core::u32> _sparse;
    std::vector<T>         _dense;
    std::vector<core::u32> _denseToSparse;
};

} // namespace lpl::container

    #include "SparseSet.inl"

#endif // LPL_CONTAINER_SPARSE_SET_HPP
