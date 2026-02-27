/**
 * @file SparseSet.inl
 * @brief Template implementation of the generational sparse set.
 * @see   SparseSet.hpp
 */

#ifndef LPL_CONTAINER_SPARSE_SET_INL
    #define LPL_CONTAINER_SPARSE_SET_INL

namespace lpl::container {

template <typename T>
SparseSet<T>::SparseSet(core::u32 sparseCapacity)
    : _sparse(sparseCapacity, kInvalid)
{
}

template <typename T>
bool SparseSet<T>::insert(core::u32 id, T val)
{
    if (id >= _sparse.size())
        return false;
    if (_sparse[id] != kInvalid)
        return false;

    _sparse[id] = static_cast<core::u32>(_dense.size());
    _dense.push_back(std::move(val));
    _denseToSparse.push_back(id);
    return true;
}

template <typename T>
bool SparseSet<T>::remove(core::u32 id)
{
    if (id >= _sparse.size())
        return false;

    core::u32 denseIdx = _sparse[id];
    if (denseIdx == kInvalid)
        return false;

    core::u32 lastDenseIdx = static_cast<core::u32>(_dense.size()) - 1;
    if (denseIdx != lastDenseIdx) {
        core::u32 lastSparseId = _denseToSparse[lastDenseIdx];
        _dense[denseIdx]         = std::move(_dense[lastDenseIdx]);
        _denseToSparse[denseIdx] = lastSparseId;
        _sparse[lastSparseId]    = denseIdx;
    }

    _dense.pop_back();
    _denseToSparse.pop_back();
    _sparse[id] = kInvalid;
    return true;
}

template <typename T>
T *SparseSet<T>::find(core::u32 id)
{
    if (id >= _sparse.size())
        return nullptr;

    core::u32 denseIdx = _sparse[id];
    if (denseIdx == kInvalid)
        return nullptr;

    return &_dense[denseIdx];
}

template <typename T>
const T *SparseSet<T>::find(core::u32 id) const
{
    return const_cast<SparseSet *>(this)->find(id);
}

template <typename T>
bool SparseSet<T>::contains(core::u32 id) const
{
    if (id >= _sparse.size())
        return false;
    return _sparse[id] != kInvalid;
}

} // namespace lpl::container

#endif // LPL_CONTAINER_SPARSE_SET_INL
