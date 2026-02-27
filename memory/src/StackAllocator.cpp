/**
 * @file StackAllocator.cpp
 * @brief Implementation of the LIFO stack allocator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#include "lpl/memory/StackAllocator.hpp"

#include <cstdlib>
#include <cstdint>

namespace lpl::memory {

StackAllocator::StackAllocator(core::usize capacity)
    : _capacity(capacity)
{
    _memory = static_cast<char *>(std::aligned_alloc(alignof(std::max_align_t), capacity));
}

StackAllocator::~StackAllocator()
{
    std::free(_memory);
}

void *StackAllocator::allocate(core::usize size, core::usize alignment)
{
    auto current = reinterpret_cast<std::uintptr_t>(_memory + _offset);
    core::usize padding = (alignment - (current % alignment)) % alignment;

    if (_offset + padding + size > _capacity)
        return nullptr;

    _offset += padding;
    void *ptr = _memory + _offset;
    _offset += size;
    return ptr;
}

void StackAllocator::deallocate([[maybe_unused]] void *ptr) {}

void StackAllocator::reset() { _offset = 0; }

void StackAllocator::freeToMarker(Marker marker)
{
    if (marker <= _offset)
        _offset = marker;
}

bool StackAllocator::ownsPtr(const void *ptr) const
{
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto base = reinterpret_cast<std::uintptr_t>(_memory);
    return addr >= base && addr < base + _capacity;
}

} // namespace lpl::memory
