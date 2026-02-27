/**
 * @file ArenaAllocator.cpp
 * @brief Implementation of the bump-pointer arena allocator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#include "lpl/memory/ArenaAllocator.hpp"

#include <cstdlib>
#include <cstdint>

namespace lpl::memory {

ArenaAllocator::ArenaAllocator(core::usize capacity)
    : _capacity(capacity)
{
    _memory = static_cast<char *>(std::aligned_alloc(alignof(std::max_align_t), capacity));
}

ArenaAllocator::~ArenaAllocator()
{
    std::free(_memory);
}

void *ArenaAllocator::allocate(core::usize size, core::usize alignment)
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

void ArenaAllocator::deallocate([[maybe_unused]] void *ptr) {}

void ArenaAllocator::reset() { _offset = 0; }

bool ArenaAllocator::ownsPtr(const void *ptr) const
{
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto base = reinterpret_cast<std::uintptr_t>(_memory);
    return addr >= base && addr < base + _capacity;
}

} // namespace lpl::memory
