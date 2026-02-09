#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Optimizing::World {

/**
 * @brief 16-bit radix sort (4 passes) with preallocated scratch buffers.
 *
 * Sorts uint64_t keys and reorders corresponding indices.
 * Fewer passes than 8-bit variant, good cache locality for large arrays.
 * No internal allocations — caller provides scratch buffers (resized if needed).
 *
 * @param keys      Keys to sort (modified in-place)
 * @param indices   Corresponding indices (reordered to match keys)
 * @param tmpKeys   Scratch buffer for keys (resized internally if too small)
 * @param tmpIndices Scratch buffer for indices (resized internally if too small)
 */
void radix_sort_u64_indices_b16_scratch(std::vector<uint64_t> &keys, std::vector<int> &indices,
                                        std::vector<uint64_t> &tmpKeys, std::vector<int> &tmpIndices);

} // namespace Optimizing::World
