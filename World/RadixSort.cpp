#include "RadixSort.hpp"
#include <array>
#include <cstring>

namespace Optimizing::World {

void radix_sort_u64_indices_b16_scratch(std::vector<uint64_t> &keys, std::vector<int> &indices,
                                        std::vector<uint64_t> &tmpKeys, std::vector<int> &tmpIndices)
{
    const size_t n = keys.size();
    if (n == 0)
        return;

    if (tmpKeys.size() < n)
        tmpKeys.resize(n);
    if (tmpIndices.size() < n)
        tmpIndices.resize(n);

    // Stack-allocated count array (65536 × 8 bytes = 512KB).
    // For hot paths this is acceptable; if stack size is a concern,
    // consider passing the count buffer as a parameter.
    constexpr size_t BUCKETS = 65536;
    std::array<size_t, BUCKETS> count;

    for (int pass = 0; pass < 4; ++pass)
    {
        const int shift = pass * 16;

        // Zero count array
        count.fill(0);

        // Histogram
        for (size_t i = 0; i < n; ++i)
        {
            uint16_t word = static_cast<uint16_t>((keys[i] >> shift) & 0xFFFF);
            ++count[word];
        }

        // Prefix sum
        size_t sum = 0;
        for (size_t i = 0; i < BUCKETS; ++i)
        {
            size_t c = count[i];
            count[i] = sum;
            sum += c;
        }

        // Scatter
        for (size_t i = 0; i < n; ++i)
        {
            uint16_t word = static_cast<uint16_t>((keys[i] >> shift) & 0xFFFF);
            size_t pos = count[word]++;
            tmpKeys[pos] = keys[i];
            tmpIndices[pos] = indices[i];
        }

        std::swap(keys, tmpKeys);
        std::swap(indices, tmpIndices);
    }
}

} // namespace Optimizing::World
