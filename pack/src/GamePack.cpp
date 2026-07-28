/**
 * @file GamePack.cpp
 * @brief Implementation of the bounds-checked game package reader.
 *
 * Freestanding by construction: no allocation, no libc string handling, no
 * exceptions. Everything is arithmetic on offsets that is checked against the
 * image length before any dereference — a cartridge is untrusted input, and the
 * kernel is the target least able to survive a read past the end.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/pack/GamePack.hpp>

#include <lpl/std/cstring.hpp>

namespace lpl::pack {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

constexpr char kMagic[kMagicSize] = {'L', 'P', 'L', 'P', 'A', 'K', '\0', '\0'};

/// True when [offset, offset+length) fits inside [0, size), overflow included.
constexpr bool fitsWithin(core::u32 offset, core::u32 length, core::u32 size) noexcept
{
    if (offset > size)
        return false;
    // Compare against the remaining room rather than computing offset + length,
    // which could wrap on a hostile or corrupt image.
    return length <= (size - offset);
}

} // namespace

core::u32 hashBytes(const core::u8 *bytes, core::u32 size) noexcept
{
    core::u32 hash = kFnv1aOffsetBasis;
    if (bytes == nullptr)
        return hash;
    for (core::u32 i = 0u; i < size; ++i)
        hash = (hash ^ bytes[i]) * kFnv1aPrime;
    return hash;
}

bool View::open(const core::u8 *bytes, core::u32 size) noexcept
{
    _bytes = nullptr;
    _size = 0u;

    if (bytes == nullptr || size < sizeof(Header))
        return false;

    Header header{};
    lpl::pmr::memcpy(&header, bytes, sizeof(Header));

    for (core::u32 i = 0u; i < kMagicSize; ++i)
        if (header.magic[i] != kMagic[i])
            return false;

    if (header.formatVersion != kFormatVersion)
        return false;

    // The image must be exactly as long as it claims: a short buffer means a
    // truncated transfer, a long one means we were handed something else.
    if (header.totalSize < sizeof(Header) || header.totalSize > size)
        return false;

    // The section table must fit before any section does.
    const core::u32 tableBytes = header.sectionCount * static_cast<core::u32>(sizeof(SectionEntry));
    if (header.sectionCount != 0u && tableBytes / header.sectionCount != sizeof(SectionEntry))
        return false; // multiplication overflowed
    if (!fitsWithin(static_cast<core::u32>(sizeof(Header)), tableBytes, header.totalSize))
        return false;

    const core::u32 contentSize = header.totalSize - static_cast<core::u32>(sizeof(Header));
    if (hashBytes(bytes + sizeof(Header), contentSize) != header.contentHash)
        return false;

    _bytes = bytes;
    _size = header.totalSize;
    return true;
}

core::u32 View::sectionCount() const noexcept
{
    if (_bytes == nullptr)
        return 0u;
    Header header{};
    lpl::pmr::memcpy(&header, _bytes, sizeof(Header));
    return header.sectionCount;
}

bool View::findSection(SectionType type, const core::u8 *&outBytes, core::u32 &outSize) const noexcept
{
    if (_bytes == nullptr)
        return false;

    const core::u32 count = sectionCount();
    for (core::u32 i = 0u; i < count; ++i)
    {
        SectionEntry entry{};
        lpl::pmr::memcpy(&entry, _bytes + sizeof(Header) + i * sizeof(SectionEntry), sizeof(SectionEntry));

        if (entry.type != static_cast<core::u32>(type))
            continue; // includes section types this build does not know about
        if (!fitsWithin(entry.offset, entry.size, _size))
            return false;

        outBytes = _bytes + entry.offset;
        outSize = entry.size;
        return true;
    }
    return false;
}

bool View::readRecipe(RecipeV1 &outRecipe) const noexcept
{
    const core::u8 *payload = nullptr;
    core::u32 payloadSize = 0u;
    if (!findSection(SectionType::WorldRecipe, payload, payloadSize))
        return false;
    if (payloadSize != sizeof(RecipeV1))
        return false;

    lpl::pmr::memcpy(&outRecipe, payload, sizeof(RecipeV1));
    return true;
}

bool View::readLiving(LivingV1 &outLiving) const noexcept
{
    const core::u8 *payload = nullptr;
    core::u32 payloadSize = 0u;
    if (!findSection(SectionType::LivingRecipe, payload, payloadSize))
        return false;
    if (payloadSize != sizeof(LivingV1))
        return false;

    lpl::pmr::memcpy(&outLiving, payload, sizeof(LivingV1));
    return true;
}

} // namespace lpl::pack
