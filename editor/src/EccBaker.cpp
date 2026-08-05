/**
 * @file EccBaker.cpp
 * @brief Computes the transversal parity and rewrites the header around it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/editor/EccBaker.hpp>

#include <lpl/codec/ReedSolomon.hpp>
#include <lpl/pack/EccSection.hpp>
#include <lpl/pack/GamePack.hpp>

#include <cstring>

namespace lpl::editor {

namespace {

/**
 * @brief Appends a POD to a byte vector.
 * @param out   Destination.
 * @param value What to append.
 */
template <typename T> void appendPod(std::vector<core::u8> &out, const T &value)
{
    const auto *raw = reinterpret_cast<const core::u8 *>(&value);
    out.insert(out.end(), raw, raw + sizeof(T));
}

} // namespace

std::vector<core::u8> attachEcc(const std::vector<core::u8> &image, const EccPolicy &policy)
{
    if (image.size() < sizeof(pack::Header))
        return image;

    pack::Header header{};
    std::memcpy(&header, image.data(), sizeof(pack::Header));
    if (header.totalSize != image.size())
        return image;

    const core::u32 dataShards = policy.dataShards == 0u ? 1u : policy.dataShards;
    const core::u32 parityShards =
        policy.parityShards == 0u ?
            1u :
            (policy.parityShards > codec::kMaxParitySymbols ? codec::kMaxParitySymbols : policy.parityShards);
    if (dataShards + parityShards > codec::kMaxCodewordSymbols)
        return image;

    // The span to protect runs from the end of the header to the end of the image as
    // it stands, plus the entry this call is about to insert into the table. That
    // entry shifts every payload by exactly one entry's worth, which is why the whole
    // image is rebuilt rather than patched.
    const core::u32 entryBytes = static_cast<core::u32>(sizeof(pack::SectionEntry));
    const core::u32 oldTableBytes = header.sectionCount * entryBytes;
    const core::u32 payloadBytes = header.totalSize - static_cast<core::u32>(sizeof(pack::Header)) - oldTableBytes;

    const core::u32 newSectionCount = header.sectionCount + 1u;
    const core::u32 newTableBytes = newSectionCount * entryBytes;
    const core::u32 protectedOffset = static_cast<core::u32>(sizeof(pack::Header));
    const core::u32 protectedBytes = newTableBytes + payloadBytes;
    const core::u32 rowBytes = (protectedBytes + dataShards - 1u) / dataShards;
    const core::u32 eccOffset = protectedOffset + protectedBytes;

    pack::EccV1 ecc{};
    ecc.protectedOffset = protectedOffset;
    ecc.protectedBytes = protectedBytes;
    ecc.rowBytes = rowBytes;
    ecc.dataShards = dataShards;
    ecc.parityShards = parityShards;

    // Build the new content first: the parity has to be taken over the bytes as they
    // will actually be stored, not over the ones they were before the table grew.
    std::vector<core::u8> content;
    content.reserve(protectedBytes + sizeof(pack::EccV1) + static_cast<std::size_t>(parityShards) * rowBytes);

    for (core::u32 i = 0u; i < header.sectionCount; ++i)
    {
        pack::SectionEntry entry{};
        std::memcpy(&entry, image.data() + sizeof(pack::Header) + i * entryBytes, sizeof(entry));
        entry.offset += entryBytes; // one more row in the table pushes every payload
        appendPod(content, entry);
    }

    pack::SectionEntry eccEntry{};
    eccEntry.type = static_cast<core::u32>(pack::SectionType::Ecc);
    eccEntry.offset = eccOffset;
    eccEntry.size = static_cast<core::u32>(sizeof(pack::EccV1)) + parityShards * rowBytes;
    eccEntry.reserved = 0u;
    appendPod(content, eccEntry);

    content.insert(content.end(), image.begin() + sizeof(pack::Header) + oldTableBytes, image.end());

    // Transversal, from codec/: one codeword down each column of the row-major span.
    std::vector<core::u8> parity(static_cast<std::size_t>(parityShards) * rowBytes, core::u8{0});
    if (!codec::transversalEncode(content.data(), static_cast<core::u32>(content.size()), dataShards, parityShards,
                                  rowBytes, parity.data()))
        return image;

    appendPod(content, ecc);
    content.insert(content.end(), parity.begin(), parity.end());

    pack::Header grown = header;
    grown.sectionCount = newSectionCount;
    // The locator lives in the header, outside the span the parity protects. Without
    // it a burst on the section table takes the parity's own address with it.
    grown.eccOffset = eccOffset;
    grown.eccSize = eccEntry.size;
    grown.totalSize = static_cast<core::u32>(sizeof(pack::Header) + content.size());
    grown.contentHash = pack::hashBytes(content.data(), static_cast<core::u32>(content.size()));

    std::vector<core::u8> out;
    out.resize(sizeof(pack::Header));
    std::memcpy(out.data(), &grown, sizeof(pack::Header));
    out.insert(out.end(), content.begin(), content.end());
    return out;
}

} // namespace lpl::editor
