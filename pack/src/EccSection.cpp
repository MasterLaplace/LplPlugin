/**
 * @file EccSection.cpp
 * @brief Transversal Reed-Solomon repair of a pack image.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/pack/EccSection.hpp>

#include <lpl/codec/ReedSolomon.hpp>
#include <lpl/pack/GamePack.hpp>
#include <lpl/std/cstring.hpp>

namespace lpl::pack {

namespace {

/**
 * @brief Locates the parity section without going through View.
 *
 * View::open verifies the content hash, and on a damaged image that check is exactly
 * what fails — so the repair path cannot use it. It walks the section table itself,
 * bounds-checking every step, because the table it is walking is the thing that might
 * be corrupt.
 *
 * @param bytes        The image.
 * @param size         Bytes available.
 * @param outEcc       Receives the section header.
 * @param outParity    Receives a pointer to the parity bytes.
 * @return false when there is no usable parity section.
 */
[[nodiscard]] bool findEcc(core::u8 *bytes, core::u32 size, EccV1 &outEcc, core::u8 *&outParity) noexcept
{
    if (bytes == nullptr || size < sizeof(Header) + sizeof(SectionEntry))
        return false;

    Header header{};
    lpl::pmr::memcpy(&header, bytes, sizeof(Header));
    if (header.totalSize > size)
        return false;

    const core::u64 tableBytes = static_cast<core::u64>(header.sectionCount) * sizeof(SectionEntry);
    if (sizeof(Header) + tableBytes > header.totalSize)
        return false;

    // The header, not the section table. The table is inside the span the parity
    // protects, so on the one burst that matters most — the one that lands on it —
    // the table is precisely what is unreadable. See Header::eccOffset.
    const core::u32 offset = header.eccOffset;
    const core::u32 sectionSize = header.eccSize;
    {
        if (offset == 0u || sectionSize < sizeof(EccV1))
            return false;
        if (static_cast<core::u64>(offset) + sectionSize > header.totalSize)
            return false;

        lpl::pmr::memcpy(&outEcc, bytes + offset, sizeof(EccV1));

        const core::u64 parityBytes = static_cast<core::u64>(outEcc.parityShards) * outEcc.rowBytes;
        if (sizeof(EccV1) + parityBytes != sectionSize)
            return false;
        if (static_cast<core::u64>(outEcc.protectedOffset) + outEcc.protectedBytes > header.totalSize)
            return false;
        if (outEcc.dataShards == 0u || outEcc.parityShards == 0u || outEcc.rowBytes == 0u)
            return false;
        if (outEcc.dataShards + outEcc.parityShards > codec::kMaxCodewordSymbols)
            return false;
        if (outEcc.parityShards > codec::kMaxParitySymbols)
            return false;

        outParity = bytes + offset + sizeof(EccV1);
        return true;
    }
}

} // namespace

bool repairPack(core::u8 *bytes, core::u32 size, EccRepairReport &outReport) noexcept
{
    outReport = EccRepairReport{};

    EccV1 ecc{};
    core::u8 *parity = nullptr;
    if (!findEcc(bytes, size, ecc, parity))
        return false;

    outReport.present = true;

    // The transversal engine lives in codec/, not here. Writing the column walk a
    // second time for the Rosetta plate would have been the third copy of "cut a span
    // into rows and code down the columns", and the first two would then have been
    // free to disagree about how the tail is padded.
    codec::TransversalReport columns{};
    if (!codec::transversalRepair(bytes + ecc.protectedOffset, ecc.protectedBytes, parity, ecc.dataShards,
                                  ecc.parityShards, ecc.rowBytes, columns))
        return false;

    outReport.codewords = columns.codewords;
    outReport.damagedCodewords = columns.damagedCodewords;
    outReport.correctedBytes = columns.correctedBytes;

    outReport.repaired = true;
    return true;
}

} // namespace lpl::pack
