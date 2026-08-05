/**
 * @file Engraving.cpp
 * @brief The plate: five replicas, a payload, and a transversal parity field.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/rosetta/Engraving.hpp>

#include <lpl/codec/ReedSolomon.hpp>

namespace lpl::rosetta {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief FNV-1a over a byte span.
 * @param bytes Span.
 * @param size  Its length.
 * @return The signature.
 */
[[nodiscard]] core::u32 foldBytes(const core::u8 *bytes, core::u32 size) noexcept
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < size; ++i)
        hash = (hash ^ bytes[i]) * kFnv1aPrime;
    return hash;
}

/**
 * @brief Appends a little-endian word.
 * @param out   Destination.
 * @param value What to append.
 */
void appendWord(lpl::pmr::vector<core::u8> &out, core::u32 value)
{
    out.push_back(static_cast<core::u8>(value & 0xFFu));
    out.push_back(static_cast<core::u8>((value >> 8) & 0xFFu));
    out.push_back(static_cast<core::u8>((value >> 16) & 0xFFu));
    out.push_back(static_cast<core::u8>((value >> 24) & 0xFFu));
}

/**
 * @brief Reads a little-endian word.
 * @param bytes Source.
 * @return The value.
 */
[[nodiscard]] core::u32 readWord(const core::u8 *bytes) noexcept
{
    return static_cast<core::u32>(bytes[0]) | (static_cast<core::u32>(bytes[1]) << 8) |
           (static_cast<core::u32>(bytes[2]) << 16) | (static_cast<core::u32>(bytes[3]) << 24);
}

/**
 * @brief Builds one replica: magic, level lengths, level bytes, and a fold over them.
 * @param bootstrap The four levels.
 * @param out       Receives the replica.
 */
void buildReplica(const Bootstrap &bootstrap, lpl::pmr::vector<core::u8> &out)
{
    out.clear();
    for (core::u32 i = 0u; i < 4u; ++i)
        out.push_back(kReplicaMagic[i]);

    const core::u32 levels = static_cast<core::u32>(BootstrapLevel::Count) - 1u;
    appendWord(out, levels);
    for (core::u32 i = 0u; i < levels; ++i)
        appendWord(out, static_cast<core::u32>(bootstrap.level[i].size()));
    for (core::u32 i = 0u; i < levels; ++i)
        for (core::usize b = 0u; b < bootstrap.level[i].size(); ++b)
            out.push_back(bootstrap.level[i][b]);

    // The fold covers everything above it, so a replica that survived a cut can prove
    // it. Without it a reader would take the first four bytes that happen to spell the
    // magic and read rubble as a specification.
    appendWord(out, foldBytes(out.data(), static_cast<core::u32>(out.size())));
}

} // namespace

bool Engraving::engrave(const Bootstrap &bootstrap, const core::u8 *payload, core::u32 payloadSize)
{
    _image.clear();
    if (payload == nullptr || payloadSize == 0u)
        return false;

    lpl::pmr::vector<core::u8> replica;
    buildReplica(bootstrap, replica);
    const core::u32 replicaBytes = static_cast<core::u32>(replica.size());

    lpl::pmr::vector<core::u8> coded;
    // Header: replica stride and count, then the payload offset. Outside the coded
    // region on purpose — it is the bootstrap of the bootstrap, the same reason a
    // pack's parity locator lives in its header rather than in its section table.
    appendWord(coded, replicaBytes);
    appendWord(coded, kBootstrapCopies);
    appendWord(coded, payloadSize);

    for (core::u32 copy = 0u; copy < kBootstrapCopies; ++copy)
        for (core::u32 i = 0u; i < replicaBytes; ++i)
            coded.push_back(replica[i]);

    for (core::u32 i = 0u; i < payloadSize; ++i)
        coded.push_back(payload[i]);
    appendWord(coded, foldBytes(payload, payloadSize));

    // A fifth of the coded area, as parity rows. dataShards is chosen so a row is a
    // meaningful slice of the plate rather than a byte: a burst has to be able to sit
    // inside ONE row for the transversal argument to hold.
    const core::u32 codedBytes = static_cast<core::u32>(coded.size());
    core::u32 dataShards = 32u;
    if (dataShards > codedBytes)
        dataShards = codedBytes;
    if (dataShards == 0u)
        return false;

    const core::u32 rowBytes = (codedBytes + dataShards - 1u) / dataShards;
    core::u32 parityShards = (dataShards * _parityPermille) / 1000u;
    if (parityShards == 0u)
        parityShards = 1u;
    if (parityShards > codec::kMaxParitySymbols)
        parityShards = codec::kMaxParitySymbols;

    lpl::pmr::vector<core::u8> parity;
    parity.resize(static_cast<core::usize>(parityShards) * rowBytes, core::u8{0});
    if (!codec::transversalEncode(coded.data(), codedBytes, dataShards, parityShards, rowBytes, parity.data()))
        return false;

    appendWord(_image, codedBytes);
    appendWord(_image, dataShards);
    appendWord(_image, parityShards);
    appendWord(_image, rowBytes);
    for (core::u32 i = 0u; i < codedBytes; ++i)
        _image.push_back(coded[i]);
    for (core::usize i = 0u; i < parity.size(); ++i)
        _image.push_back(parity[i]);

    return true;
}

bool Engraving::read(core::u8 *bytes, core::u32 size, lpl::pmr::vector<core::u8> &outSpec,
                     lpl::pmr::vector<core::u8> &outPayload, EngravingReport &outReport)
{
    outSpec.clear();
    outPayload.clear();
    outReport = EngravingReport{};

    constexpr core::u32 kPlateHeader = 16u;
    if (bytes == nullptr || size < kPlateHeader)
        return false;

    const core::u32 codedBytes = readWord(bytes);
    const core::u32 dataShards = readWord(bytes + 4u);
    const core::u32 parityShards = readWord(bytes + 8u);
    const core::u32 rowBytes = readWord(bytes + 12u);
    if (codedBytes == 0u || dataShards == 0u || parityShards == 0u || rowBytes == 0u)
        return false;
    if (static_cast<core::u64>(kPlateHeader) + codedBytes + static_cast<core::u64>(parityShards) * rowBytes > size)
        return false;

    core::u8 *const coded = bytes + kPlateHeader;
    core::u8 *const parity = coded + codedBytes;

    // Repair first. It may fail — a plate broken in half is far past any column code —
    // and that is NOT the end of the read: the five replicas exist precisely for the
    // damage the parity cannot touch.
    codec::TransversalReport columns{};
    if (codec::transversalRepair(coded, codedBytes, parity, dataShards, parityShards, rowBytes, columns))
    {
        outReport.repairedColumns = columns.damagedCodewords;
        outReport.repairedBytes = columns.correctedBytes;
    }

    if (codedBytes < 12u)
        return false;
    const core::u32 replicaBytes = readWord(coded);
    const core::u32 replicaCount = readWord(coded + 4u);
    const core::u32 payloadSize = readWord(coded + 8u);
    if (replicaBytes < 12u || replicaCount == 0u || payloadSize == 0u)
        return false;

    // Find a replica that proves itself. Scanning every copy rather than trusting the
    // first is the whole reason there are five of them.
    for (core::u32 copy = 0u; copy < replicaCount; ++copy)
    {
        const core::u64 offset = 12u + static_cast<core::u64>(copy) * replicaBytes;
        if (offset + replicaBytes > codedBytes)
            break;
        const core::u8 *const replica = coded + offset;

        bool magicOk = true;
        for (core::u32 i = 0u; i < 4u; ++i)
            magicOk = magicOk && replica[i] == kReplicaMagic[i];
        if (!magicOk)
            continue;
        if (foldBytes(replica, replicaBytes - 4u) != readWord(replica + replicaBytes - 4u))
            continue;

        ++outReport.replicasIntact;
        if (outReport.bootstrapFound)
            continue;

        const core::u32 levels = readWord(replica + 4u);
        if (levels == 0u)
            continue;

        core::u32 cursor = 8u;
        core::u32 lengths[static_cast<core::u32>(BootstrapLevel::Count)]{};
        const core::u32 kept = levels > static_cast<core::u32>(BootstrapLevel::Count)
                                   ? static_cast<core::u32>(BootstrapLevel::Count)
                                   : levels;
        for (core::u32 i = 0u; i < kept; ++i)
        {
            lengths[i] = readWord(replica + cursor);
            cursor += 4u;
        }

        // Level 3 is the machine: the bytes an interpreter is rebuilt from. That is
        // what the caller wants out of a plate, and everything above it is what makes
        // those bytes legible in the first place.
        core::u32 levelStart = cursor;
        for (core::u32 i = 0u; i + 1u < kept; ++i)
            levelStart += lengths[i];
        if (kept >= 4u && levelStart + lengths[3] <= replicaBytes)
        {
            for (core::u32 i = 0u; i < lengths[3]; ++i)
                outSpec.push_back(replica[levelStart + i]);
            outReport.bootstrapFound = true;
        }
    }

    const core::u64 payloadOffset = 12u + static_cast<core::u64>(replicaCount) * replicaBytes;
    if (payloadOffset + payloadSize + 4u > codedBytes)
        return false;

    const core::u8 *const payload = coded + payloadOffset;
    if (foldBytes(payload, payloadSize) != readWord(payload + payloadSize))
        return false;

    for (core::u32 i = 0u; i < payloadSize; ++i)
        outPayload.push_back(payload[i]);
    outReport.payloadRecovered = true;
    return outReport.bootstrapFound;
}

} // namespace lpl::rosetta
