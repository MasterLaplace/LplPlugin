/**
 * @file Erasure.cpp
 * @brief The façade: cut, emit, and put back together.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/Erasure.hpp>

namespace lpl::codec {

bool encodeErasure(const core::u8 *payload, core::u32 bytes, const ErasureParams &params, ErasureShape &outShape,
                   lpl::pmr::vector<Droplet> &outDroplets)
{
    outShape = ErasureShape{};
    outDroplets.clear();

    if (payload == nullptr || bytes == 0u || params.blockBytes == 0u)
        return false;

    const core::u32 blockCount = (bytes + params.blockBytes - 1u) / params.blockBytes;
    outShape.blockCount = blockCount;
    outShape.blockBytes = params.blockBytes;
    outShape.originalBytes = bytes;

    // The padded copy is the encoder's, not the caller's: a SourceView is a view, and
    // reading past the caller's buffer to reach the tail of the last block is exactly
    // the kind of read a fuzzer finds and a test never does.
    lpl::pmr::vector<core::u8> padded;
    padded.resize(static_cast<core::usize>(blockCount) * params.blockBytes, core::u8{0});
    for (core::u32 i = 0u; i < bytes; ++i)
        padded[i] = payload[i];

    SourceView source;
    source.bytes = padded.data();
    source.blockBytes = params.blockBytes;
    source.blockCount = blockCount;

    const Fountain fountain{source, params.tuning};

    // K * (1 + epsilon). The overhead is what pays for the droplets that turn out to
    // be linearly dependent on the ones already received — a fountain has no schedule,
    // so a few of them necessarily say nothing new.
    const core::u64 wanted = static_cast<core::u64>(blockCount) * (1000u + params.overheadPermille) / 1000u;
    const core::u32 count = wanted <= blockCount ? blockCount + 1u : static_cast<core::u32>(wanted);

    Droplet droplet;
    for (core::u32 i = 0u; i < count; ++i)
    {
        fountain.emit(params.firstSeed + i, droplet);
        Droplet kept;
        kept.seed = droplet.seed;
        kept.payload = droplet.payload;
        outDroplets.push_back(kept);
    }

    return true;
}

bool decodeErasure(const lpl::pmr::vector<Droplet> &droplets, const ErasureShape &shape, const ErasureParams &params,
                   lpl::pmr::vector<core::u8> &outBytes, DecodeReport &outReport)
{
    outBytes.clear();
    outReport = DecodeReport{};

    if (shape.blockCount == 0u || shape.blockBytes == 0u)
        return false;

    // The decoder rebuilds the SAME distribution the encoder drew from. Not a copy of
    // it — the same construction from the same K and the same constants, which is what
    // makes a seed mean one thing rather than two.
    SolitonParams tuning = params.tuning;
    tuning.sourceBlocks = shape.blockCount;
    SolitonTable table;
    table.build(tuning);

    lpl::pmr::vector<core::u8> blocks;
    if (!decodeDroplets(droplets, table, shape.blockCount, shape.blockBytes, blocks, outReport))
        return false;

    const core::u32 length =
        shape.originalBytes <= blocks.size() ? shape.originalBytes : static_cast<core::u32>(blocks.size());
    outBytes.resize(length, core::u8{0});
    for (core::u32 i = 0u; i < length; ++i)
        outBytes[i] = blocks[i];
    return true;
}

} // namespace lpl::codec
