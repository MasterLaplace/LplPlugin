/**
 * @file Peeling.cpp
 * @brief Belief propagation over the droplet graph, and the elimination behind it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/Peeling.hpp>

#include <lpl/codec/GaussJordan.hpp>

namespace lpl::codec {

namespace {

/**
 * @struct WorkingDroplet
 * @brief A droplet as the decoder mutates it: the unknowns still in it, and its value.
 */
struct WorkingDroplet {
    lpl::pmr::vector<core::u32> unknowns{};
    lpl::pmr::vector<core::u8> value{};
    bool consumed{false};
};

/**
 * @brief Removes @p index from @p list, if it is there.
 * @return true when it was removed.
 */
bool removeIndex(lpl::pmr::vector<core::u32> &list, core::u32 index) noexcept
{
    for (core::usize i = 0u; i < list.size(); ++i)
    {
        if (list[i] != index)
            continue;
        for (core::usize j = i + 1u; j < list.size(); ++j)
            list[j - 1u] = list[j];
        list.pop_back();
        return true;
    }
    return false;
}

/**
 * @brief destination ^= source, over @p bytes bytes.
 */
void xorBytes(core::u8 *destination, const core::u8 *source, core::u32 bytes) noexcept
{
    for (core::u32 i = 0u; i < bytes; ++i)
        destination[i] = static_cast<core::u8>(destination[i] ^ source[i]);
}

} // namespace

bool decodeDroplets(const lpl::pmr::vector<Droplet> &droplets, const SolitonTable &table, core::u32 blockCount,
                    core::u32 blockBytes, lpl::pmr::vector<core::u8> &outBlocks, DecodeReport &outReport)
{
    outReport = DecodeReport{};
    outReport.dropletsUsed = static_cast<core::u32>(droplets.size());

    outBlocks.clear();
    if (blockCount == 0u || blockBytes == 0u)
        return false;
    outBlocks.resize(static_cast<core::usize>(blockCount) * blockBytes, core::u8{0});

    lpl::pmr::vector<core::u8> solved;
    solved.resize(blockCount, core::u8{0});

    // Expand every seed once. This is where the wire's four bytes become a row of the
    // matrix again, and it is the only place the decoder needs the distribution — the
    // seed is meaningless without the same table the encoder drew from.
    lpl::pmr::vector<WorkingDroplet> working;
    working.resize(droplets.size());
    DropletPlan plan;
    for (core::usize d = 0u; d < droplets.size(); ++d)
    {
        expandDroplet(droplets[d].seed, table, plan);
        working[d].unknowns = plan.indices;
        working[d].value = droplets[d].payload;
        working[d].value.resize(blockBytes, core::u8{0});
    }

    // ── Belief propagation ────────────────────────────────────────────────────
    //
    // A droplet of degree one names its block outright. Resolving it removes that
    // block from every other droplet, which lowers their degree, which may expose the
    // next degree-one droplet. The whole decode is that cascade, and it is why the
    // robust soliton spends part of its mass manufacturing degree-one droplets: with
    // the ideal distribution alone the cascade often has nothing to start from.
    core::u32 resolved = 0u;
    bool progressed = true;
    while (progressed && resolved < blockCount)
    {
        progressed = false;
        for (core::usize d = 0u; d < working.size(); ++d)
        {
            WorkingDroplet &droplet = working[d];
            if (droplet.consumed || droplet.unknowns.size() != 1u)
                continue;

            const core::u32 block = droplet.unknowns[0];
            droplet.consumed = true;
            if (solved[block] != 0u)
                continue;

            for (core::u32 b = 0u; b < blockBytes; ++b)
                outBlocks[static_cast<core::usize>(block) * blockBytes + b] = droplet.value[b];
            solved[block] = 1u;
            ++resolved;
            ++outReport.peeledBlocks;
            progressed = true;

            // Propagate: every droplet still carrying this block loses it.
            for (core::usize e = 0u; e < working.size(); ++e)
            {
                if (e == d || working[e].consumed)
                    continue;
                if (!removeIndex(working[e].unknowns, block))
                    continue;
                xorBytes(working[e].value.data(), outBlocks.data() + static_cast<core::usize>(block) * blockBytes,
                         blockBytes);
            }
        }
    }

    if (resolved == blockCount)
    {
        outReport.recovered = true;
        return true;
    }

    // ── The Gaussian tail ─────────────────────────────────────────────────────
    //
    // What is left is a residual system with no degree-one droplet in it. It goes to
    // the same elimination the rest of the module uses, with the payload carried as
    // AUGMENTED COLUMNS of the very same matrix — that is what GaussJordan's
    // systemColumns parameter is for. Carrying the bytes in a parallel array instead
    // would mean writing a second elimination that mirrors every row operation, and a
    // mirror is a thing that can fall out of step.
    lpl::pmr::vector<core::u32> unknownColumn;
    unknownColumn.resize(blockCount, kNoPivot);
    core::u32 unknowns = 0u;
    for (core::u32 b = 0u; b < blockCount; ++b)
        if (solved[b] == 0u)
            unknownColumn[b] = unknowns++;

    core::u32 rows = 0u;
    for (core::usize d = 0u; d < working.size(); ++d)
        if (!working[d].consumed && !working[d].unknowns.empty())
            ++rows;

    outReport.residualRows = rows;
    if (rows == 0u || unknowns == 0u)
        return false;

    const core::u32 payloadBits = blockBytes * 8u;
    BitMatrix system{rows, unknowns + payloadBits};

    core::u32 row = 0u;
    for (core::usize d = 0u; d < working.size(); ++d)
    {
        const WorkingDroplet &droplet = working[d];
        if (droplet.consumed || droplet.unknowns.empty())
            continue;

        for (core::usize i = 0u; i < droplet.unknowns.size(); ++i)
        {
            const core::u32 column = unknownColumn[droplet.unknowns[i]];
            if (column != kNoPivot)
                system.set(row, column);
        }
        for (core::u32 b = 0u; b < blockBytes; ++b)
            for (core::u32 bit = 0u; bit < 8u; ++bit)
                if (((droplet.value[b] >> bit) & 1u) != 0u)
                    system.set(row, unknowns + b * 8u + bit);
        ++row;
    }

    const EliminationResult elimination = gaussJordan(system, unknowns);
    outReport.rank = elimination.rank;

    if (!isConsistent(system, elimination, unknowns))
        return false;
    if (elimination.rank < unknowns)
        return false; // under-determined: some blocks genuinely cannot be known

    // Read the answers off the augmented tail, one pivot row per unknown.
    for (core::u32 b = 0u; b < blockCount; ++b)
    {
        const core::u32 column = unknownColumn[b];
        if (column == kNoPivot)
            continue;
        const core::u32 pivotRow = column < elimination.rowOfPivotColumn.size()
                                       ? elimination.rowOfPivotColumn[column]
                                       : kNoPivot;
        if (pivotRow == kNoPivot)
            return false;

        core::u8 *const destination = outBlocks.data() + static_cast<core::usize>(b) * blockBytes;
        for (core::u32 byte = 0u; byte < blockBytes; ++byte)
        {
            core::u8 value = 0u;
            for (core::u32 bit = 0u; bit < 8u; ++bit)
                if (system.test(pivotRow, unknowns + byte * 8u + bit))
                    value = static_cast<core::u8>(value | (1u << bit));
            destination[byte] = value;
        }
        ++outReport.eliminatedBlocks;
    }

    outReport.recovered = true;
    return true;
}

} // namespace lpl::codec
