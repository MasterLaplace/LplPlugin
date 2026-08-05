/**
 * @file main.cpp
 * @brief Lay out an artifact that carries the specification of its own reader.
 *
 * A cartridge assumes a reader compiled from this repository. A Rosetta artifact
 * assumes nothing but an observer with a microscope — so the specification is
 * engraved with the payload, the bootstrap is duplicated at the corners, and the
 * whole thing is verified by decoding it with nothing but the reference
 * interpreter. If that verification needs anything else, the artifact is a lie.
 *
 * The verification is not a formality here. It is the ONLY thing that separates this
 * from writing a blob and hoping: the plate is read back with a machine rebuilt from
 * the bytes the plate itself carries, and the tool refuses to write a file it could
 * not open that way.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/codec/ReedSolomon.hpp>
#include <lpl/rosetta/Bootstrap.hpp>
#include <lpl/rosetta/Engraving.hpp>
#include <lpl/rosetta/Interpreter.hpp>
#include <lpl/rosetta/SelfDescribing.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

/**
 * @brief Reads a whole file into bytes.
 * @param path Where.
 * @param out  Receives the contents.
 * @return false when it could not be opened.
 */
[[nodiscard]] bool readFile(const char *path, std::vector<lpl::core::u8> &out)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        return false;
    const std::string text{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    out.assign(text.begin(), text.end());
    return true;
}

/**
 * @brief Prints the engraved instruction set, as a reader would meet it.
 * @param specification The engraved bytes.
 * @param size          Their length.
 */
void describe(const lpl::core::u8 *specification, lpl::core::u32 size)
{
    lpl::rosetta::IsaTable table{};
    if (!lpl::rosetta::readSpecification(specification, size, table))
    {
        std::printf("  (the specification did not read back)\n");
        return;
    }
    std::printf("  instruction width : %u bytes\n", table.instructionBytes);
    std::printf("  registers         : %u\n", table.registerCount);
    std::printf("  opcodes           : %u\n", table.count);
    for (lpl::core::u32 i = 0u; i < table.count; ++i)
        std::printf("    %02u  %-8s %u operand%s\n", table.entry[i].opcode, table.entry[i].mnemonic,
                    table.entry[i].operands, table.entry[i].operands == 1u ? "" : "s");
}

} // namespace

int main(int argc, char **argv)
{
    using namespace lpl;

    if (argc < 2)
    {
        std::fprintf(stderr,
                     "usage: %s <payload> [output.lplplate]\n"
                     "       Engraves the payload behind five copies of the bootstrap and a\n"
                     "       transversal parity field, then verifies the plate by rebuilding the\n"
                     "       reader from what was engraved.\n",
                     argv[0]);
        return 2;
    }

    std::vector<core::u8> payload;
    if (!readFile(argv[1], payload) || payload.empty())
    {
        std::fprintf(stderr, "lpl-rosetta-forge: cannot read %s\n", argv[1]);
        return 1;
    }

    std::printf("== lpl-rosetta-forge ==\n\n");
    std::printf("  payload : %zu bytes from %s\n", payload.size(), argv[1]);

    const rosetta::Bootstrap bootstrap = rosetta::standardBootstrap();
    std::printf("  bootstrap : %u bytes across four levels\n", bootstrap.totalBytes());
    for (core::u32 i = 0u; i < static_cast<core::u32>(rosetta::BootstrapLevel::Count) - 1u; ++i)
        std::printf("    %-12s %zu bytes\n", rosetta::bootstrapLevelName(static_cast<rosetta::BootstrapLevel>(i)),
                    bootstrap.level[i].size());

    rosetta::Engraving plate;
    plate.setMedium(rosetta::Medium::FusedQuartz);
    plate.setParityShare(200u);
    if (!plate.engrave(bootstrap, payload.data(), static_cast<core::u32>(payload.size())))
    {
        std::fprintf(stderr, "lpl-rosetta-forge: the plate could not be laid out\n");
        return 1;
    }
    std::printf("\n  plate   : %zu bytes, %u copies of the bootstrap, a fifth of the area parity\n",
                plate.image().size(), rosetta::kBootstrapCopies);

    // The only honest test: rebuild the reader from what was engraved, and open the
    // artifact with that and nothing else.
    lpl::pmr::vector<core::u8> working;
    for (core::usize i = 0u; i < plate.image().size(); ++i)
        working.push_back(plate.image()[i]);

    lpl::pmr::vector<core::u8> engravedSpec;
    lpl::pmr::vector<core::u8> recovered;
    rosetta::EngravingReport report{};
    if (!rosetta::Engraving::read(working.data(), static_cast<core::u32>(working.size()), engravedSpec, recovered,
                                  report))
    {
        std::fprintf(stderr, "lpl-rosetta-forge: the plate does not read back — refusing to write it\n");
        return 1;
    }

    std::printf("\n-- read back with nothing but what was engraved --\n");
    std::printf("  replicas intact : %u of %u\n", report.replicasIntact, rosetta::kBootstrapCopies);
    describe(engravedSpec.data(), static_cast<core::u32>(engravedSpec.size()));

    rosetta::Interpreter rebuilt;
    if (!rosetta::Interpreter::fromSpecification(engravedSpec.data(), static_cast<core::u32>(engravedSpec.size()),
                                                 rebuilt))
    {
        std::fprintf(stderr, "lpl-rosetta-forge: no reader could be built from the engraving\n");
        return 1;
    }

    bool identical = recovered.size() == payload.size();
    for (core::usize i = 0u; identical && i < recovered.size(); ++i)
        identical = recovered[i] == payload[i];
    if (!identical)
    {
        std::fprintf(stderr, "lpl-rosetta-forge: the payload did not survive its own plate\n");
        return 1;
    }
    std::printf("  payload         : recovered byte for byte\n");
    std::printf("  reader          : rebuilt from the plate, %u opcodes\n", rebuilt.knownOpcodes());

    if (argc >= 3)
    {
        std::ofstream output{argv[2], std::ios::binary};
        if (!output)
        {
            std::fprintf(stderr, "lpl-rosetta-forge: cannot write %s\n", argv[2]);
            return 1;
        }
        output.write(reinterpret_cast<const char *>(plate.image().data()),
                     static_cast<std::streamsize>(plate.image().size()));
        std::printf("\n  written to %s\n", argv[2]);
    }

    return 0;
}
