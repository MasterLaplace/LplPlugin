/**
 * @file main.cpp
 * @brief lpl-bake — turns an authored `.lplscene` into a loadable `.lplpak`.
 *
 * The oven, as a command. A build embeds the result as a GRUB module so a
 * kernel with no filesystem still loads a real game, and a server can bake a
 * freshly pulled game without stopping.
 *
 * Usage: lpl-bake [--ecc] <input.lplscene> <output.lplpak>
 *        lpl-bake --header <symbol> <input.lplscene|-> <output.hpp>
 *
 * `--ecc` attaches a transversal Reed-Solomon parity section, so the ring-0 reader
 * corrects a bad sector instead of refusing the cartridge. It is opt-in because it
 * costs image size and only a cartridge that will be STORED needs it — one baked into
 * a build that is about to be rebuilt does not.
 *
 * The second form is the missing half of the checked-in cartridges. Two packs live
 * in the tree as byte arrays — `ParityPackBlob.hpp` and `ViewerPackBlob.hpp` — because
 * the kernel build must need no host tool, and `test-game-pack` asserts they still
 * match what the oven emits so a stale blob fails a test rather than quietly shipping
 * last month's world. That guard had no counterpart: nothing could REFRESH them, so
 * every recipe change dead-ended on a hex dump pasted by hand. The two blobs had even
 * drifted into different formatting, which is what hand-pasting looks like after a
 * while.
 *
 * Only the array body is rewritten, between the symbol's `= {` and its `};`. The prose
 * around it says why the blob exists and is worth more than the bytes.
 *
 * An input of `-` means "the parity recipe", which comes from code rather than from a
 * document — that is exactly what makes it the reference.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/editor/EccBaker.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// Reads a whole file, or reports why it could not.
[[nodiscard]] bool readFile(const char *path, std::string &out)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        return false;
    out.assign(std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{});
    return true;
}

/// The bytes, sixteen per line, in the house style.
[[nodiscard]] std::string formatBytes(const std::vector<lpl::core::u8> &bytes)
{
    std::ostringstream out;
    for (std::size_t i = 0u; i < bytes.size(); ++i)
    {
        if (i % 16u == 0u)
            out << "    ";
        char cell[12];
        std::snprintf(cell, sizeof(cell), "0x%02Xu,", static_cast<unsigned>(bytes[i]));
        out << cell;
        out << ((i % 16u == 15u || i + 1u == bytes.size()) ? "\n" : " ");
    }
    return out.str();
}

/**
 * @brief Replaces one array body in a header, leaving everything else alone.
 * @return false when the symbol is not there, which is a typo rather than a stale file.
 */
[[nodiscard]] bool rewriteArray(std::string &header, const std::string &symbol, const std::string &body)
{
    const std::string opening = "inline constexpr core::u8 " + symbol + "[] = {";
    const std::size_t at = header.find(opening);
    if (at == std::string::npos)
        return false;

    const std::size_t bodyStart = at + opening.size();
    const std::size_t bodyEnd = header.find("\n};", bodyStart);
    if (bodyEnd == std::string::npos)
        return false;

    header.replace(bodyStart, bodyEnd - bodyStart, "\n" + body);
    return true;
}

/// Bakes what @p source names: a document, or the parity recipe when it is "-".
[[nodiscard]] bool bakeSource(const char *source, std::vector<lpl::core::u8> &out)
{
    if (std::strcmp(source, "-") == 0)
    {
        out = lpl::editor::bakeGamePack(lpl::procgen::parityWorldRecipe());
        return true;
    }

    std::string document;
    if (!readFile(source, document))
    {
        std::fprintf(stderr, "lpl-bake: cannot read %s\n", source);
        return false;
    }
    const auto image = lpl::editor::bakeSceneDocument(document);
    if (!image)
    {
        std::fprintf(stderr, "lpl-bake: %s: %s\n", source, image.error().message().c_str());
        return false;
    }
    out = *image;
    return true;
}

/// The `--header` form: refresh one checked-in blob in place.
[[nodiscard]] int emitHeader(const char *symbol, const char *source, const char *headerPath)
{
    std::vector<lpl::core::u8> image;
    if (!bakeSource(source, image))
        return 1;

    std::string header;
    if (!readFile(headerPath, header))
    {
        std::fprintf(stderr, "lpl-bake: cannot read %s\n", headerPath);
        return 1;
    }

    if (!rewriteArray(header, symbol, formatBytes(image)))
    {
        std::fprintf(stderr, "lpl-bake: %s declares no `inline constexpr core::u8 %s[]`\n", headerPath, symbol);
        return 1;
    }

    std::ofstream out{headerPath, std::ios::binary | std::ios::trunc};
    if (!out)
    {
        std::fprintf(stderr, "lpl-bake: cannot write %s\n", headerPath);
        return 1;
    }
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!out)
    {
        std::fprintf(stderr, "lpl-bake: short write to %s\n", headerPath);
        return 1;
    }

    std::printf("lpl-bake: %s -> %s::%s (%zu bytes)\n", source, headerPath, symbol, image.size());
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 5 && std::strcmp(argv[1], "--header") == 0)
        return emitHeader(argv[2], argv[3], argv[4]);

    bool armour = false;
    int first = 1;
    if (argc >= 2 && std::strcmp(argv[1], "--ecc") == 0)
    {
        armour = true;
        first = 2;
    }

    if (argc != first + 2)
    {
        std::fprintf(stderr,
                     "usage: %s [--ecc] <input.lplscene> <output.lplpak>\n"
                     "       %s --header <symbol> <input.lplscene|-> <output.hpp>\n",
                     argv[0], argv[0]);
        return 2;
    }

    std::string document;
    if (!readFile(argv[first], document))
    {
        std::fprintf(stderr, "lpl-bake: cannot read %s\n", argv[first]);
        return 1;
    }

    const auto baked = lpl::editor::bakeSceneDocument(document);
    if (!baked)
    {
        std::fprintf(stderr, "lpl-bake: %s: %s\n", argv[first], baked.error().message().c_str());
        return 1;
    }
    const std::vector<lpl::core::u8> armoured = armour ? lpl::editor::attachEcc(*baked) : *baked;
    const std::vector<lpl::core::u8> *image = &armoured;

    std::ofstream output{argv[first + 1], std::ios::binary};
    if (!output)
    {
        std::fprintf(stderr, "lpl-bake: cannot write %s\n", argv[first + 1]);
        return 1;
    }
    output.write(reinterpret_cast<const char *>(image->data()), static_cast<std::streamsize>(image->size()));
    if (!output)
    {
        std::fprintf(stderr, "lpl-bake: short write to %s\n", argv[first + 1]);
        return 1;
    }

    std::printf("lpl-bake: %s -> %s (%zu bytes%s)\n", argv[first], argv[first + 1], image->size(),
                armour ? ", parity attached" : "");
    return 0;
}
