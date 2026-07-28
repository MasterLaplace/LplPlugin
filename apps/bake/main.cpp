/**
 * @file main.cpp
 * @brief lpl-bake — turns an authored `.lplscene` into a loadable `.lplpak`.
 *
 * The oven, as a command. A build embeds the result as a GRUB module so a
 * kernel with no filesystem still loads a real game, and a server can bake a
 * freshly pulled game without stopping.
 *
 * Usage: lpl-bake <input.lplscene> <output.lplpak>
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/editor/GamePackBaker.hpp>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s <input.lplscene> <output.lplpak>\n", argv[0]);
        return 2;
    }

    std::ifstream input{argv[1], std::ios::binary};
    if (!input)
    {
        std::fprintf(stderr, "lpl-bake: cannot read %s\n", argv[1]);
        return 1;
    }
    const std::string document{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};

    const auto image = lpl::editor::bakeSceneDocument(document);
    if (!image)
    {
        std::fprintf(stderr, "lpl-bake: %s: %s\n", argv[1], image.error().message().c_str());
        return 1;
    }

    std::ofstream output{argv[2], std::ios::binary};
    if (!output)
    {
        std::fprintf(stderr, "lpl-bake: cannot write %s\n", argv[2]);
        return 1;
    }
    output.write(reinterpret_cast<const char *>(image->data()), static_cast<std::streamsize>(image->size()));
    if (!output)
    {
        std::fprintf(stderr, "lpl-bake: short write to %s\n", argv[2]);
        return 1;
    }

    std::printf("lpl-bake: %s -> %s (%zu bytes)\n", argv[1], argv[2], image->size());
    return 0;
}
