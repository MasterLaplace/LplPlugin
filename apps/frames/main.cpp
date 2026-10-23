#include <lpl/core/Types.hpp>
#include <lpl/ecology/LivingRecipe.hpp>
#include <lpl/ecs/System.hpp>
#include <lpl/engine/Config.hpp>
#include <lpl/engine/HostProfile.hpp>
#include <lpl/engine/ResourceManager.hpp>
#include <lpl/engine/ViewProfile.hpp>
#include <lpl/engine/World.hpp>
#include <lpl/image/Codec.hpp>
#include <lpl/image/Image.hpp>
#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/pack/Cartridge.hpp>
#include <lpl/pack/ViewerPackBlob.hpp>
#include <lpl/platform/linux/LinuxPlatform.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/render/SoftwareRasterizer.hpp>
#include <lpl/samples/TerrainWorld.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

/**
 * @brief One frame the tool draws: the world settles, keys are pressed, fixed steps run, and the
 *        scene is drawn.
 */
struct ReferenceFrame {
    std::string name;            /**< The frame's name, and its file name without the extension. */
    lpl::core::u32 settle{600u}; /**< Fixed steps of 1/60 s run before the keys, while the ground streams in. */
    std::string keys;            /**< Keys pressed then, as the client's keyboard sends them. */
    lpl::core::u32 ticks{1u};    /**< Fixed steps of 1/60 s run after the keys. */
    std::string shows;           /**< What the frame exercises. */
};

/** The list of frames lpl-frames reads unless --frames names another, from the project's root. */
constexpr const char *kDefaultFrames = "apps/frames/reference-frames.tsv";

/**
 * @brief Reads a count of fixed steps from @p text.
 *
 * @return The count, or nothing when @p text is not a whole number below a million.
 */
std::optional<lpl::core::u32> readTicks(const std::string &text)
{
    char *end = nullptr;
    const unsigned long value = std::strtoul(text.c_str(), &end, 10);

    if (text.empty() || *end != '\0' || value >= 1000000ul)
        return std::nullopt;
    return static_cast<lpl::core::u32>(value);
}

/**
 * @brief Reads the frames of @p path: one per line, five fields separated by tabs (name, settle,
 *        keys, ticks, what it shows), with - for no keys; blank lines and lines starting with #
 *        are skipped.
 *
 * @return The frames, or nothing after printing the line that does not read.
 */
std::optional<std::vector<ReferenceFrame>> readFrames(const char *path)
{
    std::ifstream file{path};
    std::vector<ReferenceFrame> frames;
    std::string line;
    lpl::core::u32 number = 0u;

    if (!file)
    {
        std::fprintf(stderr, "lpl-frames: cannot read %s\n", path);
        return std::nullopt;
    }
    while (std::getline(file, line))
    {
        ++number;
        if (line.empty() || line[0] == '#')
            continue;

        std::vector<std::string> fields;
        std::stringstream stream{line};
        std::string field;
        while (std::getline(stream, field, '\t'))
            fields.push_back(field);

        const std::optional<lpl::core::u32> settle = fields.size() == 5u ? readTicks(fields[1]) : std::nullopt;
        const std::optional<lpl::core::u32> ticks = fields.size() == 5u ? readTicks(fields[3]) : std::nullopt;
        if (!settle || !ticks || fields[0].empty())
        {
            std::fprintf(stderr,
                         "lpl-frames: %s:%u: expected name, settle, keys, ticks and what it shows, tab-separated\n",
                         path, number);
            return std::nullopt;
        }
        frames.push_back(
            ReferenceFrame{fields[0], *settle, fields[2] == "-" ? std::string{} : fields[2], *ticks, fields[4]});
    }
    return frames;
}

/** Bytes of the per-frame arena the world gets, as the engine gives one. */
constexpr lpl::core::usize kArenaBytes = 1u << 20;

/**
 * @brief Draws @p frame into @p colour, in a world built from the built-in cartridge.
 *
 * @details The world runs as the ring-0 client runs it: the Ring0Client host profile, the view the
 *          pack carries, and the registry's buffers swapped after the physics phase. It settles,
 *          onRender reads the keys, and the frame's steps run. The frame drawn is renderScene's,
 *          without the HUD.
 *
 * @return false when the world does not initialise or its scheduler is malformed.
 */
bool drawReferenceFrame(const ReferenceFrame &frame, std::vector<lpl::core::u32> &colour,
                        std::vector<lpl::core::f32> &depth)
{
    const lpl::pack::Cartridge cartridge =
        lpl::pack::loadCartridge(nullptr, 0u, lpl::pack::kViewerPackBytes, lpl::pack::kViewerPackSize,
                                 lpl::procgen::parityWorldRecipe(), lpl::ecology::parityLivingRecipe());
    const lpl::engine::ViewProfile view =
        cartridge.viewFromPack ? lpl::engine::toEngineView(cartridge.view) : lpl::engine::ViewProfile{};

    lpl::engine::Config::Builder builder;
    builder.tickRate(60u);
    lpl::engine::applyHostProfile(builder, lpl::engine::HostProfile::Ring0Client);
    const lpl::engine::Config config = builder.build();

    lpl::platform::linux_host::LinuxPlatform platform;
    lpl::engine::ResourceManager resources;
    std::vector<lpl::core::u8> arenaBytes(kArenaBytes);
    lpl::memory::ArenaAllocator arena{arenaBytes.data(), arenaBytes.size()};
    lpl::engine::WorldContext context{platform, resources, arena, config, nullptr};
    lpl::samples::TerrainWorld world{cartridge.recipe, cartridge.living, view};

    if (!world.onInit(context).has_value() || !world.scheduler().buildGraph().has_value())
        return false;
    world.scheduler().setPhaseCallback(lpl::ecs::SchedulePhase::Physics,
                                       [&world]() { world.registry().swapAllBuffers(); });

    for (lpl::core::u32 tick = 0u; tick < frame.settle; ++tick)
        world.onFixedStep(1.0f / 60.0f);
    for (const char key : frame.keys)
        platform.inputBackend().pushCharacter(key);
    arena.reset();
    world.onRender(context, 0.0);
    for (lpl::core::u32 tick = 0u; tick < frame.ticks; ++tick)
        world.onFixedStep(1.0f / 60.0f);

    const lpl::render::RenderTarget target{colour.data(), depth.data(), lpl::samples::TerrainWorld::kRenderWidth,
                                           lpl::samples::TerrainWorld::kRenderHeight};
    world.renderScene(target);
    return true;
}

/**
 * @brief Writes @p colour as a binary PPM at @p path.
 *
 * @return false when the frame does not encode or the file cannot be written whole.
 */
bool writeFrame(const std::vector<lpl::core::u32> &colour, const std::string &path)
{
    const lpl::core::u32 width = lpl::samples::TerrainWorld::kRenderWidth;
    const lpl::core::u32 height = lpl::samples::TerrainWorld::kRenderHeight;
    lpl::image::Image picture{width, height};
    lpl::pmr::vector<lpl::core::u8> encoded;

    for (lpl::core::u32 y = 0u; y < height; ++y)
        for (lpl::core::u32 x = 0u; x < width; ++x)
            picture.set(static_cast<lpl::core::i32>(x), static_cast<lpl::core::i32>(y),
                        0xFF000000u | colour[static_cast<std::size_t>(y) * width + x]);
    if (!lpl::image::writePpm(picture, encoded))
        return false;

    std::FILE *const file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
        return false;
    const std::size_t written = std::fwrite(encoded.data(), 1u, encoded.size(), file);
    return std::fclose(file) == 0 && written == encoded.size();
}

/**
 * @brief Prints how to call the tool.
 */
void printUsage()
{
    std::printf("usage: lpl-frames [--frames <file>] [--write <directory>] [--only <name>] [--list]\n"
                "  Draws the reference frames of the client's world, the HUD left out, and prints\n"
                "  one line per frame: its name and the fold of its pixels. Two builds that print\n"
                "  the same fold draw the same frame.\n"
                "  --frames <file>      the frames to draw, %s unless named: one per line,\n"
                "                       name, settle, keys, ticks and what it shows, tab-separated\n"
                "  --write <directory>  also writes each frame there, as <name>.ppm\n"
                "  --only <name>        draws that frame alone\n"
                "  --list               prints the frames and what each shows, and draws nothing\n",
                kDefaultFrames);
}

} // namespace

/**
 * @brief Draws the reference frames, and prints the fold of each.
 *
 * @return 0 when every frame asked for was drawn, 1 when one was not or could not be written,
 *         2 on a usage error.
 */
int main(int argc, char **argv)
{
    const char *framesPath = kDefaultFrames;
    const char *directory = nullptr;
    const char *only = nullptr;
    bool list = false;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            framesPath = argv[++i];
        else if (std::strcmp(argv[i], "--write") == 0 && i + 1 < argc)
            directory = argv[++i];
        else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc)
            only = argv[++i];
        else if (std::strcmp(argv[i], "--list") == 0)
            list = true;
        else
        {
            printUsage();
            return 2;
        }
    }

    const std::optional<std::vector<ReferenceFrame>> frames = readFrames(framesPath);
    if (!frames)
        return 2;

    std::vector<lpl::core::u32> colour(static_cast<std::size_t>(lpl::samples::TerrainWorld::kRenderWidth) *
                                       lpl::samples::TerrainWorld::kRenderHeight);
    std::vector<lpl::core::f32> depth(colour.size());
    int status = 0;
    bool matched = false;

    for (const ReferenceFrame &frame : *frames)
    {
        if (only != nullptr && frame.name != only)
            continue;
        matched = true;
        if (list)
        {
            std::printf("%-18s %s\n", frame.name.c_str(), frame.shows.c_str());
            continue;
        }
        if (!drawReferenceFrame(frame, colour, depth))
        {
            std::fprintf(stderr, "lpl-frames: %s: the world did not initialise\n", frame.name.c_str());
            status = 1;
            continue;
        }

        const lpl::render::RenderTarget target{colour.data(), depth.data(), lpl::samples::TerrainWorld::kRenderWidth,
                                               lpl::samples::TerrainWorld::kRenderHeight};
        std::printf("%s 0x%08X\n", frame.name.c_str(), lpl::render::foldTarget(target));
        if (directory != nullptr && !writeFrame(colour, std::string{directory} + "/" + frame.name + ".ppm"))
        {
            std::fprintf(stderr, "lpl-frames: %s: could not write %s/%s.ppm\n", frame.name.c_str(), directory,
                         frame.name.c_str());
            status = 1;
        }
    }

    if (only != nullptr && !matched)
    {
        std::fprintf(stderr, "lpl-frames: no frame is named %s; --list names them\n", only);
        return 2;
    }
    return status;
}
