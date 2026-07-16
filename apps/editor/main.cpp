/**
 * @file main.cpp
 * @brief `lpl-editor` — a REPL over the editor command stream.
 *
 * Reads one JSON command (or batch array) per line from stdin, runs it against a
 * live ECS world through @c editor::CommandProcessor, and prints the JSON report.
 * It is the human-facing twin of the Caine AI bridge: whatever the model would
 * emit, you can type here. A leading `#` line is a comment; the bare word
 * `save`, `count`, or `quit` are shortcuts. Piping a file of commands in
 * replays a whole scene recipe deterministically.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <cstdio>
#include <string>

#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandProcessor.hpp>

namespace {

void printBanner()
{
    std::puts("lpl-editor — deterministic scene command REPL");
    std::puts("  type a JSON command, e.g. {\"cmd\":\"generate_heightfield\",\"seed\":7,\"cols\":16,\"rows\":16}");
    std::puts("  shortcuts: count | save | quit    (# starts a comment line)");
    std::puts("");
}

std::string trim(const std::string &s)
{
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n'))
        ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n'))
        --b;
    return s.substr(a, b - a);
}

} // namespace

int main()
{
    printBanner();

    lpl::ecs::Registry world;
    lpl::editor::CommandProcessor processor(world);

    std::string line;
    char buffer[8192];
    while (std::fgets(buffer, sizeof(buffer), stdin) != nullptr)
    {
        line = trim(buffer);
        if (line.empty() || line[0] == '#')
            continue;
        if (line == "quit" || line == "exit")
            break;

        std::string command;
        if (line == "count")
            command = R"({"cmd":"count"})";
        else if (line == "save")
            command = R"({"cmd":"save_scene"})";
        else
            command = line;

        const auto report = processor.execute(command);
        if (report.has_value())
            std::printf("%s\n", report.value().c_str());
        else
            std::printf("{\"ok\":false,\"error\":\"malformed command\"}\n");
        std::fflush(stdout);
    }

    std::printf("bye — %u entities in the world\n", lpl::editor::entityCount(world));
    return 0;
}
