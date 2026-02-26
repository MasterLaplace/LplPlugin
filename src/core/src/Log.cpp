/**
 * @file Log.cpp
 * @brief Default ILogger implementation writing to stderr.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#include "lpl/core/Log.hpp"

#include <cstdio>

namespace lpl::core {

namespace {

class StderrLogger final : public ILogger {
public:
    void write(LogLevel level, std::string_view tag, std::string_view message) override
    {
        static constexpr const char *kLevelNames[] = {
            "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"
        };
        const auto idx = static_cast<unsigned>(level);
        std::fprintf(
            stderr,
            "[%s][%.*s] %.*s\n",
            kLevelNames[idx],
            static_cast<int>(tag.size()), tag.data(),
            static_cast<int>(message.size()), message.data()
        );
    }
};

StderrLogger  gDefaultLogger;
ILogger      *gActiveLogger  = &gDefaultLogger;
LogLevel      gMinLevel      = LogLevel::kInfo;

} // anonymous namespace

void Log::setLogger(ILogger *logger)  { gActiveLogger = logger ? logger : &gDefaultLogger; }
void Log::setMinLevel(LogLevel level) { gMinLevel = level; }

static void dispatch(LogLevel level, std::string_view tag, std::string_view msg)
{
    if (level < gMinLevel)
        return;
    gActiveLogger->write(level, tag, msg);
}

void Log::debug(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kDebug, tag, msg); }
void Log::info (std::string_view tag, std::string_view msg) { dispatch(LogLevel::kInfo,  tag, msg); }
void Log::warn (std::string_view tag, std::string_view msg) { dispatch(LogLevel::kWarn,  tag, msg); }
void Log::error(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kError, tag, msg); }
void Log::fatal(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kFatal, tag, msg); }

} // namespace lpl::core
