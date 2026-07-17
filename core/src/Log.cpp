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

#include <lpl/core/Platform.hpp>

#if !LPL_TARGET_KERNEL
#    include <cstdio>
#endif

namespace lpl::core {

namespace {

#if !LPL_TARGET_KERNEL
// Hosted default sink. The freestanding kernel build has no stderr/cstdio, so it
// starts with no logger and log calls are no-ops until one is installed. The
// kernel entry point installs platform::kernel::KernelLogger (which writes the
// same format to the console HAL) before anything else runs; without that the
// engine is silent in-kernel.
class StderrLogger final : public ILogger {
public:
    void write(LogLevel level, std::string_view tag, std::string_view message) override
    {
        static constexpr const char *kLevelNames[] = {"DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"};
        const auto idx = static_cast<unsigned>(level);
        std::fprintf(stderr, "[%s][%.*s] %.*s\n", kLevelNames[idx], static_cast<int>(tag.size()), tag.data(),
                     static_cast<int>(message.size()), message.data());
    }
};

StderrLogger gDefaultLogger;
ILogger *gActiveLogger = &gDefaultLogger;
#else
ILogger *gActiveLogger = nullptr;
#endif

LogLevel gMinLevel = LogLevel::kInfo;

} // anonymous namespace

#if !LPL_TARGET_KERNEL
void Log::setLogger(ILogger *logger) { gActiveLogger = logger ? logger : &gDefaultLogger; }
#else
void Log::setLogger(ILogger *logger) { gActiveLogger = logger; }
#endif
void Log::setMinLevel(LogLevel level) { gMinLevel = level; }

static void dispatch(LogLevel level, std::string_view tag, std::string_view msg)
{
    if (level < gMinLevel || gActiveLogger == nullptr)
        return;
    gActiveLogger->write(level, tag, msg);
}

void Log::debug(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kDebug, tag, msg); }
void Log::info(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kInfo, tag, msg); }
void Log::warn(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kWarn, tag, msg); }
void Log::error(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kError, tag, msg); }
void Log::fatal(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kFatal, tag, msg); }

} // namespace lpl::core
