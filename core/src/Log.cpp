#include "lpl/core/Log.hpp"

#include <lpl/core/Platform.hpp>

#if !LPL_TARGET_KERNEL
#    include <cstdio>
#endif

namespace lpl::core {

namespace {

#if !LPL_TARGET_KERNEL
/** @brief The hosted default sink: writes each message to stderr. */
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

} // namespace

static void dispatch(LogLevel level, std::string_view tag, std::string_view msg)
{
    if (level < gMinLevel || gActiveLogger == nullptr)
        return;
    gActiveLogger->write(level, tag, msg);
}

#if !LPL_TARGET_KERNEL
void Log::setLogger(ILogger *logger) { gActiveLogger = logger ? logger : &gDefaultLogger; }
#else
void Log::setLogger(ILogger *logger) { gActiveLogger = logger; }
#endif
void Log::setMinLevel(LogLevel level) { gMinLevel = level; }

void Log::debug(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kDebug, tag, msg); }
void Log::info(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kInfo, tag, msg); }
void Log::warn(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kWarn, tag, msg); }
void Log::error(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kError, tag, msg); }
void Log::fatal(std::string_view tag, std::string_view msg) { dispatch(LogLevel::kFatal, tag, msg); }

} // namespace lpl::core
