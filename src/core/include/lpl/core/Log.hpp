/**
 * @file Log.hpp
 * @brief Minimal logging façade with compile-time severity filtering.
 *
 * Provides a static Log class backed by an injectable ILogger interface.
 * The default implementation writes to stderr.  A custom logger can be
 * installed via Log::setLogger() at engine startup.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_LOG_HPP
    #define LPL_CORE_LOG_HPP

    #include "Types.hpp"

    #include <string_view>

namespace lpl::core {

/**
 * @brief Severity levels for log messages.
 */
enum class LogLevel : u8 {
    kDebug = 0,
    kInfo,
    kWarn,
    kError,
    kFatal
};

/**
 * @brief Abstract sink for log messages.
 */
class ILogger {
public:
    virtual ~ILogger() = default;

    /**
     * @brief Write a log entry.
     * @param level   Severity.
     * @param tag     Subsystem tag (e.g. "NET", "ECS", "GPU").
     * @param message Formatted message body.
     */
    virtual void write(LogLevel level, std::string_view tag, std::string_view message) = 0;
};

/**
 * @brief Static logging façade used throughout the engine.
 *
 * All methods are thread-safe provided the installed ILogger is thread-safe.
 */
class Log final {
public:
    Log() = delete;

    static void setLogger(ILogger *logger);
    static void setMinLevel(LogLevel level);

    static void debug(std::string_view tag, std::string_view msg);
    static void info (std::string_view tag, std::string_view msg);
    static void warn (std::string_view tag, std::string_view msg);
    static void error(std::string_view tag, std::string_view msg);
    static void fatal(std::string_view tag, std::string_view msg);

    static void debug(std::string_view msg) { debug("lpl", msg); }
    static void info (std::string_view msg) { info ("lpl", msg); }
    static void warn (std::string_view msg) { warn ("lpl", msg); }
    static void error(std::string_view msg) { error("lpl", msg); }
    static void fatal(std::string_view msg) { fatal("lpl", msg); }
};

} // namespace lpl::core

#endif // LPL_CORE_LOG_HPP
