/**
 * @file JsonRows.hpp
 * @brief One JSON row per measurement, so that two runs can be compared row by row and a
 *        series of commits kept, instead of results lost with the terminal.
 *
 * A file of rows is JSON Lines: one object per line, each one whole, so files are
 * concatenated into a series and read by any JSON reader, line by line. Each row carries
 * the label that pairs it with the same measurement in another file, its statistics, its
 * energy or the reason it has none, and what it must be read against: the commit, the
 * build, the compiler and the machine class. @ref kJsonRowFields lists the fields in the
 * order they are written, and `lpl-benchmark --help` prints them.
 *
 * Strings are written as UTF-8, with quotes, backslashes and control characters escaped.
 * A number that is not finite is written null, the only value JSON has for it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-10-08
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_BENCH_JSONROWS_HPP
#    define LPL_BENCH_JSONROWS_HPP

#    include <lpl/bench/Harness.hpp>
#    include <lpl/bench/SystemInfo.hpp>
#    include <lpl/core/Types.hpp>

#    include <array>
#    include <cstdio>
#    include <expected>
#    include <functional>
#    include <memory>
#    include <optional>
#    include <set>
#    include <string>
#    include <string_view>

namespace lpl::bench {

/** Version of the row layout, written in every row; a reader refuses one it does not know. */
inline constexpr core::u32 kJsonRowSchema = 1u;

/** One field of a JSON row: its key, and what its value means. */
struct JsonRowField {
    std::string_view key;     ///< The key, as written.
    std::string_view meaning; ///< What the value is, with its unit or its closed set of values.
};

/** The fields of a JSON row, in the order every row writes them. */
inline constexpr std::array<JsonRowField, 13> kJsonRowFields{
    {
     {"schema", "1, the version of this layout"},
     {"label", "name of the measurement, unique in a file: rows of two files pair by it"},
     {"median_ns", "median time of one repetition, in nanoseconds"},
     {"cv_percent", "standard deviation over mean of the repetition times, in percent, or null for a mean of zero"},
     {"min_ns", "fastest repetition, in nanoseconds"},
     {"p99_ns", "99th-percentile repetition, in nanoseconds"},
     {"n", "number of timed repetitions"},
     {"energy_uj_per_rep", "package energy of one repetition, in microjoules, or null"},
     {"energy_absent_reason", "null beside an energy figure, else why there is none: denied, unreadable, absent, "
                                 "no-repetition, short-window, read-failed or ambiguous-wrap"},
     {"commit", "short commit lpl-benchmark was built from, -dirty for a modified tree, or unknown"},
     {"build", "build configuration: Release, Profile, Debug or Unknown"},
     {"compiler", "compiler name and version"},
     {"machine_class", "platform, processor, logical cores and hypervisor or bare metal: the key that groups "
                          "runs of one kind of machine"},
     }
};

/**
 * @brief Writes one measurement as a JSON object on one line.
 * @param label  Name of the measurement.
 * @param result Its statistics and its energy, or the reason it has none.
 * @param system The context the run is read against: commit, build, compiler, machine class.
 * @return The object, without a line break.
 */
[[nodiscard]] std::string formatJsonRow(std::string_view label, const Result &result, const SystemInfo &system);

/** @brief A file of JSON rows, written one row at a time and flushed after each. */
class JsonRowFile final {
public:
    /**
     * @brief Creates @p path, or empties it when it exists.
     * @param path   File to write the rows to.
     * @param system Context every row carries.
     * @return The file, or a message naming @p path and why it could not be opened.
     */
    [[nodiscard]] static std::expected<JsonRowFile, std::string> create(const std::string &path, SystemInfo system);

    /**
     * @brief Writes one row and flushes it, so an interrupted run keeps every row before.
     *
     * @details A label already written is refused, since a reader pairs rows by label; the
     *          refusal is recorded as the failure when it is the first. The file keeps every
     *          row whose label it had not seen.
     *
     * @param label  Name of the measurement, unique in the file.
     * @param result Its statistics.
     */
    void append(std::string_view label, const Result &result);

    /**
     * @brief The first row that is missing from the file, and why.
     * @return A message naming the path and the label or the error, or nothing when every row
     *         appended is in the file.
     */
    [[nodiscard]] const std::optional<std::string> &failure() const noexcept { return _failure; }

    /**
     * @brief How many rows the file holds.
     * @return The count.
     */
    [[nodiscard]] core::usize rowCount() const noexcept { return _rowsWritten; }

    /**
     * @brief The file the rows go to.
     * @return Its path.
     */
    [[nodiscard]] const std::string &path() const noexcept { return _path; }

private:
    struct FileCloser {
        void operator()(std::FILE *file) const noexcept { std::fclose(file); }
    };

    JsonRowFile(std::FILE *file, std::string path, SystemInfo system);

    void recordFirstFailure(std::string message);

    std::unique_ptr<std::FILE, FileCloser> _file;
    std::string _path;
    SystemInfo _system;
    std::set<std::string, std::less<>> _labels;
    core::usize _rowsWritten = 0u;
    std::optional<std::string> _failure;
};

} // namespace lpl::bench

#endif // LPL_BENCH_JSONROWS_HPP
