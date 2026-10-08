#include <lpl/bench/JsonRows.hpp>

#include <lpl/bench/EnergyMeter.hpp>

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iterator>
#include <utility>

namespace lpl::bench {

namespace {

/** Bytes below this one are control characters, which a JSON string must escape. */
constexpr unsigned char kFirstPrintableByte = 0x20u;

/**
 * @brief Appends one byte of a string to @p out, escaped as JSON requires.
 * @param out       Text being built.
 * @param character The byte.
 */
void appendEscaped(std::string &out, char character)
{
    switch (character)
    {
    case '"': out += "\\\""; return;
    case '\\': out += "\\\\"; return;
    case '\b': out += "\\b"; return;
    case '\f': out += "\\f"; return;
    case '\n': out += "\\n"; return;
    case '\r': out += "\\r"; return;
    case '\t': out += "\\t"; return;
    default: break;
    }

    const auto byte = static_cast<unsigned char>(character);
    if (byte >= kFirstPrintableByte)
    {
        out += character;
        return;
    }
    char escape[sizeof("\\u0000")];
    std::snprintf(escape, sizeof(escape), "\\u%04x", static_cast<unsigned int>(byte));
    out += escape;
}

/**
 * @brief A JSON string holding @p text.
 * @param text UTF-8 text.
 * @return The quoted, escaped string.
 */
std::string jsonString(std::string_view text)
{
    std::string out;
    out.reserve(text.size() + 2u);
    out += '"';
    for (const char character : text)
        appendEscaped(out, character);
    out += '"';
    return out;
}

/**
 * @brief A JSON number holding @p value, in the shortest form that reads back to it.
 * @param value The number.
 * @return The number, or null when it is not finite.
 */
std::string jsonNumber(core::f64 value)
{
    if (!std::isfinite(value))
        return "null";
    char buffer[32];
    const std::to_chars_result written = std::to_chars(std::begin(buffer), std::end(buffer), value);
    return std::string(std::begin(buffer), written.ptr);
}

/**
 * @brief The energy of a row: its figure, or null.
 * @param result The measurement.
 * @return The JSON value.
 */
std::string energyFigure(const Result &result)
{
    return result.microjoulesPerRep ? jsonNumber(*result.microjoulesPerRep) : std::string{"null"};
}

/**
 * @brief Why a row has no energy: null beside a figure, else the reason's name.
 * @param result The measurement.
 * @return The JSON value.
 */
std::string energyAbsenceReason(const Result &result)
{
    return result.microjoulesPerRep ? std::string{"null"} :
                                      jsonString(energyAbsenceName(result.microjoulesPerRep.error()));
}

} // namespace

std::string formatJsonRow(std::string_view label, const Result &result, const SystemInfo &system)
{
    const std::array<std::string, kJsonRowFields.size()> values{
        std::to_string(kJsonRowSchema),   jsonString(label),
        jsonNumber(result.medianNs),      jsonNumber(coefficientOfVariationPercent(result)),
        jsonNumber(result.minNs),         jsonNumber(result.p99Ns),
        std::to_string(result.samples),   energyFigure(result),
        energyAbsenceReason(result),      jsonString(system.commit),
        jsonString(system.buildConfig),   jsonString(system.compiler),
        jsonString(machineClass(system)),
    };

    std::string row = "{";
    for (core::usize field = 0u; field < kJsonRowFields.size(); ++field)
    {
        if (field != 0u)
            row += ',';
        row += jsonString(kJsonRowFields[field].key);
        row += ':';
        row += values[field];
    }
    row += '}';
    return row;
}

std::expected<JsonRowFile, std::string> JsonRowFile::create(const std::string &path, SystemInfo system)
{
    std::FILE *file = std::fopen(path.c_str(), "w");
    if (file == nullptr)
        return std::unexpected{"cannot open '" + path + "' for writing: " + std::strerror(errno)};
    return JsonRowFile{file, path, std::move(system)};
}

JsonRowFile::JsonRowFile(std::FILE *file, std::string path, SystemInfo system)
    : _file{file}, _path{std::move(path)}, _system{std::move(system)}
{
}

void JsonRowFile::append(std::string_view label, const Result &result)
{
    if (!_labels.emplace(label).second)
    {
        recordFirstFailure("the label '" + std::string{label} + "' was measured twice: '" + _path +
                           "' keeps its first row only, since rows are paired by label");
        return;
    }

    const std::string row = formatJsonRow(label, result, _system) + '\n';
    if (std::fputs(row.c_str(), _file.get()) == EOF || std::fflush(_file.get()) != 0)
    {
        recordFirstFailure("writing the row '" + std::string{label} + "' to '" + _path +
                           "' failed: " + std::strerror(errno));
        return;
    }
    ++_rowsWritten;
}

void JsonRowFile::recordFirstFailure(std::string message)
{
    if (!_failure)
        _failure = std::move(message);
}

} // namespace lpl::bench
