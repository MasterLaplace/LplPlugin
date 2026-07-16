/**
 * @file Json.hpp
 * @brief Minimal, exception-free JSON value + recursive-descent parser.
 *
 * The small JSON layer the editor module speaks: it backs both the `.lplscene`
 * serializer and the command processor, so there is a single implementation. It
 * is deliberately tiny — no schema validation, no number-format guarantees — just
 * enough to read the data-driven documents an editor UI (or a future AI bridge)
 * exchanges. Values live in @c lpl::editor::detail; parse a document with
 * @c detail::parse, then walk it with @c JVal::find / @c JVal::numOr.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_EDITOR_JSON_HPP
#    define LPL_EDITOR_JSON_HPP

#    include <string>
#    include <string_view>
#    include <utility>
#    include <vector>

namespace lpl::editor::detail {

/**
 * @struct JVal
 * @brief A parsed JSON value (null / bool / number / string / array / object).
 */
struct JVal {
    enum class T {
        Null,
        Bool,
        Num,
        Str,
        Arr,
        Obj
    };
    T t{T::Null};
    bool b{false};
    double num{0.0};
    std::string str;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;

    /// Member value for @p key on an object, or nullptr if absent.
    [[nodiscard]] const JVal *find(std::string_view key) const
    {
        for (const auto &kv : obj)
            if (kv.first == key)
                return &kv.second;
        return nullptr;
    }

    /// Numeric field @p key, or @p fallback if absent / not a number.
    [[nodiscard]] double numOr(std::string_view key, double fallback) const
    {
        const JVal *v = find(key);
        return (v != nullptr && v->t == T::Num) ? v->num : fallback;
    }
};

/**
 * @struct Parser
 * @brief Single-pass recursive-descent JSON parser over a @c string_view.
 *
 * Construct over the source, call @c value() for the root; @c ok reports whether
 * parsing stayed well-formed. Never throws.
 */
struct Parser {
    std::string_view s;
    std::size_t i{0};
    bool ok{true};

    void ws();
    bool eat(char c);
    JVal value();
    std::string string();
    JVal number();
    JVal boolean();
    JVal array();
    JVal object();
};

/// Parses @p text into a JSON value; sets @p ok (if given) to the parse status.
[[nodiscard]] JVal parse(std::string_view text, bool *ok = nullptr);

} // namespace lpl::editor::detail

#endif // LPL_EDITOR_JSON_HPP
