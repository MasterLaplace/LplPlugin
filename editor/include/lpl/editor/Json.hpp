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

    /**
     * @brief Member value for @p key on an object, or nullptr if absent.
     * @param key The key to search for.
     * @return A pointer to the value if found, nullptr otherwise.
     */
    [[nodiscard]] const JVal *find(std::string_view key) const
    {
        for (const auto &kv : obj)
            if (kv.first == key)
                return &kv.second;
        return nullptr;
    }

    /**
     * @brief Numeric field @p key, or @p fallback if absent / not a number.
     * @param key The key to search for.
     * @param fallback The value to return if the key is absent or not a number.
     * @return The numeric value if found and valid, otherwise the fallback.
     */
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

/**
 * @brief Parses @p text into a JSON value; sets @p ok (if given) to the parse status.
 * @param text The JSON text to parse.
 * @param ok Optional pointer to a boolean that will be set to true if parsing succeeded, false otherwise.
 * @return The parsed JSON value.
 */
[[nodiscard]] JVal parse(std::string_view text, bool *ok = nullptr);

/**
 * @brief Re-serialises a parsed value back to JSON text.
 *
 * Needed wherever a sub-document has to survive on its own — a journal entry
 * replayed later, a template instantiated elsewhere — because the parser keeps
 * no byte spans into the source. Numbers print with %.17g so a value that made
 * the round trip rebuilds the same world it described.
 */
[[nodiscard]] std::string emit(const JVal &value);

/**
 * @brief Overlays @p patch onto @p target, RECURSIVELY.
 *
 * A key the patch names replaces the target's value, except when both sides are
 * objects — then it descends. So `{"terrain":{"octaves":6}}` laid over
 * `{"seed":7,"terrain":{"amplitude":3,"octaves":1}}` keeps the seed AND the
 * amplitude, and changes only the octaves.
 *
 * Not to be confused with the field-level overlay SceneSerializer uses for
 * template inheritance, which stops at one level on purpose: a component's
 * fields are flat, and descending into them would let a partial override leave a
 * component half-inherited. Same word, two depths, two reasons — unifying them
 * would silently change what a `$use` chain means.
 */
void overlay(JVal &target, const JVal &patch);

} // namespace lpl::editor::detail

#endif // LPL_EDITOR_JSON_HPP
