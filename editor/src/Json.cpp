/**
 * @file Json.cpp
 * @brief Implementation of the editor's minimal JSON parser.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <lpl/editor/Json.hpp>

#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace lpl::editor::detail {

void Parser::ws()
{
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        ++i;
}

bool Parser::eat(char c)
{
    ws();
    if (i < s.size() && s[i] == c)
    {
        ++i;
        return true;
    }
    return false;
}

JVal Parser::value()
{
    ws();
    if (i >= s.size())
    {
        ok = false;
        return {};
    }
    const char c = s[i];
    if (c == '{')
        return object();
    if (c == '[')
        return array();
    if (c == '"')
    {
        JVal v;
        v.t = JVal::T::Str;
        v.str = string();
        return v;
    }
    if (c == 't' || c == 'f')
        return boolean();
    if (c == 'n')
    {
        i += 4; // "null"
        return {};
    }
    return number();
}

std::string Parser::string()
{
    std::string out;
    if (!eat('"'))
    {
        ok = false;
        return out;
    }
    while (i < s.size() && s[i] != '"')
    {
        char c = s[i++];
        if (c == '\\' && i < s.size())
        {
            const char e = s[i++];
            switch (e)
            {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            default: out += e; break;
            }
        }
        else
        {
            out += c;
        }
    }
    if (i < s.size() && s[i] == '"')
        ++i;
    else
        ok = false;
    return out;
}

JVal Parser::number()
{
    const std::size_t start = i;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' || s[i] == '+' ||
                            s[i] == '.' || s[i] == 'e' || s[i] == 'E'))
        ++i;
    JVal v;
    v.t = JVal::T::Num;
    const std::string tok{s.substr(start, i - start)};
    v.num = std::strtod(tok.c_str(), nullptr);
    return v;
}

JVal Parser::boolean()
{
    JVal v;
    v.t = JVal::T::Bool;
    if (s.compare(i, 4, "true") == 0)
    {
        v.b = true;
        i += 4;
    }
    else if (s.compare(i, 5, "false") == 0)
    {
        v.b = false;
        i += 5;
    }
    else
    {
        ok = false;
    }
    return v;
}

JVal Parser::array()
{
    JVal v;
    v.t = JVal::T::Arr;
    eat('[');
    if (eat(']'))
        return v;
    while (ok)
    {
        v.arr.push_back(value());
        if (eat(','))
            continue;
        if (eat(']'))
            break;
        ok = false;
    }
    return v;
}

JVal Parser::object()
{
    JVal v;
    v.t = JVal::T::Obj;
    eat('{');
    if (eat('}'))
        return v;
    while (ok)
    {
        ws();
        std::string k = string();
        if (!eat(':'))
        {
            ok = false;
            break;
        }
        v.obj.emplace_back(std::move(k), value());
        if (eat(','))
            continue;
        if (eat('}'))
            break;
        ok = false;
    }
    return v;
}

JVal parse(std::string_view text, bool *ok)
{
    Parser parser{text, 0, true};
    JVal root = parser.value();
    if (ok != nullptr)
        *ok = parser.ok;
    return root;
}

namespace {

void emitEscaped(std::string &out, std::string_view text)
{
    for (const char c : text)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
}

} // namespace

std::string emit(const JVal &value)
{
    switch (value.t)
    {
    case JVal::T::Null: return "null";
    case JVal::T::Bool: return value.b ? "true" : "false";
    case JVal::T::Num: {
        char buffer[40];
        // %.17g is the shortest precision that round-trips every double
        // exactly; anything shorter can shift a parameter and therefore the
        // world a replayed command rebuilds. Integral values are printed
        // without a decimal point so a journal stays readable.
        if (value.num == static_cast<double>(static_cast<long long>(value.num)))
            std::snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(value.num));
        else
            std::snprintf(buffer, sizeof(buffer), "%.17g", value.num);
        return std::string{buffer};
    }
    case JVal::T::Str: {
        std::string out = "\"";
        emitEscaped(out, value.str);
        out += "\"";
        return out;
    }
    case JVal::T::Arr: {
        std::string out = "[";
        for (std::size_t i = 0; i < value.arr.size(); ++i)
        {
            if (i != 0)
                out += ",";
            out += emit(value.arr[i]);
        }
        out += "]";
        return out;
    }
    case JVal::T::Obj: {
        std::string out = "{";
        for (std::size_t i = 0; i < value.obj.size(); ++i)
        {
            if (i != 0)
                out += ",";
            out += "\"";
            emitEscaped(out, value.obj[i].first);
            out += "\":";
            out += emit(value.obj[i].second);
        }
        out += "}";
        return out;
    }
    }
    return "null";
}

void overlay(JVal &target, const JVal &patch)
{
    if (target.t != JVal::T::Obj || patch.t != JVal::T::Obj)
    {
        target = patch;
        return;
    }
    for (const auto &member : patch.obj)
    {
        JVal *existing = nullptr;
        for (auto &candidate : target.obj)
            if (candidate.first == member.first)
            {
                existing = &candidate.second;
                break;
            }
        if (existing == nullptr)
            target.obj.push_back(member);
        else
            overlay(*existing, member.second);
    }
}

} // namespace lpl::editor::detail
