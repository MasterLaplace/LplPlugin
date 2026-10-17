#include <lpl/agent/Grammar.hpp>

#include <lpl/ecs/ComponentReflection.hpp>

namespace lpl::agent {

namespace {

/**
 * @brief GBNF rule names accept letters, digits and dashes; tool names use underscores.
 */
std::string ruleName(std::string_view prefix, std::string_view name)
{
    std::string out{prefix};
    for (const char c : name)
        out += (c == '_') ? '-' : c;
    return out;
}

std::string ruleName(std::string_view prefix, std::string_view tool, std::string_view param)
{
    std::string out = ruleName(prefix, tool);
    out += '-';
    for (const char c : param)
        out += (c == '_') ? '-' : c;
    return out;
}

/**
 * @brief A JSON string literal as a GBNF terminal.
 */
std::string literal(std::string_view s)
{
    std::string out = "\"\\\"";
    out += s;
    out += "\\\"\"";
    return out;
}

/**
 * @brief The GBNF rule producing a value of @p kind.
 */
std::string_view valueRule(ParamKind kind)
{
    switch (kind)
    {
    case ParamKind::Number: return "number";
    case ParamKind::Integer: return "integer";
    case ParamKind::String: return "string";
    case ParamKind::Bool: return "boolean";
    case ParamKind::Array: return "array";
    case ParamKind::Object: return "object";
    }
    return "value";
}

/**
 * @brief The right-hand side of one parameter member.
 *
 * A closed set of choices goes into the rule itself: the one thing a grammar does
 * that a schema cannot is put the closed set into the sampler, so an unknown word
 * is not merely refused afterwards, it is never generated. WHICH set is
 * agent::forEachChoice's business — asking here would be the second place to
 * remember when a set is added.
 */
std::string memberBody(const ToolParam &param)
{
    std::string out = literal(param.name);
    out += " ws \":\" ws ";
    if (param.choices != DynamicEnum::None)
    {
        out += "( ";
        bool first = true;
        forEachChoice(param.choices, [&](std::string_view word) {
            if (!first)
                out += " | ";
            first = false;
            out += literal(word);
        });
        out += " )";
        return out;
    }
    out += valueRule(param.kind);
    return out;
}

} // namespace

std::string emitGbnf(const ToolRegistry &registry)
{
    std::string out;
    out += "# Generated from the tool registry — do not edit.\n";
    out += "# Regenerated every step: this grammar describes what is callable NOW.\n";
    out += "root ::= call\n\n";

    out += "ws ::= [ \\t\\n]*\n";
    out += "string ::= \"\\\"\" ( [^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt] )* \"\\\"\"\n";
    out += "integer ::= \"-\"? ( \"0\" | [1-9] [0-9]* )\n";
    out += "number ::= \"-\"? ( \"0\" | [1-9] [0-9]* ) ( \".\" [0-9]+ )? ( [eE] [-+]? [0-9]+ )?\n";
    out += "boolean ::= \"true\" | \"false\"\n";
    out += "value ::= object | array | string | number | boolean | \"null\"\n";
    out += "object ::= \"{\" ws ( string ws \":\" ws value ( ws \",\" ws string ws \":\" ws value )* )? ws \"}\"\n";
    out += "array ::= \"[\" ws ( value ( ws \",\" ws value )* )? ws \"]\"\n\n";

    out += "call ::= \"{\" ws ( \"\\\"thought\\\"\" ws \":\" ws string ws \",\" ws )? \"\\\"tool\\\"\" ws \":\" ws "
           "tool-choice ws \"}\"\n";

    out += "tool-choice ::= ";
    if (registry.tools().empty())
    {
        out += "\"\\u0000\"\n";
        return out;
    }
    bool first = true;
    for (const ToolDesc *tool : registry.tools())
    {
        if (!first)
            out += " | ";
        first = false;
        out += ruleName("tc-", tool->name);
    }
    out += "\n\n";

    for (const ToolDesc *tool : registry.tools())
    {
        const std::string args = ruleName("args-", tool->name);
        out += ruleName("tc-", tool->name);
        out += " ::= ";
        out += literal(tool->name);
        out += " ws \",\" ws \"\\\"args\\\"\" ws \":\" ws ";
        out += args;
        out += '\n';

        core::u32 required = 0u;
        for (const ToolParam &p : tool->params)
            required += p.required ? 1u : 0u;

        if (tool->params.empty())
        {
            out += args + " ::= \"{\" ws \"}\"\n\n";
            continue;
        }

        if (required != 0u)
        {
            out += args + " ::= \"{\" ws ";
            bool firstRequired = true;
            for (const ToolParam &p : tool->params)
            {
                if (!p.required)
                    continue;
                if (!firstRequired)
                    out += " ws \",\" ws ";
                firstRequired = false;
                out += ruleName("p-", tool->name, p.name);
            }
            for (const ToolParam &p : tool->params)
            {
                if (p.required)
                    continue;
                out += " ( ws \",\" ws ";
                out += ruleName("p-", tool->name, p.name);
                out += " )?";
            }
            out += " ws \"}\"\n";
        }
        else
        {
            const std::string member = ruleName("m-", tool->name);
            out += args + " ::= \"{\" ws ( " + member + " ( ws \",\" ws " + member + " )* )? ws \"}\"\n";
            out += member + " ::= ";
            bool firstMember = true;
            for (const ToolParam &p : tool->params)
            {
                if (!firstMember)
                    out += " | ";
                firstMember = false;
                out += ruleName("p-", tool->name, p.name);
            }
            out += '\n';
        }

        for (const ToolParam &p : tool->params)
        {
            out += ruleName("p-", tool->name, p.name);
            out += " ::= ";
            out += memberBody(p);
            out += '\n';
        }
        out += '\n';
    }

    return out;
}

} // namespace lpl::agent
