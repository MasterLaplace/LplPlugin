/**
 * @file Parity.cpp
 * @brief Implementation of the tool-surface signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Parity.hpp>

#include <lpl/agent/Grammar.hpp>
#include <lpl/agent/Schema.hpp>

namespace lpl::agent {

namespace {

constexpr core::u32 kFnvOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnvPrime = 0x01000193u;

core::u32 fold(core::u32 seed, std::string_view bytes)
{
    core::u32 hash = seed;
    for (const char c : bytes)
    {
        hash ^= static_cast<core::u32>(static_cast<unsigned char>(c));
        hash *= kFnvPrime;
    }
    return hash;
}

} // namespace

core::u32 foldToolSurface(const ToolRegistry &registry)
{
    core::u32 hash = fold(kFnvOffsetBasis, emitJsonSchema(registry));
    return fold(hash, emitGbnf(registry));
}

} // namespace lpl::agent
