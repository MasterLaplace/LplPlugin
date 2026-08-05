/**
 * @file Parity.hpp
 * @brief A signature over the surface an intelligence is offered.
 *
 * NOT a host↔kernel parity gate, and the distinction matters enough to state at
 * the top of the file. Inference runs in ring 3; `agent/` dispatches through
 * `editor::CommandProcessor`, which is on the WRITER side of the reader/writer
 * line that keeps ring 0 free of tooling. Nothing in this module is linked into
 * the kernel, so there is no second implementation to agree with.
 *
 * What this fold does instead is guard against drift: rename a tool, move a
 * bound, add an enum value, reorder the table, and the signature changes and a
 * test says so. It plays for the tool surface the role @c ParityPackBlob plays
 * for the cartridge — a tripwire on a format other things depend on, not a proof
 * that two arithmetics agree.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_PARITY_HPP
#    define LPL_LPL_AGENT_PARITY_HPP

#    include <lpl/agent/ToolRegistry.hpp>
#    include <lpl/core/Types.hpp>

namespace lpl::agent {

/**
 * @brief FNV-1a over the JSON-Schema and the GBNF grammar of @p registry.
 *
 * Both artefacts, not one: they are derived from the same table by two different
 * emitters, and a change that moved only one of them would be exactly the kind of
 * silent divergence this exists to catch.
 *
 * Offset basis @c 0x811C9DC5, prime @c 0x01000193 — the same constants every fold
 * in this project uses, so a signature is comparable with any other.
 */
[[nodiscard]] core::u32 foldToolSurface(const ToolRegistry &registry);

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_PARITY_HPP
