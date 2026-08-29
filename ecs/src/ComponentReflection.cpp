/**
 * @file ComponentReflection.cpp
 * @brief A translation unit whose only job is to make the reflection assertions run.
 *
 * @warning **It exists because a check nobody compiles is not a check.** `ComponentReflection.hpp`
 * asserts that every schema tiles its component exactly -- that the offsets and widths the editor
 * and the baker read a component THROUGH agree with the struct every system reads it AS. A
 * `static_assert` in a header only fires in translation units that include it, and the header was
 * included by `agent/`, `editor/` and one app: none of them is the module that owns the
 * components. Measured, before this file: an offset deliberately broken from 8 to 12 compiled
 * clean.
 *
 * Compiling it here means the agreement is verified wherever `lpl-ecs` is built, which is every
 * target including ring 0 -- and it fails at build time rather than as a component that
 * serialises to something no system ever wrote.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecs/ComponentReflection.hpp>

namespace lpl::ecs {

// Nothing to define. The assertions in the header are the payload, and they run because this
// file includes it.

} // namespace lpl::ecs
