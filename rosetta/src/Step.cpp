/**
 * @file Step.cpp
 * @brief One instruction of the minimal ISA, executed.
 *
 * Plain C++ on purpose. The whole argument of the minimal instruction set is that
 * a stranger with none of our tools can re-implement it from a drawing; writing our
 * own interpreter in assembly would say the opposite about how simple it is. If
 * this dispatch grows past a page, the instruction set is too rich for its purpose
 * and should be cut rather than optimised.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/rosetta/Interpreter.hpp>

namespace lpl::rosetta {

// TODO(lot 9): implementation.

} // namespace lpl::rosetta
