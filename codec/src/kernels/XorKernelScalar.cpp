/**
 * @file XorKernelScalar.cpp
 * @brief Row XOR over GF(2) — plain 64-bit words — the reference.
 *
 * The definition of correct. Every other kernel here exists only because it
 * reproduces this one exactly, and the parity fold is what proves it.
 *
 * Written with intrinsics rather than hand-written assembly, deliberately. The
 * generated instructions are the same; what changes is that the compiler keeps the
 * register allocation, the calling convention and the stack alignment correct, and
 * that the next reader can follow the algorithm. Assembly is reserved for the cases
 * where it is genuinely unavoidable — privileged instructions, register state a C
 * expression cannot name — which this is not.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/XorKernel.hpp>

namespace lpl::codec {

// TODO(lot 3): implementation.

} // namespace lpl::codec
