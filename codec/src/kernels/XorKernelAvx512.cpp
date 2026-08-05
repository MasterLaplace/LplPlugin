/**
 * @file XorKernelAvx512.cpp
 * @brief Row XOR over GF(2) — 512-bit lanes.
 *
 * The widest XOR the machine offers, and the reason GF(2) elimination collapses
 * from O(N^3) to O(N^3 / 512). Unrolled so several independent chains keep the
 * vector ports fed while loads retire; rows are 64-byte aligned so no load ever
 * straddles a cache line.
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
