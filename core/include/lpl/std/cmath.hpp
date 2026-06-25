/**
 * @file cmath.hpp
 * @brief Portable math-function umbrella. Hosted: lpl::pmr::<fn> aliases std::<fn>.
 *        Kernel: constexpr wrappers over the compiler builtins (__builtin_*), so
 *        NO libm dependency and the values are fixed by the compiler.
 *
 * Determinism (HARD contract): authoritative state is Fixed32 and must use the
 * CORDIC LUTs, never these transcendental functions at runtime. These exist for
 * (a) the COMPILE-TIME CORDIC table generation (constexpr atan, folded by the
 * compiler — must be validated bit-identical across the host gcc and kernel gcc)
 * and (b) non-authoritative float paths (e.g. statistics RMS/stddev). sqrt maps
 * to the hardware sqrtsd/sqrtss under -msse2, which is IEEE-exact and therefore
 * deterministic; sin/cos here are NOT for authoritative trig — use CORDIC.
 *
 * Use lpl::pmr::sqrt / atan / sin / cos at call sites instead of std::.
 */
#pragma once

#ifndef LPL_STD_CMATH_HPP
#    define LPL_STD_CMATH_HPP

#    include <lpl/core/Platform.hpp>

#    if LPL_TARGET_KERNEL
namespace lpl::pmr {

[[nodiscard]] constexpr double sqrt(double x) noexcept { return __builtin_sqrt(x); }
[[nodiscard]] constexpr float sqrt(float x) noexcept { return __builtin_sqrtf(x); }

[[nodiscard]] constexpr double atan(double x) noexcept { return __builtin_atan(x); }
[[nodiscard]] constexpr float atan(float x) noexcept { return __builtin_atanf(x); }

[[nodiscard]] constexpr double atan2(double y, double x) noexcept { return __builtin_atan2(y, x); }

[[nodiscard]] constexpr double sin(double x) noexcept { return __builtin_sin(x); }
[[nodiscard]] constexpr float sin(float x) noexcept { return __builtin_sinf(x); }

[[nodiscard]] constexpr double cos(double x) noexcept { return __builtin_cos(x); }
[[nodiscard]] constexpr float cos(float x) noexcept { return __builtin_cosf(x); }

[[nodiscard]] constexpr double pow(double base, double exp) noexcept { return __builtin_pow(base, exp); }

[[nodiscard]] constexpr double floor(double x) noexcept { return __builtin_floor(x); }
[[nodiscard]] constexpr double ceil(double x) noexcept { return __builtin_ceil(x); }

[[nodiscard]] constexpr double fabs(double x) noexcept { return __builtin_fabs(x); }
[[nodiscard]] constexpr float fabs(float x) noexcept { return __builtin_fabsf(x); }

} // namespace lpl::pmr
#    else
#        include <cmath>
namespace lpl::pmr {
using ::std::atan;
using ::std::atan2;
using ::std::ceil;
using ::std::cos;
using ::std::fabs;
using ::std::floor;
using ::std::pow;
using ::std::sin;
using ::std::sqrt;
}
#    endif

#endif // LPL_STD_CMATH_HPP
