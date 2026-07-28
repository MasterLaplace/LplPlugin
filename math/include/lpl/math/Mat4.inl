/**
 * @file Mat4.inl
 * @brief Inline implementations of Mat4 template methods.
 *
 * @note This file is automatically included at the end of Mat4.hpp.
 *       Do not include it directly.
 */

namespace lpl::math {

template <core::Arithmetic T> constexpr T &Mat4<T>::operator()(core::u32 row, core::u32 col)
{
    return m[col * 4 + row];
}

template <core::Arithmetic T> constexpr T Mat4<T>::operator()(core::u32 row, core::u32 col) const
{
    return m[col * 4 + row];
}

template <core::Arithmetic T> constexpr Mat4<T> Mat4<T>::operator*(Mat4 rhs) const
{
    Mat4 result;
    for (core::u32 col = 0; col < 4; ++col)
    {
        for (core::u32 row = 0; row < 4; ++row)
        {
            T sum{};
            for (core::u32 k = 0; k < 4; ++k)
            {
                sum = sum + (*this)(row, k) * rhs(k, col);
            }
            result(row, col) = sum;
        }
    }
    return result;
}

template <core::Arithmetic T> constexpr Vec3<T> Mat4<T>::transformPoint(Vec3<T> p) const
{
    return Vec3<T>((*this)(0, 0) * p.x + (*this)(0, 1) * p.y + (*this)(0, 2) * p.z + (*this)(0, 3),
                   (*this)(1, 0) * p.x + (*this)(1, 1) * p.y + (*this)(1, 2) * p.z + (*this)(1, 3),
                   (*this)(2, 0) * p.x + (*this)(2, 1) * p.y + (*this)(2, 2) * p.z + (*this)(2, 3));
}

template <core::Arithmetic T> constexpr Vec3<T> Mat4<T>::transformDirection(Vec3<T> d) const
{
    return Vec3<T>((*this)(0, 0) * d.x + (*this)(0, 1) * d.y + (*this)(0, 2) * d.z,
                   (*this)(1, 0) * d.x + (*this)(1, 1) * d.y + (*this)(1, 2) * d.z,
                   (*this)(2, 0) * d.x + (*this)(2, 1) * d.y + (*this)(2, 2) * d.z);
}

template <core::Arithmetic T> constexpr Mat4<T> Mat4<T>::identity()
{
    Mat4 r;
    r.m.fill(T{});
    r(0, 0) = T{1};
    r(1, 1) = T{1};
    r(2, 2) = T{1};
    r(3, 3) = T{1};
    return r;
}

template <core::Arithmetic T> constexpr Mat4<T> Mat4<T>::translate(Vec3<T> offset)
{
    auto r = identity();
    r(0, 3) = offset.x;
    r(1, 3) = offset.y;
    r(2, 3) = offset.z;
    return r;
}

template <core::Arithmetic T> constexpr Mat4<T> Mat4<T>::scale(Vec3<T> s)
{
    auto r = identity();
    r(0, 0) = s.x;
    r(1, 1) = s.y;
    r(2, 2) = s.z;
    return r;
}

template <core::Arithmetic T> constexpr Mat4<T> Mat4<T>::fromQuat(Quat<T> q)
{
    auto r = identity();
    const T xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const T xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const T wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    const T one{1}, two{2};
    (void) one;
    r(0, 0) = one - two * (yy + zz);
    r(0, 1) = two * (xy - wz);
    r(0, 2) = two * (xz + wy);
    r(1, 0) = two * (xy + wz);
    r(1, 1) = one - two * (xx + zz);
    r(1, 2) = two * (yz - wx);
    r(2, 0) = two * (xz - wy);
    r(2, 1) = two * (yz + wx);
    r(2, 2) = one - two * (xx + yy);
    return r;
}

template <core::Arithmetic T> Mat4<T> Mat4<T>::perspective(T fovRad, T aspect, T nearPlane, T farPlane)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        // tan(fov/2) comes from CORDIC, not from libm.
        //
        // The determinism contract forbids libm transcendentals in engine code
        // linked into the kernel, and tan is the one that has no way around it:
        // sqrt folds to the SSE instruction and is IEEE-exact on both targets,
        // but tan always becomes a call — resolved to glibc on the host and to
        // the kernel's own Taylor-series tanf in ring 0. Those two disagree in
        // the low bits, so every projection matrix, and every image folded
        // through one, would differ between the oracle and the kernel.
        //
        // CORDIC is shifts and adds over Fixed32, identical everywhere. The
        // divide and the widening below are SSE operations, which are
        // IEEE-defined and therefore also identical on both targets.
        Fixed32 halfSin{};
        Fixed32 halfCos{};
        Cordic::sincos(Fixed32::fromFloat(static_cast<float>(fovRad) * 0.5f), halfSin, halfCos);
        const T tanHalf = static_cast<T>(halfSin.toFloat()) / static_cast<T>(halfCos.toFloat());
        Mat4 r;
        r.m.fill(T{});
        r(0, 0) = T(1) / (aspect * tanHalf);
        r(1, 1) = T(1) / tanHalf;
        r(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        r(2, 3) = -(T(2) * farPlane * nearPlane) / (farPlane - nearPlane);
        r(3, 2) = -T(1);
        return r;
    }
    else
    {
        return identity();
    }
}

template <core::Arithmetic T> Mat4<T> Mat4<T>::lookAt(Vec3<T> eye, Vec3<T> target, Vec3<T> up)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        const auto f = (target - eye).normalize();
        const auto s = f.cross(up).normalize();
        const auto u = s.cross(f);

        auto r = identity();
        r(0, 0) = s.x;
        r(0, 1) = s.y;
        r(0, 2) = s.z;
        r(1, 0) = u.x;
        r(1, 1) = u.y;
        r(1, 2) = u.z;
        r(2, 0) = -f.x;
        r(2, 1) = -f.y;
        r(2, 2) = -f.z;
        r(0, 3) = -(s.dot(eye));
        r(1, 3) = -(u.dot(eye));
        r(2, 3) = f.dot(eye);
        return r;
    }
    else
    {
        (void) eye;
        (void) target;
        (void) up;
        return identity();
    }
}

} // namespace lpl::math
