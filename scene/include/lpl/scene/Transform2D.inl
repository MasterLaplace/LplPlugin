#ifndef LPL_SCENE_TRANSFORM2D_INL
#define LPL_SCENE_TRANSFORM2D_INL

namespace lpl::scene {

inline Transform2D Transform2D::translation(math::Fixed32 x, math::Fixed32 y) noexcept
{
    Transform2D t;
    t.tx = x;
    t.ty = y;
    return t;
}

inline Transform2D Transform2D::fromTRS(math::Fixed32 x, math::Fixed32 y, math::Fixed32 angle, math::Fixed32 sx,
                                        math::Fixed32 sy) noexcept
{
    math::Fixed32 s{math::Fixed32::fromInt(0)};
    math::Fixed32 co{math::Fixed32::fromInt(0)};
    math::Cordic::sincos(angle, s, co);
    Transform2D t;
    t.a = co * sx;
    t.b = s * sx;
    t.c = -(s * sy);
    t.d = co * sy;
    t.tx = x;
    t.ty = y;
    return t;
}

inline Transform2D Transform2D::operator*(const Transform2D &inner) const noexcept
{
    Transform2D r;
    r.a = a * inner.a + c * inner.b;
    r.b = b * inner.a + d * inner.b;
    r.c = a * inner.c + c * inner.d;
    r.d = b * inner.c + d * inner.d;
    r.tx = a * inner.tx + c * inner.ty + tx;
    r.ty = b * inner.tx + d * inner.ty + ty;
    return r;
}

inline void Transform2D::apply(math::Fixed32 x, math::Fixed32 y, math::Fixed32 &outX,
                               math::Fixed32 &outY) const noexcept
{
    outX = a * x + c * y + tx;
    outY = b * x + d * y + ty;
}

} // namespace lpl::scene

#endif // LPL_SCENE_TRANSFORM2D_INL
