#ifndef LPL_INPUT_INPUTMANAGER_INL
#define LPL_INPUT_INPUTMANAGER_INL

namespace lpl::input {

inline void PerEntityInput::setKey(core::u16 key, bool pressed) noexcept
{
    if (key < kMaxKeys)
        keys[key] = pressed;
}

inline void PerEntityInput::setAxis(core::u8 axisId, float value) noexcept
{
    if (axisId < kMaxAxes)
        axes[axisId] = value;
}

} // namespace lpl::input

#endif // LPL_INPUT_INPUTMANAGER_INL
