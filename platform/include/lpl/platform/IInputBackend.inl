#ifndef LPL_PLATFORM_IINPUTBACKEND_INL
#define LPL_PLATFORM_IINPUTBACKEND_INL

namespace lpl::platform {

inline bool IInputBackend::tryPopPointerMotion(core::i32 &outDeltaX, core::i32 &outDeltaY, core::u32 &outButtons)
{
    outDeltaX = 0;
    outDeltaY = 0;
    outButtons = 0u;
    return false;
}

inline bool IInputBackend::isKeyHeld(char character) const noexcept
{
    (void) character;
    return false;
}

} // namespace lpl::platform

#endif // LPL_PLATFORM_IINPUTBACKEND_INL
