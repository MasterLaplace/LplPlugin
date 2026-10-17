#ifndef LPL_ROSETTA_BOOTSTRAP_INL
#define LPL_ROSETTA_BOOTSTRAP_INL

namespace lpl::rosetta {

constexpr const char *bootstrapLevelName(BootstrapLevel level) noexcept
{
    switch (level)
    {
    case BootstrapLevel::Calibration: return "CALIBRATION";
    case BootstrapLevel::ReadOut: return "READOUT";
    case BootstrapLevel::Primitives: return "PRIMITIVES";
    case BootstrapLevel::Machine: return "MACHINE";
    case BootstrapLevel::Payload: return "PAYLOAD";
    case BootstrapLevel::Count: break;
    }
    return "?";
}

} // namespace lpl::rosetta

#endif // LPL_ROSETTA_BOOTSTRAP_INL
