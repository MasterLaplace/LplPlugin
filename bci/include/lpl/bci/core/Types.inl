#ifndef LPL_BCI_CORE_TYPES_INL
#define LPL_BCI_CORE_TYPES_INL

namespace lpl::bci {

constexpr std::string_view acquisitionModeName(AcquisitionMode mode) noexcept
{
    switch (mode)
    {
    case AcquisitionMode::kSerial: return "Serial (OpenBCI)";
    case AcquisitionMode::kSynthetic: return "Synthetic";
    case AcquisitionMode::kLsl: return "LSL Inlet";
    case AcquisitionMode::kCsvReplay: return "CSV Replay";
    case AcquisitionMode::kBrainFlow: return "BrainFlow";
    }
    return "Unknown";
}

} // namespace lpl::bci

#endif // LPL_BCI_CORE_TYPES_INL
