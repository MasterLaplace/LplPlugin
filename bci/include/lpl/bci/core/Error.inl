#ifndef LPL_BCI_CORE_ERROR_INL
#define LPL_BCI_CORE_ERROR_INL

namespace lpl::bci {

constexpr std::string_view errorCodeName(ErrorCode code) noexcept
{
    switch (code)
    {
    case ErrorCode::kSerialPortNotFound: return "SerialPortNotFound";
    case ErrorCode::kSerialPortConfigFailed: return "SerialPortConfigFailed";
    case ErrorCode::kSerialReadFailed: return "SerialReadFailed";
    case ErrorCode::kSerialWriteFailed: return "SerialWriteFailed";
    case ErrorCode::kInvalidPacket: return "InvalidPacket";
    case ErrorCode::kLslStreamNotFound: return "LslStreamNotFound";
    case ErrorCode::kLslConnectionFailed: return "LslConnectionFailed";
    case ErrorCode::kBrainFlowInitFailed: return "BrainFlowInitFailed";
    case ErrorCode::kBrainFlowStreamFailed: return "BrainFlowStreamFailed";
    case ErrorCode::kFileNotFound: return "FileNotFound";
    case ErrorCode::kFileParseError: return "FileParseError";
    case ErrorCode::kCalibrationIncomplete: return "CalibrationIncomplete";
    case ErrorCode::kCalibrationInsufficientSamples: return "CalibrationInsufficientSamples";
    case ErrorCode::kFftSizeMismatch: return "FftSizeMismatch";
    case ErrorCode::kChannelCountMismatch: return "ChannelCountMismatch";
    case ErrorCode::kEmptyInput: return "EmptyInput";
    case ErrorCode::kInvalidArgument: return "InvalidArgument";
    case ErrorCode::kInvalidState: return "InvalidState";
    case ErrorCode::kPipelineEmpty: return "PipelineEmpty";
    case ErrorCode::kMatrixNotSpd: return "MatrixNotSpd";
    case ErrorCode::kSingularMatrix: return "SingularMatrix";
    case ErrorCode::kMathError: return "MathError";
    case ErrorCode::kNotInitialized: return "NotInitialized";
    case ErrorCode::kAlreadyRunning: return "AlreadyRunning";
    case ErrorCode::kUnknown: return "Unknown";
    }
    return "Unknown";
}

} // namespace lpl::bci

#endif // LPL_BCI_CORE_ERROR_INL
