#ifndef LPL_BCI_SOURCEBCIDRIVER_INL
#define LPL_BCI_SOURCEBCIDRIVER_INL

namespace lpl::bci {

inline SourceBciDriver::~SourceBciDriver()
{
    if (_status == DriverStatus::Streaming)
        stopStream();
    if (_status == DriverStatus::Connected)
        disconnect();
}

inline core::Expected<void> SourceBciDriver::connect()
{
    _status = DriverStatus::Connected;
    return {};
}

inline core::Expected<void> SourceBciDriver::startStream()
{
    auto res = _source->start();
    if (!res)
        return core::makeError(core::ErrorCode::kDeviceOpenFailed, "BCI source start failed");
    _status = DriverStatus::Streaming;
    return {};
}

inline core::Expected<RawSample> SourceBciDriver::poll()
{
    Sample buf{};
    std::span<Sample> span{&buf, 1};
    auto readResult = _source->read(span);
    if (!readResult.has_value() || readResult.value() == 0)
        return core::makeError(core::ErrorCode::kInvalidState, "No BCI data available");

    RawSample raw{};
    raw.channelCount = static_cast<core::u16>(buf.channelCount());
    for (core::u16 ch = 0; ch < raw.channelCount && ch < RawSample::kMaxChannels; ++ch)
        raw.channels[ch] = buf.channels[ch];
    raw.timestampUs = static_cast<core::u64>(buf.timestamp * 1e6);
    raw.sequence = _seq++;
    return raw;
}

inline void SourceBciDriver::stopStream()
{
    _source->stop();
    _status = DriverStatus::Connected;
}

} // namespace lpl::bci

#endif // LPL_BCI_SOURCEBCIDRIVER_INL
