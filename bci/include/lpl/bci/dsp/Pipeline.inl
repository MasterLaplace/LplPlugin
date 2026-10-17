#ifndef LPL_BCI_DSP_PIPELINE_INL
#define LPL_BCI_DSP_PIPELINE_INL

namespace lpl::bci::dsp {

template <typename T, typename... Args>
requires std::derived_from<T, IStage>
inline PipelineBuilder &PipelineBuilder::add(Args &&...args)
{
    _stages.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    return *this;
}

} // namespace lpl::bci::dsp

#endif // LPL_BCI_DSP_PIPELINE_INL
