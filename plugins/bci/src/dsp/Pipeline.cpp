/**
 * @file Pipeline.cpp
 * @brief Implementation of the DSP Pipeline and PipelineBuilder.
 * @author MasterLaplace
 */

#include "dsp/Pipeline.hpp"

namespace bci::dsp {

// ─── PipelineBuilder ─────────────────────────────────────────────────────────

Pipeline PipelineBuilder::build()
{
    return Pipeline(std::move(_stages));
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

Pipeline::Pipeline(std::vector<std::unique_ptr<IStage>> stages)
    : _stages(std::move(stages))
{
}

PipelineBuilder Pipeline::builder()
{
    return PipelineBuilder{};
}

Expected<SignalBlock> Pipeline::process(const SignalBlock &input)
{
    if (_stages.empty()) {
        return input;
    }

    Expected<SignalBlock> current = _stages.front()->process(input);

    for (std::size_t i = 1; i < _stages.size(); ++i) {
        if (!current.has_value()) {
            return current;
        }
        current = _stages[i]->process(current.value());
    }

    return current;
}

std::size_t Pipeline::stageCount() const noexcept
{
    return _stages.size();
}

bool Pipeline::empty() const noexcept
{
    return _stages.empty();
}

} // namespace bci::dsp
