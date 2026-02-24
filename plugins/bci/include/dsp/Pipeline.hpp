/**
 * @file Pipeline.hpp
 * @brief Fluent Builder for composing sequential DSP processing stages.
 * @author MasterLaplace
 *
 * The Pipeline chains IStage instances sequentially: each stage receives
 * the output of the previous one. Processing short-circuits on the first
 * error, propagating the std::unexpected through the chain.
 *
 * @code
 *   auto pipeline = Pipeline::builder()
 *       .add<HannWindow>()
 *       .add<FftProcessor>(256)
 *       .add<BandExtractor>(bands, 250.0f, 256)
 *       .build();
 *
 *   auto result = pipeline.process(block);
 * @endcode
 *
 * @see IStage
 */

#pragma once

#include "IStage.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace bci::dsp {

class Pipeline;

/**
 * @brief Fluent builder that accumulates IStage instances for a Pipeline.
 *
 * Supports in-place construction of any IStage-derived type via add<T>(args...).
 */
class PipelineBuilder {
public:
    /**
     * @brief Constructs and appends a stage of type T.
     *
     * @tparam T    A concrete class derived from IStage
     * @tparam Args Constructor argument types for T
     * @param args  Arguments forwarded to T's constructor
     * @return Reference to this builder (for chaining)
     */
    template <typename T, typename... Args>
        requires std::derived_from<T, IStage>
    PipelineBuilder &add(Args &&...args)
    {
        _stages.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return *this;
    }

    /**
     * @brief Finalizes construction and returns the Pipeline.
     */
    [[nodiscard]] Pipeline build();

private:
    std::vector<std::unique_ptr<IStage>> _stages;
};

/**
 * @brief An ordered chain of DSP processing stages.
 *
 * Immutable after construction. Thread-safe for concurrent process() calls
 * only if each IStage implementation is itself thread-safe (which is the
 * case for the built-in stages).
 */
class Pipeline {
public:
    /**
     * @brief Returns a PipelineBuilder for fluent stage composition.
     */
    [[nodiscard]] static PipelineBuilder builder();

    /**
     * @brief Runs the input block through every stage sequentially.
     *
     * @param input The initial signal block
     * @return The final transformed SignalBlock, or the first Error encountered
     */
    [[nodiscard]] Expected<SignalBlock> process(const SignalBlock &input);

    /**
     * @brief Returns the number of stages in this pipeline.
     */
    [[nodiscard]] std::size_t stageCount() const noexcept;

    /**
     * @brief Returns true if the pipeline contains no stages.
     */
    [[nodiscard]] bool empty() const noexcept;

private:
    friend class PipelineBuilder;

    explicit Pipeline(std::vector<std::unique_ptr<IStage>> stages);

    std::vector<std::unique_ptr<IStage>> _stages;
};

} // namespace bci::dsp
