/**
 * @file Concepts.hpp
 * @brief C++20 concepts constraining generic template parameters in the BCI pipeline.
 * @author MasterLaplace
 *
 * These concepts are used to enforce structural contracts at compile time,
 * providing clear error messages when a type does not satisfy the required
 * interface. They complement (but do not replace) the runtime-polymorphic
 * interfaces ISource and IStage.
 */

#pragma once

#include "Error.hpp"
#include "Types.hpp"

#include <concepts>
#include <span>
#include <string_view>

namespace bci {

/**
 * @brief A type that can acquire raw EEG samples.
 *
 * Satisfied by any class exposing start(), read(), stop(), and info()
 * with the expected signatures. Useful for constraining template
 * parameters that accept source-like objects without requiring
 * inheritance from ISource.
 */
template <typename T>
concept SourceLike = requires(T source, std::span<Sample> buffer) {
    { source.start() } -> std::same_as<ExpectedVoid>;
    { source.read(buffer) } -> std::same_as<Expected<std::size_t>>;
    { source.stop() } noexcept;
    { source.info() } noexcept -> std::same_as<SourceInfo>;
};

/**
 * @brief A type that transforms a SignalBlock through a processing stage.
 *
 * Satisfied by any class exposing process() and name() with the expected
 * signatures. Used to constrain Pipeline builder template parameters.
 */
template <typename T>
concept DspStageLike = requires(T stage, const SignalBlock &block) {
    { stage.process(block) } -> std::same_as<Expected<SignalBlock>>;
    { stage.name() } noexcept -> std::convertible_to<std::string_view>;
};

} // namespace bci
