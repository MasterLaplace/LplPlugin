/**
 * @file Covariance.cpp
 * @brief Implementation of batch and online covariance estimators.
 */

#include "lpl/bci/math/Covariance.hpp"

namespace lpl::bci::math {

Eigen::MatrixXf batchCovariance(const Eigen::MatrixXf& data) noexcept
{
    const auto rows = data.rows();
    const auto cols = data.cols();

    if (rows < 2)
        return Eigen::MatrixXf::Zero(cols, cols);

    const Eigen::VectorXf mean = data.colwise().mean();
    const Eigen::MatrixXf centered = data.rowwise() - mean.transpose();
    return (centered.transpose() * centered) / static_cast<float>(rows - 1);
}

Eigen::MatrixXf regularizeCovariance(
    const Eigen::MatrixXf& cov,
    float alpha) noexcept
{
    const auto p = cov.rows();

    if (p == 0)
        return cov;

    const float traceOverP = cov.trace() / static_cast<float>(p);
    return (1.0f - alpha) * cov + alpha * traceOverP * Eigen::MatrixXf::Identity(p, p);
}

WelfordCovariance::WelfordCovariance(std::size_t channelCount)
    : _channelCount(channelCount)
    , _mean(Eigen::VectorXf::Zero(static_cast<Eigen::Index>(channelCount)))
    , _coProduct(Eigen::MatrixXf::Zero(
          static_cast<Eigen::Index>(channelCount),
          static_cast<Eigen::Index>(channelCount)))
{
}

void WelfordCovariance::update(std::span<const float> sample) noexcept
{
    if (sample.size() < _channelCount)
        return;

    const auto p = static_cast<Eigen::Index>(_channelCount);
    const Eigen::Map<const Eigen::VectorXf> x(sample.data(), p);

    ++_n;
    const Eigen::VectorXf delta = x - _mean;
    _mean += delta / static_cast<float>(_n);
    const Eigen::VectorXf delta2 = x - _mean;
    _coProduct.noalias() += delta * delta2.transpose();
}

Eigen::MatrixXf WelfordCovariance::covariance() const noexcept
{
    if (_n < 2)
        return Eigen::MatrixXf::Zero(
            static_cast<Eigen::Index>(_channelCount),
            static_cast<Eigen::Index>(_channelCount));

    return _coProduct / static_cast<float>(_n - 1);
}

const Eigen::VectorXf& WelfordCovariance::mean() const noexcept
{
    return _mean;
}

std::size_t WelfordCovariance::count() const noexcept
{
    return _n;
}

void WelfordCovariance::reset() noexcept
{
    _n = 0;
    _mean.setZero();
    _coProduct.setZero();
}

} // namespace lpl::bci::math
