/**
 * @file Riemannian.cpp
 * @brief Implementation of Riemannian geometry on the SPD manifold.
 */

#include "lpl/bci/math/Riemannian.hpp"

#include <cmath>
#include <format>

namespace lpl::bci::math {

namespace {

constexpr float kEigenvalueEpsilon = 1e-10f;

Expected<Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf>> decomposeSpd(
    const Eigen::MatrixXf& spd,
    std::source_location loc = std::source_location::current())
{
    if (spd.rows() != spd.cols())
        return std::unexpected(Error{
            ErrorCode::kMathError,
            std::format("expected square matrix, got {}x{}", spd.rows(), spd.cols()),
            loc});

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(spd);

    if (solver.info() != Eigen::Success)
        return std::unexpected(Error{
            ErrorCode::kMathError,
            "eigenvalue decomposition failed",
            loc});

    if (solver.eigenvalues().minCoeff() < kEigenvalueEpsilon)
        return std::unexpected(Error{
            ErrorCode::kMathError,
            std::format("matrix is not SPD (min eigenvalue: {})", solver.eigenvalues().minCoeff()),
            loc});

    return solver;
}

} // namespace

Expected<Eigen::MatrixXf> matrixSqrt(const Eigen::MatrixXf& spd)
{
    auto solver = decomposeSpd(spd);
    if (!solver)
        return std::unexpected(solver.error());

    const Eigen::VectorXf sqrtEv = solver->eigenvalues().array().sqrt().matrix();
    return solver->eigenvectors() * sqrtEv.asDiagonal() * solver->eigenvectors().transpose();
}

Expected<Eigen::MatrixXf> matrixSqrtInv(const Eigen::MatrixXf& spd)
{
    auto solver = decomposeSpd(spd);
    if (!solver)
        return std::unexpected(solver.error());

    const Eigen::VectorXf invSqrtEv = solver->eigenvalues().array().rsqrt().matrix();
    return solver->eigenvectors() * invSqrtEv.asDiagonal() * solver->eigenvectors().transpose();
}

Expected<Eigen::MatrixXf> matrixLog(const Eigen::MatrixXf& spd)
{
    auto solver = decomposeSpd(spd);
    if (!solver)
        return std::unexpected(solver.error());

    const Eigen::VectorXf logEv = solver->eigenvalues().array().log().matrix();
    return solver->eigenvectors() * logEv.asDiagonal() * solver->eigenvectors().transpose();
}

Expected<float> riemannianDistance(
    const Eigen::MatrixXf& a,
    const Eigen::MatrixXf& b)
{
    auto sqrtInvA = matrixSqrtInv(a);
    if (!sqrtInvA)
        return std::unexpected(sqrtInvA.error());

    const Eigen::MatrixXf inner = *sqrtInvA * b * *sqrtInvA;

    auto logInner = matrixLog(inner);
    if (!logInner)
        return std::unexpected(logInner.error());

    return logInner->norm();
}

Expected<float> mahalanobisDistance(
    const Eigen::VectorXf& sample,
    const Eigen::VectorXf& mean,
    const Eigen::MatrixXf& cov)
{
    if (sample.size() != mean.size() || sample.size() != cov.rows())
        return std::unexpected(Error{
            ErrorCode::kMathError,
            std::format(
                "dimension mismatch: sample({}), mean({}), cov({}x{})",
                sample.size(), mean.size(), cov.rows(), cov.cols())});

    const Eigen::VectorXf diff = sample - mean;

    Eigen::LLT<Eigen::MatrixXf> llt(cov);
    if (llt.info() != Eigen::Success)
        return std::unexpected(Error{
            ErrorCode::kMathError,
            "covariance matrix is not positive-definite (Cholesky failed)"});

    const Eigen::VectorXf solved = llt.solve(diff);
    return std::sqrt(diff.dot(solved));
}

Expected<Eigen::MatrixXf> frechetMean(
    std::span<const Eigen::MatrixXf> matrices,
    std::size_t maxIter,
    float tolerance)
{
    if (matrices.empty())
        return std::unexpected(Error{
            ErrorCode::kMathError,
            "cannot compute Fréchet mean of empty set"});

    const auto p = matrices[0].rows();
    const auto n = static_cast<float>(matrices.size());

    Eigen::MatrixXf mean = matrices[0];

    for (std::size_t iter = 0; iter < maxIter; ++iter) {
        auto sqrtMean = matrixSqrt(mean);
        if (!sqrtMean)
            return std::unexpected(sqrtMean.error());

        auto sqrtInvMean = matrixSqrtInv(mean);
        if (!sqrtInvMean)
            return std::unexpected(sqrtInvMean.error());

        Eigen::MatrixXf tangentSum = Eigen::MatrixXf::Zero(p, p);

        for (const auto& ci : matrices) {
            const Eigen::MatrixXf projected = *sqrtInvMean * ci * *sqrtInvMean;

            auto logProjected = matrixLog(projected);
            if (!logProjected)
                return std::unexpected(logProjected.error());

            tangentSum += *logProjected;
        }

        tangentSum /= n;

        auto solver = decomposeSpd(Eigen::MatrixXf::Identity(p, p) + tangentSum);
        const Eigen::MatrixXf expTangent = [&]() -> Eigen::MatrixXf {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> ev(tangentSum);
            const Eigen::VectorXf expEv = ev.eigenvalues().array().exp().matrix();
            return ev.eigenvectors() * expEv.asDiagonal() * ev.eigenvectors().transpose();
        }();

        const Eigen::MatrixXf newMean = *sqrtMean * expTangent * *sqrtMean;

        if ((newMean - mean).norm() < tolerance) {
            mean = newMean;
            break;
        }

        mean = newMean;
    }

    return mean;
}

} // namespace lpl::bci::math
