/**
 * @file TestCovariance.cpp
 * @brief Unit tests for bci::math covariance estimators.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lpl/bci/math/Covariance.hpp"

using namespace bci::math;
using Catch::Matchers::WithinAbs;

TEST_CASE("batchCovariance produces correct result", "[math][covariance]")
{
    SECTION("identity-like data")
    {
        Eigen::MatrixXf data(4, 2);
        data << 1, 2,
                3, 4,
                5, 6,
                7, 8;

        auto cov = batchCovariance(data);

        REQUIRE(cov.rows() == 2);
        REQUIRE(cov.cols() == 2);
        REQUIRE_THAT(cov(0, 0), WithinAbs(cov(1, 1), 1e-5f));
        REQUIRE_THAT(cov(0, 1), WithinAbs(cov(1, 0), 1e-5f));
    }

    SECTION("single sample returns zero matrix")
    {
        Eigen::MatrixXf data(1, 3);
        data << 1, 2, 3;

        auto cov = batchCovariance(data);
        REQUIRE(cov.isZero());
    }
}

TEST_CASE("regularizeCovariance shrinks towards identity", "[math][covariance]")
{
    Eigen::MatrixXf cov(2, 2);
    cov << 4, 2,
           2, 4;

    auto reg = regularizeCovariance(cov, 1.0f);
    const float expectedDiag = cov.trace() / 2.0f;

    REQUIRE_THAT(reg(0, 0), WithinAbs(expectedDiag, 1e-5f));
    REQUIRE_THAT(reg(0, 1), WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("WelfordCovariance converges to batch result", "[math][covariance]")
{
    Eigen::MatrixXf data(100, 3);
    for (int i = 0; i < 100; ++i) {
        data(i, 0) = static_cast<float>(i);
        data(i, 1) = static_cast<float>(i * 2);
        data(i, 2) = static_cast<float>(i * 3 + 1);
    }

    auto batchCov = batchCovariance(data);

    WelfordCovariance welford(3);
    for (int i = 0; i < 100; ++i) {
        std::array<float, 3> sample = {data(i, 0), data(i, 1), data(i, 2)};
        welford.update(sample);
    }

    REQUIRE(welford.count() == 100);

    auto onlineCov = welford.covariance();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            REQUIRE_THAT(
                static_cast<double>(onlineCov(r, c)),
                WithinAbs(static_cast<double>(batchCov(r, c)), 0.1));
        }
    }
}

TEST_CASE("WelfordCovariance reset clears state", "[math][covariance]")
{
    WelfordCovariance welford(2);
    std::array<float, 2> sample = {1.0f, 2.0f};
    welford.update(sample);
    welford.reset();

    REQUIRE(welford.count() == 0);
    REQUIRE(welford.mean().isZero());
}
