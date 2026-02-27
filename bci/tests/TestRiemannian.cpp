/**
 * @file TestRiemannian.cpp
 * @brief Unit tests for bci::math Riemannian geometry operations.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lpl/bci/math/Riemannian.hpp"

namespace lpl::bci {

using namespace bci::math;
using Catch::Matchers::WithinAbs;

TEST_CASE("matrixSqrt of identity is identity", "[math][riemannian]")
{
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(3, 3);
    auto result = matrixSqrt(I);

    REQUIRE(result.has_value());
    REQUIRE(result->isApprox(I, 1e-5f));
}

TEST_CASE("matrixSqrt of diagonal matrix", "[math][riemannian]")
{
    Eigen::MatrixXf diag = Eigen::MatrixXf::Zero(2, 2);
    diag(0, 0) = 4.0f;
    diag(1, 1) = 9.0f;

    auto result = matrixSqrt(diag);
    REQUIRE(result.has_value());

    REQUIRE_THAT(static_cast<double>((*result)(0, 0)), WithinAbs(2.0, 1e-4));
    REQUIRE_THAT(static_cast<double>((*result)(1, 1)), WithinAbs(3.0, 1e-4));
}

TEST_CASE("matrixSqrtInv is inverse of matrixSqrt", "[math][riemannian]")
{
    Eigen::MatrixXf spd(2, 2);
    spd << 2, 1,
           1, 3;

    auto sq = matrixSqrt(spd);
    auto sqInv = matrixSqrtInv(spd);

    REQUIRE(sq.has_value());
    REQUIRE(sqInv.has_value());

    Eigen::MatrixXf product = *sq * *sqInv;
    REQUIRE(product.isApprox(Eigen::MatrixXf::Identity(2, 2), 1e-4f));
}

TEST_CASE("matrixLog of identity is zero", "[math][riemannian]")
{
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(3, 3);
    auto result = matrixLog(I);

    REQUIRE(result.has_value());
    REQUIRE(result->isApprox(Eigen::MatrixXf::Zero(3, 3), 1e-5f));
}

TEST_CASE("riemannianDistance of matrix with itself is zero", "[math][riemannian]")
{
    Eigen::MatrixXf spd(2, 2);
    spd << 2, 0.5f,
           0.5f, 3;

    auto dist = riemannianDistance(spd, spd);
    REQUIRE(dist.has_value());
    REQUIRE_THAT(static_cast<double>(*dist), WithinAbs(0.0, 1e-4));
}

TEST_CASE("riemannianDistance is symmetric", "[math][riemannian]")
{
    Eigen::MatrixXf a(2, 2);
    a << 2, 0.3f,
         0.3f, 2;

    Eigen::MatrixXf b(2, 2);
    b << 3, 0.5f,
         0.5f, 4;

    auto dAB = riemannianDistance(a, b);
    auto dBA = riemannianDistance(b, a);

    REQUIRE(dAB.has_value());
    REQUIRE(dBA.has_value());
    REQUIRE_THAT(static_cast<double>(*dAB), WithinAbs(static_cast<double>(*dBA), 1e-3));
}

TEST_CASE("mahalanobisDistance zero for matching sample and mean", "[math][riemannian]")
{
    Eigen::VectorXf v(3);
    v << 1, 2, 3;

    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(3, 3);

    auto dist = mahalanobisDistance(v, v, cov);
    REQUIRE(dist.has_value());
    REQUIRE_THAT(static_cast<double>(*dist), WithinAbs(0.0, 1e-5));
}

TEST_CASE("frechetMean of identical matrices returns that matrix", "[math][riemannian]")
{
    Eigen::MatrixXf spd(2, 2);
    spd << 3, 1,
           1, 3;

    std::vector<Eigen::MatrixXf> matrices = {spd, spd, spd};
    auto mean = frechetMean(matrices);

    REQUIRE(mean.has_value());
    REQUIRE(mean->isApprox(spd, 1e-3f));
}

TEST_CASE("Riemannian functions reject non-SPD matrices", "[math][riemannian]")
{
    Eigen::MatrixXf singular = Eigen::MatrixXf::Zero(2, 2);

    REQUIRE_FALSE(matrixSqrt(singular).has_value());
    REQUIRE_FALSE(matrixSqrtInv(singular).has_value());
    REQUIRE_FALSE(matrixLog(singular).has_value());
}

TEST_CASE("Riemannian functions reject non-square matrices", "[math][riemannian]")
{
    Eigen::MatrixXf rect(2, 3);
    rect.setOnes();

    REQUIRE_FALSE(matrixSqrt(rect).has_value());
}

} // namespace lpl::bci
