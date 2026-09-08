# Correctness obligation: independent numerical regressions for internal
# primitives shared by several public families.

struct _QuantileStepFixture <: Distributions.DiscreteUnivariateDistribution end
Base.minimum(::_QuantileStepFixture) = 0.0
Base.maximum(::_QuantileStepFixture) = 2.0
function Distributions.cdf(::_QuantileStepFixture, x::Real)
    x < 0 && return 0.0
    x < 1 && return 0.25
    x < 2 && return 0.75
    return 1.0
end

struct _IncompleteCDFFixture <: Distributions.ContinuousUnivariateDistribution end
Base.minimum(::_IncompleteCDFFixture) = 0.0
Base.maximum(::_IncompleteCDFFixture) = 1.0
Distributions.cdf(::_IncompleteCDFFixture, x::Real) = x <= 0 ? 0.0 : min(0.5, x)

struct _NaNCDFFixture <: Distributions.ContinuousUnivariateDistribution end
Base.minimum(::_NaNCDFFixture) = 0.0
Base.maximum(::_NaNCDFFixture) = 1.0
Distributions.cdf(::_NaNCDFFixture, x::Real) = NaN

struct _LogCDFQuantileFixture <: Copulas.Distortion end
function Distributions.logcdf(::_LogCDFQuantileFixture, x::Real)
    x <= 0 && return -Inf
    x >= 1 && return 0.0
    return 2log(x)
end

@testset "support-aware generalized quantiles" begin
    step = _QuantileStepFixture()
    @test Copulas._quantile_from_cdf(step, 0.0) == 0.0
    @test Copulas._quantile_from_cdf(step, 0.25) == 0.0
    @test Copulas._quantile_from_cdf(step, 0.26) == 1.0
    @test Copulas._quantile_from_cdf(step, 0.75) == 1.0
    @test Copulas._quantile_from_cdf(step, 0.76) == 2.0
    @test Copulas._quantile_from_cdf(step, 1.0) == 2.0

    for D in (Exponential(2.0), Normal(1.0, 2.0))
        for p in (0.01, 0.4, 0.99)
            @test Copulas._quantile_from_cdf(D, p) ≈ quantile(D, p) rtol=2e-14
        end
    end
    @test Copulas._quantile_from_cdf(
        Copulas.LogCDFQuantile(), Normal(), 1e-12,
    ) ≈ quantile(Normal(), 1e-12) rtol=2e-12
    @test quantile(_LogCDFQuantileFixture(), 0.36) ≈ 0.6

    @test Copulas._quantile_from_cdf(Uniform(), Float32(0.3)) isa Float32
    @test_throws ArgumentError Copulas._quantile_from_cdf(Uniform(), -0.1)
    @test_throws ArgumentError Copulas._quantile_from_cdf(Uniform(), 1.1)
    @test_throws ArgumentError Copulas._quantile_from_cdf(_IncompleteCDFFixture(), 0.8)
    @test_throws ArgumentError Copulas._quantile_from_cdf(_NaNCDFFixture(), 0.5)

    @test Copulas.quantile_strategy(Copulas.NoDistortion) isa Copulas.LogCDFQuantile
    @test Copulas.quantile_strategy(Uniform) isa Copulas.CDFQuantile
end

@testset "stable factorial recurrences" begin
    @test Copulas._falling_factorial(19.0, 2) == 342.0
    @test Copulas._falling_factorial(3.5, 2) == 8.75
    @test Copulas._mul_factorial(1.0, 22) ≈ gamma(23)
    @test Copulas._div_factorial(1.0, 22) ≈ inv(gamma(23))
    @test Copulas._rising_factorial(0.5, 9) ≈ gamma(9.5) / gamma(0.5)

    G = Copulas.ClaytonGenerator(1.0)
    generic_derivative = invoke(
        Copulas.ϕ⁽ᵏ⁾,
        Tuple{Copulas.Generator, Int, Any},
        G,
        22,
        1.0,
    )
    @test generic_derivative ≈ Copulas.ϕ⁽ᵏ⁾(G, 22, 1.0)

    radial = Copulas.𝒲₋₁(G, 22)
    @test 0 <= cdf(radial, 1.0) <= 1

    clayton_radial = Copulas.ClaytonWilliamsonDistribution(-0.001, 25)
    @test cdf(clayton_radial, 0.0) == 0
    @test 0 <= cdf(clayton_radial, 500.0) <= 1
    @test isfinite(logpdf(clayton_radial, 500.0))

    @test isfinite(Copulas.γ(rand(rng, 25, 10)))
end
