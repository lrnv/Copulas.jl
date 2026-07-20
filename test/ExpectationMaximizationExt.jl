using Copulas
using Distributions
using ExpectationMaximization
using Random
using Test

const EM_RNG = MersenneTwister(27)
const EM_N = 80

function em_weighted_loglikelihood(distribution, data, weights)
    return sum(eachindex(weights)) do j
        weights[j] * logpdf(distribution, view(data, :, j))
    end
end

@testset "ExpectationMaximization extension" begin
    normal_mix = MixtureModel(
        [Normal(-1.0, 0.6), Normal(1.2, 0.8)],
        [0.35, 0.65],
    )

    @testset "SklarDist with mixture margins" begin
        initial = SklarDist(
            ClaytonCopula(2, 1.4),
            (normal_mix, LogNormal(0.2, 0.5)),
        )
        data = rand(EM_RNG, initial, EM_N)
        initial_loglikelihood = em_weighted_loglikelihood(
            initial,
            data,
            ones(EM_N),
        )
        fitted = fit_mle(initial, data)

        @test fitted isa SklarDist
        @test fitted.m[1] isa MixtureModel
        @test fitted.m[2] isa LogNormal
        @test all(isfinite, logpdf(fitted, data[:, 1:5]))
        @test em_weighted_loglikelihood(fitted, data, ones(EM_N)) >=
              initial_loglikelihood - 1e-8
    end

    @testset "Weighted component fits" begin
        copula = ClaytonCopula(2, 0.4)
        uniforms = rand(EM_RNG, ClaytonCopula(2, 2.0), EM_N)
        weights = collect(range(0.2, 1.0; length=EM_N))
        fitted_copula = fit_mle(copula, uniforms, weights)
        @test fitted_copula isa ClaytonCopula
        @test isfinite(Distributions.params(fitted_copula).θ)
        @test em_weighted_loglikelihood(fitted_copula, uniforms, weights) >
              em_weighted_loglikelihood(copula, uniforms, weights) + 1e-6

        unweighted_copula = fit_mle(copula, uniforms)
        equally_weighted_copula = fit_mle(copula, uniforms, ones(EM_N))
        @test Distributions.params(equally_weighted_copula).θ ≈
              Distributions.params(unweighted_copula).θ atol=0.15 rtol=0.15

        low_dependence = rand(EM_RNG, ClaytonCopula(2, 0.2), EM_N ÷ 2)
        high_dependence = rand(EM_RNG, ClaytonCopula(2, 4.0), EM_N ÷ 2)
        contrasted_data = hcat(low_dependence, high_dependence)
        low_weights = vcat(ones(EM_N ÷ 2), zeros(EM_N ÷ 2))
        high_weights = vcat(zeros(EM_N ÷ 2), ones(EM_N ÷ 2))
        low_fit = fit_mle(ClaytonCopula(2, 1.0), contrasted_data, low_weights)
        high_fit = fit_mle(ClaytonCopula(2, 1.0), contrasted_data, high_weights)
        @test Distributions.params(low_fit).θ < Distributions.params(high_fit).θ

        true_sklar = SklarDist(
            ClaytonCopula(2, 1.8),
            (Normal(1.0, 0.7), LogNormal(0.3, 0.4)),
        )
        sklar = SklarDist(
            copula,
            (Normal(-0.5, 1.4), LogNormal(-0.2, 0.9)),
        )
        data = rand(EM_RNG, true_sklar, EM_N)
        initial_loglikelihood = em_weighted_loglikelihood(sklar, data, weights)
        fitted_sklar = fit_mle(sklar, data, weights)
        @test fitted_sklar isa SklarDist
        @test all(isfinite, logpdf(fitted_sklar, data[:, 1:5]))
        @test em_weighted_loglikelihood(fitted_sklar, data, weights) >
              initial_loglikelihood + 1e-6
    end

    @testset "Input validation and numerical boundaries" begin
        copula = ClaytonCopula(2, 1.0)
        uniforms = rand(EM_RNG, copula, 12)
        weights = ones(12)

        @test_throws DimensionMismatch fit_mle(copula, uniforms[1:1, :], weights)
        @test_throws DimensionMismatch fit_mle(copula, uniforms, weights[1:end-1])
        @test_throws ArgumentError fit_mle(copula, uniforms, fill(0.0, 12))
        @test_throws ArgumentError fit_mle(copula, uniforms, vcat(-1.0, ones(11)))
        @test_throws ArgumentError fit_mle(copula, uniforms, vcat(Inf, ones(11)))
        @test_throws ArgumentError fit_mle(copula, uniforms, weights; method=:itau)
        @test_throws ArgumentError fit_mle(copula, uniforms, weights; iterations=2)
        @test fit_mle(IndependentCopula(2), uniforms, weights) isa IndependentCopula

        sklar = SklarDist(
            ClaytonCopula(2, 0.8),
            (Normal(0.0, 10.0), Normal(0.0, 10.0)),
        )
        integer_extremes = [
            -1000 -20 -1 0 1 20 1000
            1000 20 1 0 -1 -20 -1000
        ]
        integer_fit = fit_mle(sklar, integer_extremes)
        @test integer_fit isa SklarDist
        @test isfinite(em_weighted_loglikelihood(
            integer_fit,
            integer_extremes[:, 2:end-1],
            ones(5),
        ))

        @test_throws DimensionMismatch fit_mle(sklar, integer_extremes[1:1, :])
        @test_throws ArgumentError fit_mle(sklar, Matrix{Float64}(undef, 2, 0))
        nonfinite_data = Float64.(integer_extremes)
        nonfinite_data[1, 1] = NaN
        @test_throws ArgumentError fit_mle(sklar, nonfinite_data)

        discrete_sklar = SklarDist(
            IndependentCopula(2),
            (Poisson(2.0), Normal()),
        )
        @test_throws ArgumentError fit_mle(discrete_sklar, Float64.(integer_extremes))

        unsupported_sklar = SklarDist(
            IndependentCopula(2),
            (TriangularDist(-2.0, 2.0, 0.0), Normal()),
        )
        @test_throws ArgumentError fit_mle(
            unsupported_sklar,
            Float64.(integer_extremes),
        )

        zero_probability_margin = MixtureModel(
            [Normal(-1.0, 1.0), Normal(1.0, 1.0)],
            [1.0, 0.0],
        )
        zero_probability_sklar = SklarDist(
            IndependentCopula(2),
            (zero_probability_margin, Normal()),
        )
        @test_throws ArgumentError fit_mle(
            zero_probability_sklar,
            Float64.(integer_extremes),
        )
    end

    @testset "Mixtures of copulas" begin
        initial = MixtureModel(
            [ClaytonCopula(2, 0.6), ClaytonCopula(2, 2.5)],
            [0.5, 0.5],
        )
        data = rand(EM_RNG, initial, EM_N)
        fitted = fit_mle(initial, data; maxiter=3)

        @test fitted isa MixtureModel
        @test all(component -> component isa ClaytonCopula, components(fitted))
        @test loglikelihood(fitted, data) >= loglikelihood(initial, data) - 1e-8

        heterogeneous = MixtureModel(
            Copulas.Copula[
                ClaytonCopula(2, 1.0),
                GaussianCopula(2, -0.35),
            ],
            [0.5, 0.5],
        )
        heterogeneous_data = rand(EM_RNG, heterogeneous, EM_N)
        heterogeneous_fit = fit_mle(heterogeneous, heterogeneous_data; maxiter=1)
        @test heterogeneous_fit isa MixtureModel
        @test components(heterogeneous_fit)[1] isa ClaytonCopula
        @test components(heterogeneous_fit)[2] isa GaussianCopula
    end

    @testset "Mixtures of SklarDist" begin
        component1 = SklarDist(
            ClaytonCopula(2, 0.7),
            (Normal(-1.0, 0.8), Normal(0.0, 0.4)),
        )
        component2 = SklarDist(
            ClaytonCopula(2, 2.2),
            (Normal(1.2, 0.6), Normal(0.8, 0.3)),
        )
        initial = MixtureModel([component1, component2], [0.45, 0.55])
        data = rand(EM_RNG, initial, EM_N)
        fitted = fit_mle(initial, data; maxiter=3)

        @test fitted isa MixtureModel
        @test all(component -> component isa SklarDist, components(fitted))
        @test loglikelihood(fitted, data) >= loglikelihood(initial, data) - 1e-8
    end

    @testset "Nested mixture margins" begin
        component1 = SklarDist(
            ClaytonCopula(2, 0.8),
            (normal_mix, Normal(-0.5, 0.8)),
        )
        component2 = SklarDist(
            ClaytonCopula(2, 2.0),
            (normal_mix, Normal(1.0, 0.6)),
        )
        initial = MixtureModel([component1, component2], [0.5, 0.5])
        data = rand(EM_RNG, initial, EM_N)
        fitted = fit_mle(initial, data; maxiter=2)

        @test fitted isa MixtureModel
        @test all(component -> component.m[1] isa MixtureModel, components(fitted))

        all_mixture_margins = SklarDist(
            GaussianCopula(2, 0.45),
            (
                normal_mix,
                MixtureModel(
                    [Gamma(2.0, 0.7), Gamma(6.0, 0.35)],
                    [0.55, 0.45],
                ),
            ),
        )
        all_mixture_data = rand(EM_RNG, all_mixture_margins, EM_N)
        initial_loglikelihood = em_weighted_loglikelihood(
            all_mixture_margins,
            all_mixture_data,
            ones(EM_N),
        )
        all_mixture_fit = fit_mle(all_mixture_margins, all_mixture_data)
        @test all(margin -> margin isa MixtureModel, all_mixture_fit.m)
        @test all(isfinite, logpdf(all_mixture_fit, all_mixture_data[:, 1:5]))
        @test em_weighted_loglikelihood(
            all_mixture_fit,
            all_mixture_data,
            ones(EM_N),
        ) >= initial_loglikelihood - 1e-8
    end
end
