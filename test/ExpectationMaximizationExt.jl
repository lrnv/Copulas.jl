using Copulas
using Distributions
using ExpectationMaximization
using Random
using Test

const EM_RNG = MersenneTwister(27)
const EM_N = 120

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
        fitted = fit_mle(
            initial,
            data;
            copula_kwargs=(; vcov=false, derived_measures=false),
        )

        @test fitted isa SklarDist
        @test fitted.m[1] isa MixtureModel
        @test fitted.m[2] isa LogNormal
        @test all(isfinite, logpdf(fitted, data[:, 1:5]))
    end

    @testset "Weighted component fits" begin
        copula = ClaytonCopula(2, 1.4)
        uniforms = rand(EM_RNG, copula, EM_N)
        weights = collect(range(0.2, 1.0; length=EM_N))
        fitted_copula = fit_mle(copula, uniforms, weights)
        @test fitted_copula isa ClaytonCopula

        sklar = SklarDist(copula, (normal_mix, Normal(0.2, 0.5)))
        data = rand(EM_RNG, sklar, EM_N)
        fitted_sklar = fit_mle(sklar, data, weights)
        @test fitted_sklar isa SklarDist
        @test fitted_sklar.m[1] isa MixtureModel
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
        all_mixture_fit = fit_mle(
            all_mixture_margins,
            all_mixture_data;
            copula_kwargs=(; vcov=false, derived_measures=false),
        )
        @test all(margin -> margin isa MixtureModel, all_mixture_fit.m)
    end
end
