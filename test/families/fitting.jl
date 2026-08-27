# Family-regression layer: the fitting and StatsBase contracts live under
# `contracts/` and `paths/`; only optimizer recovery, boundary starts, and an
# unavailable-metadata error regression remain here.

@testset "family fitting parameter-recovery regressions" begin
    # CopulaModel/StatsBase behavior and the covariance mechanism moved to the
    # new contracts. Retain only family-specific optimizer recovery assertions.
    rng = StableRNG(2025)
    reps = (
        (GaussianCopula, 2, :mle),
        (GaussianCopula, 3, :mle),
        (GumbelCopula, 2, :itau),
        (FrankCopula, 2, :mle),
        (JoeCopula, 2, :itau),
        (BB6Copula, 2, :mle),
        (BB7Copula, 2, :mle),
        (GalambosCopula, 2, :mle),
        (HuslerReissCopula, 2, :mle),
    )

    for (CT, d, method) in reps
        C0 = Copulas._example(CT, d)
        truth = Copulas._flatten_params(params(C0))[2]
        U = rand(rng, C0, 250)
        M = fit(CopulaModel, CT, U; method, vcov=false,
                derived_measures=false)
        estimate = StatsBase.coef(M)
        if CT <: BB6Copula
            @test prod(estimate) ≈ prod(truth) rtol=0.2
            @test M.ll >= loglikelihood(C0, U) - 1e-6
        else
            @test estimate ≈ truth atol=0.5
        end
    end
end

@testset "model metadata error regression" begin
    M = CopulaModel(IndependentCopula{2}(), 10, 0.0, :dummy)
    @test_throws ArgumentError StatsBase.residuals(M)
end

@testset "Extreme-value MLE accepts boundary starts" begin
    U = [0.10 0.25 0.40 0.55 0.70 0.85;
         0.15 0.20 0.45 0.60 0.75 0.90]

    for CT in (CuadrasAugeCopula, LogCopula)
        fitted = fit(CT, U, :mle; start=1.0)
        @test fitted isa Copulas.Copula
        @test all(isfinite, Distributions.params(fitted))
    end
end
