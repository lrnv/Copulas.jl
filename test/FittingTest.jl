

@testitem "Fitting smoke test" tags=[:Fitting] begin
    using Test
    using Random
    using Distributions
    using Copulas
    using StableRNGs
    rng = StableRNG(123)

    using Copulas: ClaytonGenerator, WilliamsonGenerator, GumbelGenerator, GalambosTail, MixedTail, ExtremeValueCopula, FrailtyGenerator # to avoid typing "Copulas." in front. 

    # Structured manifest of test cases
    # Each entry: (Type, dims::String)
    # - dims: string of digits among "2","3","4"; remove a digit to skip that dimension
    cases = [
        # No parameters
        (IndependentCopula,                              "234"),
        (MCopula,                                        "234"),
        (WCopula,                                        "2"),

        # Empirical/misc (all d)
        (BernsteinCopula,                                "234"),
        (BetaCopula,                                     "234"),
        (CheckerboardCopula,                             "234"),
        (EmpiricalCopula,                                "234"),
        (ArchimedeanCopula,                              "234"),
        (ExtremeValueCopula,                              "2"),


        # # Elliptical (all d)
        (GaussianCopula,                                "234"),
        # (TCopula,                                       "234"), # takes a loooooot of time.

        # Archimedean families wiht one parameters
        (AMHCopula,                                     "234"),
        (ClaytonCopula,                                 "234"),
        (FrankCopula,                                   "234"),
        (GumbelBarnettCopula,                           "234"),
        (GumbelCopula,                                  "234"),
        (InvGaussianCopula,                             "234"),
        (JoeCopula,                                     "234"),

        # Archimedeans families with two parameters.
        (BB1Copula,                                     "234"),
        (BB3Copula,                                     "234"),
        (BB6Copula,                                     "234"),
        (BB7Copula,                                     "234"),
        (BB8Copula,                                     "234"),
        (BB9Copula,                                     "234"),
        (BB10Copula,                                    "234"),
        
        # Bivariate-only miscellaneous
        (FGMCopula,                                     "2"),
        (PlackettCopula,                                "2"),
        (RafteryCopula,                                 "2"),

        # # Bivariate EV families
        (GalambosCopula,                                "2"),
        (HuslerReissCopula,                             "2"),
        (LogCopula,                                     "2"),
        (MixedCopula,                                   "2"),
        (CuadrasAugeCopula,                             "2"),
        (BC2Copula,                                     "2"),
        (tEVCopula,                                     "2"),
        (MOCopula,                                      "2"),
        (AsymLogCopula,                                 "2"),
        (AsymGalambosCopula,                            "2"),
        (AsymMixedCopula,                               "2"),

        # # Archimax (bivariate only)
        (ArchimaxCopula{2, GumbelGenerator, MixedTail}, "2"),
        (BB4Copula,                                     "2"),
        (BB5Copula,                                     "2"),
    ]

    for d in (2, 3, 4)
        U = rand(rng, d, 100)
        for (CT, dims) in cases
            occursin(string(d), dims) || continue
            avail = Copulas._available_fitting_methods(CT, d)
            if isempty(avail)
                @warn "Empty method list for $CT"
                continue
            end
            @testset "CT=$CT, d=$d" begin
                for m in avail
                    @testset "CT=$CT, d=$d, method=$m" begin
                        @info "CT=$CT, d=$d, method=$m..."
                        fitres = fit(CopulaModel, CT, U; method=m)
                        @test isa(fitres, CopulaModel)
                    end
                end
            end
        end
    end
end

@testitem "Fitting + vcov + StatsBase interfaces" tags=[:fitting, :vcov, :statsbase] begin
    using Test, Random, Distributions, Copulas, StableRNGs, LinearAlgebra, Statistics, StatsBase
    rng = StableRNG(2025)

    reps = [
            # Elliptical
            (GaussianCopula, 2, :mle),
            (GaussianCopula, 3, :mle),

            # Archimedean one parameter
            (ClaytonCopula,  2, :mle),
            (GumbelCopula,   2, :itau),
            (FrankCopula,    2, :mle),
            (JoeCopula,      2, :itau),

            # Archimedean two params
            (BB1Copula,      2, :mle),
            (BB7Copula,      2, :mle),

            # Bivariate Extreme Value
            (GalambosCopula, 2, :mle),
            (HuslerReissCopula, 2, :mle),
        ]

    # helper
    function psd_ok(V; tol=1e-7)
        vals = eigvals(Symmetric(Matrix(V)))
        minimum(vals) >= -tol
    end

    n = 500 # maybe this size is large?

    for (CT, d, method) in reps
        @info "Testing: $CT, d=$d, method=$method..."
        C0 = Copulas._example(CT, d)
        true_θ = _flatten_params(Distributions.params(C0))
        U  = rand(rng, C0, n)
        M  = fit(CopulaModel, CT, U; method=method, vcov=true, derived_measures=false)

        @testset "Core Fitting & Inference" begin
            estimated_θ = StatsBase.coef(M)
            @test estimated_θ ≈ true_θ atol=0.5

            @test isa(StatsBase.vcov(M), AbstractMatrix)
            @test size(StatsBase.vcov(M)) == (StatsBase.dof(M), StatsBase.dof(M))
            @test psd_ok(StatsBase.vcov(M))

            se = StatsBase.stderror(M)
            @test length(se) == StatsBase.dof(M)
            lo, hi = StatsBase.confint(M; level=0.95)
            @test length(lo) == length(hi) == StatsBase.dof(M)
        end

        @testset "Information Criteria" begin
            k = StatsBase.dof(M)
            ll = M.ll
            @test isfinite(StatsBase.aic(M))
            @test isfinite(StatsBase.bic(M))
            @test isfinite(Copulas.aicc(M))
            @test isfinite(Copulas.hqc(M))
            @test aic(M) ≈ 2*k - 2*ll
            @test bic(M) ≈ k*log(n) - 2*ll
        end

        @testset "Residuals API" begin
            R_unif = StatsBase.residuals(M)
            @test size(R_unif) == (d, n)
            @test all(0 .<= R_unif .<= 1)
            R_norm = StatsBase.residuals(M, transform=:normal)
            @test size(R_norm) == (d, n)
            @test abs(mean(R_norm)) < 0.2
            @test 0.8 < std(R_norm) < 1.2
        end

        @testset "Predict API" begin
            sim_data = StatsBase.predict(M, what=:simulate, nsim=100)
            @test size(sim_data) == (d, 100)
            @test all(0 .<= sim_data .<= 1)
            newdata = rand(rng, d, 50)
            preds_cdf = StatsBase.predict(M, newdata=newdata, what=:cdf)
            @test length(preds_cdf) == 50
            @test all(0 .<= preds_cdf .<= 1)
            preds_pdf = StatsBase.predict(M, newdata=newdata, what=:pdf)
            @test length(preds_pdf) == 50
            @test all(preds_pdf .>= 0)
        end
    end

    @testset "API Error Handling" begin
        dummy_copula = IndependentCopula(2)
        M_dummy = Copulas.CopulaModel(dummy_copula, 10, 0.0, :dummy)
        @test_throws ArgumentError StatsBase.residuals(M_dummy)
        @test_throws ArgumentError StatsBase.predict(M_dummy, what=:foo)
    end
end

@testitem "Dependence Metrics" tags=[:metrics] begin
    using Test, Random, Distributions, Copulas, StableRNGs, LinearAlgebra, Statistics, StatsBase, SpecialFunctions, HCubature, QuadGK

    rng = StableRNG(123)
    n_samples = 2000
    test_copulas = [
        (d=3, copula=GumbelCopula(2, 3.5),      description="3D Gumbel with upper tail dependence"),
        (d=3, copula=ClaytonCopula(2, 4.0),     description="Clayton 3D with lower tail dependence"),
        (d=4, copula=GumbelCopula(2, 3.5),      description="Gumbel 4D with lower tail dependence"),
        (d=4, copula=ClaytonCopula(2, 4.0),     description="Clayton 4D with lower tail dependence"),
        (d=2, copula=GalambosCopula(2, 4.0),    description="2D Galambos with lower tail dependence"),
        (d=2, copula=HuslerReissCopula(2, 4.0), description="Husler Reiss 2D with lower tail dependence"),
        (d=2, copula=LogCopula(2, 4.0),         description="2D Logistic with lower tail dependency")
    ]

    @testset "Multivariate Metrics (Copula vs. Data)" begin
        for tc in test_copulas
            C = tc.copula
            d = tc.d
            U = rand(rng, C, n_samples)

            @testset "$(tc.description)" begin
                # Spearman's ρ
                true_rho = Copulas.ρ(C)
                emp_rho = Copulas.ρ(U)
                @test emp_rho ≈ true_rho atol=0.1

                # Kendall's τ
                true_tau = Copulas.τ(C)
                emp_tau = Copulas.τ(U)
                @test emp_tau ≈ true_tau atol=0.1

                # Blomqvist's β
                true_beta = Copulas.β(C)
                emp_beta = Copulas.β(U)
                @test emp_beta ≈ true_beta atol=0.1

                # Gini's γ
                true_gamma = Copulas.γ(C)
                emp_gamma = Copulas.γ(U)
                @test emp_gamma ≈ true_gamma atol=0.15

                # Copula Entropy ι
                true_entropy = Copulas.ι(C)
                emp_entropy = Copulas.ι(U)

                @test true_entropy ≈ emp_entropy atol=0.15
            end
        end
    end

    @testset "Pairwise Metrics (on Data Matrix)" begin
        for tc in test_copulas
            d = tc.d
            d == 2 || continue

            C = tc.copula
            U = rand(rng, C, n_samples)
            X = U'

            @testset "$(tc.description)" begin
                # corblomqvist
                B = Copulas.corblomqvist(X)
                @test B[1,2] ≈ Copulas.β(C) atol=0.1

                # corgini
                G = Copulas.corgini(X)
                @test B[1,2] ≈ Copulas.γ(C) atol=0.1

                # corentropy
                H = Copulas.corentropy(X)
                @test size(H) == (d,d)
                @test H[1,1] == 0.0
                @test isfinite(H[1,2])
            end
        end
    end
end