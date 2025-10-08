
@testitem "Fitting + vcov + StatsBase interfaces" tags=[:fitting, :vcov, :statsbase] begin
    using Test, Random, Distributions, Copulas, StableRNGs, LinearAlgebra, Statistics, StatsBase
    rng = StableRNG(2025)
    function _flatten_params(p::NamedTuple)
        if haskey(p, :Σ)
            Σ = p.Σ
            return [Σ[i, j] for i in 1:size(Σ,1)-1 for j in (i+1):size(Σ,2)]
        end
        vals = Any[]
        for v in values(p)
            if isa(v, Number)
                push!(vals, Float64(v))
            else
                append!(vals, vec(Float64.(v)))
            end
        end
        return vals
    end
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
            (BB6Copula,      2, :mle),
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