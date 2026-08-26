@testset "public fitting and model-result contracts" begin
    for (i, case) in pairs(FITTING_CASES)
        @testset "$(case.name)" begin
            source = case.family(2, 1.5)
            U = rand(StableRNG(20_000 + i), source, 12)
            fitted = fit(case.family, U; method=case.method, vcov=false,
                         derived_measures=false)
            @test fitted isa Copulas.Copula{2}

            M = fit(CopulaModel, case.family, U; method=case.method,
                    vcov=false, derived_measures=false)
            @test StatsBase.nobs(M) == size(U, 2)
            @test StatsBase.coef(M) isa AbstractVector
            @test StatsBase.coefnames(M) isa AbstractVector
            @test StatsBase.dof(M) == length(StatsBase.coef(M))
            @test StatsBase.deviance(M) == -2M.ll
            @test isfinite(StatsBase.aic(M))
            @test isfinite(StatsBase.bic(M))
            @test size(StatsBase.residuals(M)) == size(U)
            @test size(StatsBase.predict(M; what=:simulate, nsim=3)) == (2, 3)
        end
    end
end
