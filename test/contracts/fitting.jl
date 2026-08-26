@testset "public fitting and model-result contracts" begin
    for (i, case) in pairs(FITTING_CASES)
        @testset "$(case.name)" begin
            source = case.build()
            U = rand(StableRNG(20_000 + i), source, 12)
            family = typeof(source)
            fitted = fit(family, U; method=case.method, case.kwargs...,
                         vcov=false, derived_measures=false)
            @test fitted isa Copulas.Copula{length(source)}

            case.model || continue
            M = fit(CopulaModel, family, U; method=case.method,
                    case.kwargs..., vcov=false, derived_measures=false)
            @test StatsBase.nobs(M) == size(U, 2)
            @test StatsBase.coef(M) isa AbstractVector
            @test StatsBase.coefnames(M) isa AbstractVector
            @test StatsBase.dof(M) == length(StatsBase.coef(M))
            @test StatsBase.deviance(M) == -2 * M.ll
            @test isfinite(StatsBase.aic(M))
            @test isfinite(StatsBase.bic(M))
            @test size(StatsBase.residuals(M)) == size(U)
            @test size(StatsBase.predict(M; what=:simulate, nsim=3)) == (2, 3)
        end
    end
end

@testset "structural and non-fittable public families" begin
    nested = NestedArchimedeanCopula{4}(Copulas.ClaytonGenerator(1.0);
        leaves=[1, 2], children=[ClaytonCopula{2}(2.0)])
    nested_data = rand(StableRNG(20_100), nested, 8)
    @test fit(nested, nested_data; vcov=false, derived_measures=false) isa
          NestedArchimedeanCopula{4}

    non_fittable = (
        LiouvilleCopula{2}(Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
        DiscreteSpectralCopula{2}([0.7 0.3; 0.2 0.8]),
    )
    for C in non_fittable
        U = rand(StableRNG(20_101), C, 4)
        @test_throws Exception fit(typeof(C), U; vcov=false,
                                   derived_measures=false)
    end
end

@testset "complete StatsBase model-result interface" begin
    C = ClaytonCopula{2}(1.5)
    U = [0.2 0.4 0.7 0.8; 0.3 0.6 0.5 0.9]
    M = CopulaModel(C, 4, loglikelihood(C, U), :fixture;
        vcov=reshape([0.04], 1, 1),
        method_details=(θ̂=(θ=1.5,), U=U, null_ll=0.0))
    @test StatsBase.isfitted(M)
    @test StatsBase.nobs(M) == 4
    @test StatsBase.coef(M) == [1.5]
    @test StatsBase.coefnames(M) == ["θ"]
    @test StatsBase.vcov(M) == reshape([0.04], 1, 1)
    @test StatsBase.stderror(M) == [0.2]
    lo, hi = StatsBase.confint(M)
    @test lo[1] < 1.5 < hi[1]
    @test StatsBase.nullloglikelihood(M) == 0
    @test StatsBase.nulldeviance(M) == 0
    @test size(StatsBase.residuals(M)) == size(U)
    @test size(StatsBase.residuals(M; transform=:normal)) == size(U)
    @test length(StatsBase.predict(M; newdata=U, what=:cdf)) == size(U, 2)
    @test length(StatsBase.predict(M; newdata=U, what=:pdf)) == size(U, 2)
end
