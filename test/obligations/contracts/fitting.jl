# Public-API contract: checks `fit`, positional adapters, non-fittable families,
# `CopulaModel`, and the StatsBase model-result interface.
@testset "public fitting registry is exhaustive" begin
    # These structural families use explicit tests below because they require a
    # constructor, an instance, or intentionally expose no fitting operation.
    exceptional = Set((
        "generic Archimedean", "nested Archimedean", "Liouville", "Tawn",
        "generic EV", "discrete spectral",
    ))
    canonical(name) = replace(name, " bound" => "")
    @test Set(canonical(case.name) for case in FITTING_CASES) ==
          setdiff(Set(canonical(case.name) for case in COPULA_CASES), exceptional)
end

@testset "public fitting and model-result contracts" begin
    for (i, fixture) in pairs(FITTING_FIXTURES)
        case, source = fixture.case, fixture.copula
        @testset "$(case.name)" begin
            test_progress("contracts", "fitting", case.name)
            U = rand(StableRNG(20_000 + i), source, 12)
            family = typeof(source)
            fitted = fit(family, U; method=case.method, case.kwargs...,
                         vcov=false, derived_measures=false)
            @test fitted isa Copulas.Copula{length(source)}
            resolved_method = Copulas._find_method(
                family, length(source), case.method)
            prove_fitting_route!(source, U, resolved_method)
            if resolved_method === :mle && !isempty(params(source))
                fitted_ll = loglikelihood(fitted, U)
                source_ll = loglikelihood(source, U)
                if isfinite(fitted_ll) && isfinite(source_ll)
                    @test fitted_ll >= source_ll - 1e-6
                end
            end
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

@testset "positional fitting adapters" begin
    U = rand(StableRNG(20_050), ClaytonCopula{2}(1.0), 12)
    @test fit(ClaytonCopula{2}, U, :itau; vcov=false,
              derived_measures=false) isa ClaytonCopula{2}
    @test fit(CopulaModel, ClaytonCopula{2}, U, :itau; vcov=false,
              derived_measures=false) isa CopulaModel

    D = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    X = rand(StableRNG(20_051), D, 12)
    family = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
    @test fit(family, X, :itau; vcov=false,
              derived_measures=false) isa SklarDist
    @test fit(CopulaModel, family, X, :itau; vcov=false,
              derived_measures=false) isa CopulaModel
end

@testset "empirical fitting routes equal their defining estimators" begin
    U = _FIXTURE_DATA
    point = [0.43, 0.71]
    estimators = (
        (EmpiricalCopula, :deheuvels, NamedTuple(),
         () -> EmpiricalCopula(U)),
        (BetaCopula, :beta, NamedTuple(),
         () -> BetaCopula(U)),
        (CheckerboardCopula, :exact, (; m=2),
         () -> CheckerboardCopula(U; m=2)),
        (BernsteinCopula, :bernstein, (; m=2),
         () -> BernsteinCopula(U; m=2)),
        (EmpiricalEVCopula{2}, :cfg, (; grid=21),
         () -> EmpiricalEVCopula(U; method=:cfg, grid=21)),
    )
    for (family, method, kwargs, direct) in estimators
        fitted = fit(family, U; method=method, kwargs...,
                     vcov=false, derived_measures=false)
        expected = direct()
        @test typeof(fitted) == typeof(expected)
        @test params(fitted) == params(expected)
        @test cdf(fitted, point) ≈ cdf(expected, point)
    end

    U3 = _FIXTURE_DATA3
    fitted3 = fit(EmpiricalEVCopula, U3; method=:cfg, degree=1,
                  vcov=false, derived_measures=false)
    expected3 = EmpiricalEVCopula(U3; method=:cfg, degree=1)
    @test typeof(fitted3) == typeof(expected3)
    @test params(fitted3) == params(expected3)
    @test cdf(fitted3, [0.41, 0.59, 0.73]) ≈
          cdf(expected3, [0.41, 0.59, 0.73])
end

@testset "structural and non-fittable public families" begin
    nested = NestedArchimedeanCopula{4}(Copulas.ClaytonGenerator(1.0);
        leaves=[1, 2], children=[ClaytonCopula{2}(2.0)])
    nested_data = rand(StableRNG(20_100), nested, 8)
    @test fit(nested, nested_data; vcov=false, derived_measures=false) isa
          NestedArchimedeanCopula{4}

    generic_data = rand(StableRNG(20_102), ClaytonCopula{2}(1.0), 64)
    @test fit(ArchimedeanCopula, generic_data; method=:gnz2011, vcov=false,
              derived_measures=false) isa ArchimedeanCopula{2}
    @test fit(ExtremeValueCopula, generic_data; method=:ols,
              vcov=false, derived_measures=false) isa ExtremeValueCopula{2}

    non_fittable = (
        LiouvilleCopula{2}(Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
        ExtremeValueCopula{2}(DiscreteSpectralTail([0.7 0.3; 0.2 0.8])),
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
    lo80, hi80 = StatsBase.confint(M; level=0.8)
    @test lo[1] < lo80[1] < 1.5 < hi80[1] < hi[1]
    @test StatsBase.nullloglikelihood(M) == 0
    @test StatsBase.nulldeviance(M) == 0
    @test size(StatsBase.residuals(M)) == size(U)
    @test size(StatsBase.residuals(M; transform=:normal)) == size(U)
    @test_throws ArgumentError StatsBase.residuals(M; transform=:invalid)
    @test length(StatsBase.predict(M; newdata=U, what=:cdf)) == size(U, 2)
    @test length(StatsBase.predict(M; newdata=U, what=:pdf)) == size(U, 2)
    @test size(StatsBase.predict(M; what=:simulate)) == size(U)
    @test_throws ArgumentError StatsBase.predict(M; what=:cdf)
    @test_throws ArgumentError StatsBase.predict(M; what=:invalid)

    M0 = CopulaModel(EmpiricalCopula(U), 4, 0.0, :empirical)
    @test StatsBase.dof(M0) == 0
    @test isempty(StatsBase.coef(M0))
    @test isempty(StatsBase.coefnames(M0))
    @test StatsBase.vcov(M0) === nothing
    @test StatsBase.stderror(M0) === nothing
    @test StatsBase.confint(M0) === nothing
    @test StatsBase.aic(M0) == StatsBase.bic(M0) == 0
end
