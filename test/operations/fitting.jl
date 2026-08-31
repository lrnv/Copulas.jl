# Fitting-operation contract: capabilities come from the package itself.
# Every advertised route is executed independently in `routing/fitting.jl`;
# this cheap family-wide pass only proves that method discovery is coherent.
@testset "public fitting method discovery" begin
    for (; case, copula) in COPULA_FIXTURES
        @testset "$(case.name)" begin
            family, d = typeof(copula), length(copula)
            methods = Copulas._available_fitting_methods(family, d)
            @test methods isa Tuple
            @test all(method -> method isa Symbol, methods)
            @test length(unique(methods)) == length(methods)
            if isempty(methods)
                @test_throws ArgumentError Copulas._find_method(family, d, :default)
            else
                @test Copulas._find_method(family, d, :default) in methods
                @test all(method -> Copulas._find_method(family, d, method) === method,
                          methods)
            end
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

@testset "unavailable model metadata" begin
    M = CopulaModel(IndependentCopula{2}(), 10, 0.0, :dummy)
    @test_throws ArgumentError StatsBase.residuals(M)
end

@testset "nested Archimedean fitting validation" begin
    C = NestedArchimedeanCopula(Copulas.ClaytonGenerator(1.0);
        children=[ClaytonCopula{2}(3.0), ClaytonCopula{2}(3.0)])
    U = rand(StableRNG(20_110), C, 4)
    @test_throws Exception Copulas._example(NestedArchimedeanCopula, 4)
    @test_throws ArgumentError fit(CopulaModel, C, U; method=:itau)
    @test_throws ArgumentError fit(CopulaModel, C, U[1:3, :])
    @test_throws ArgumentError fit(CopulaModel, C, zeros(4, 0))
    @test_throws ArgumentError fit(CopulaModel, C, hcat(zeros(4), ones(4)))
    @test_throws ArgumentError fit(CopulaModel, C, fill(NaN, 4, 2))

    rebuild = α -> (θ=exp(α[1]); NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(θ); leaves=[1],
        children=[ClaytonCopula{2}(θ)]))
    @test_throws ArgumentError fit(CopulaModel, rebuild, [log(2.0)], U[1:2, :])
    @test_throws ArgumentError fit(CopulaModel, rebuild, [log(2.0)], zeros(3, 0))
end
