# Routing obligation: exercises representative Sklar, empirical, covariance,
# optimizer, and model-result fitting routes beyond the universal fit contract.
@testset "public Sklar fitting path" begin
    source = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    data = rand(StableRNG(111), source, 16)
    fitted = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                 copula_method=:itau, vcov=false, derived_measures=false)
    @test fitted isa SklarDist
    @test fitted.C isa ClaytonCopula{2}

    model = fit(CopulaModel,
        SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
        copula_method=:itau, vcov=false, derived_measures=false)
    @test model.result isa SklarDist
    @test StatsBase.nobs(model) == size(data, 2)

    ecdf_fit = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                   sklar_method=:ecdf, copula_method=:itau, vcov=false,
                   derived_measures=false)
    @test ecdf_fit isa SklarDist
end

@testset "public covariance fitting option" begin
    U = rand(StableRNG(112), ClaytonCopula{2}(1.0), 20)
    model = fit(CopulaModel, ClaytonCopula{2}, U; method=:mle,
                vcov=true, vcov_method=:hessian, derived_measures=false)
    @test StatsBase.vcov(model) isa AbstractMatrix
    @test size(StatsBase.vcov(model)) == (StatsBase.dof(model), StatsBase.dof(model))
    @test_throws ArgumentError fit(CopulaModel, ClaytonCopula{2}, U;
        method=:mle, vcov=true, vcov_method=:invalid, derived_measures=false)
end

const _FITTING_PATH_MODELS = Tuple(case.build() for case in COPULA_CASES)
const _PRIMARY_FITTING_METHOD = Dict(case.name => case.method for case in FITTING_CASES)

_has_fitting_parameters(C) =
    !(C isa Union{IndependentCopula,MCopula,WCopula}) && !isempty(params(C))
_check_parameter_roundtrip(C) =
    !(C isa EmpiricalEVCopula) && !(C isa FGMCopula && length(C) != 2)

@testset "advertised fitting routes beyond the primary family contract" begin
    for (index, (case, C)) in enumerate(zip(COPULA_CASES, _FITTING_PATH_MODELS))
        CT, d = typeof(C), length(C)
        methods = Copulas._available_fitting_methods(CT, d)

        if :mle in methods && _has_fitting_parameters(C) &&
           _check_parameter_roundtrip(C)
            bounded = params(C)
            restored = Copulas._rebound_params(
                CT, d, Copulas._unbound_params(CT, d, bounded))
            @test all(key -> getfield(bounded, key) ≈ getfield(restored, key),
                      keys(bounded))
        end

        primary = get(_PRIMARY_FITTING_METHOD, case.name, nothing)
        remaining = filter(!=(primary), methods)
        isempty(remaining) && continue

        U = rand(StableRNG(30_000 + index), C, 12)
        for method in remaining
            fitted = fit(CT, U, method; vcov=false,
                         derived_measures=false)
            @test fitted isa Copulas.Copula{d}
            fitted_statistic = fitting_statistic(Val(method), fitted)
            isnothing(fitted_statistic) && continue
            sample_statistic = fitting_statistic(Val(method), U)
            @test isapprox(fitted_statistic, sample_statistic;
                           atol=2e-5, rtol=2e-5)
        end
    end
end
