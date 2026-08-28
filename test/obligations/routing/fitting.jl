# Routing obligation: exercises representative Sklar, empirical, covariance,
# optimizer, and model-result fitting routes beyond the universal fit contract.
@testset "public Sklar fitting path" begin
    source = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    data = rand(StableRNG(111), source, 8)
    test_progress("routing fitting", "Sklar IFM")
    fitted = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                 copula_method=:itau, vcov=false, derived_measures=false)
    @test fitted isa SklarDist
    @test fitted.C isa ClaytonCopula{2}

    test_progress("routing fitting", "Sklar model")
    model = fit(CopulaModel,
        SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
        copula_method=:itau, vcov=false, derived_measures=false)
    @test model.result isa SklarDist
    @test StatsBase.nobs(model) == size(data, 2)

    test_progress("routing fitting", "Sklar ECDF")
    ecdf_fit = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                   sklar_method=:ecdf, copula_method=:itau, vcov=false,
                   derived_measures=false)
    @test ecdf_fit isa SklarDist
end

@testset "public covariance fitting option" begin
    U = rand(StableRNG(112), ClaytonCopula{2}(1.0), 8)
    test_progress("routing fitting", "covariance hessian")
    model = fit(CopulaModel, ClaytonCopula{2}, U; method=:itau,
                vcov=true, vcov_method=:hessian, derived_measures=false)
    @test StatsBase.vcov(model) isa AbstractMatrix
    @test size(StatsBase.vcov(model)) == (StatsBase.dof(model), StatsBase.dof(model))
    test_progress("routing fitting", "invalid covariance method")
    @test_throws ArgumentError fit(CopulaModel, ClaytonCopula{2}, U;
        method=:itau, vcov=true, vcov_method=:invalid, derived_measures=false)
end

const _FITTING_PATH_MODELS = Tuple(case.build() for case in ROUTING_COPULA_CASES)
const _PRIMARY_FITTING_METHOD = Dict(case.name => begin
    C = case.build()
    Copulas._find_method(typeof(C), length(C), case.method)
end for case in FITTING_CASES)
const _PRIMARY_FITTING_TYPE = Dict(case.name => typeof(case.build())
                                   for case in FITTING_CASES)
_canonical_fitting_name(name) = replace(name,
    " bivariate" => "", " multivariate" => "")

_has_fitting_parameters(C) =
    !(C isa Union{IndependentCopula,MCopula,WCopula}) && !isempty(params(C))
_check_parameter_roundtrip(C) =
    !(C isa EmpiricalEVCopula) && !(C isa FGMCopula && length(C) != 2)

@testset "advertised fitting routes beyond the primary family contract" begin
    seen_routes = Set{Any}()
    for (index, (case, C)) in
        enumerate(zip(ROUTING_COPULA_CASES, _FITTING_PATH_MODELS))
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

        canonical_name = _canonical_fitting_name(case.name)
        primary = get(_PRIMARY_FITTING_TYPE, canonical_name, nothing) === CT ?
            get(_PRIMARY_FITTING_METHOD, canonical_name, nothing) : nothing
        remaining = filter(!=(primary), methods)
        isempty(remaining) && continue

        U = rand(StableRNG(30_000 + index), C, 12)
        for method in remaining
            route = (which(Copulas._fit,
                           Tuple{Type{CT},typeof(U),Val{method}}),
                     method, d == 2 ? :bivariate : :multivariate)
            route in seen_routes && continue
            push!(seen_routes, route)
            test_progress("routing fitting", case.name, method,
                          nameof(CT), d)
            # Routing only needs to exercise the empirical EV estimator. Its
            # high-resolution grid is validated in the fitting contract.
            route_kwargs = C isa EmpiricalEVCopula ?
                (d == 2 ? (; grid=21) : (; degree=1)) : (;)
            fitted = fit(CT, U, method; vcov=false,
                         derived_measures=false, route_kwargs...)
            @test fitted isa Copulas.Copula{d}
            if method === :mle && case.kind === :continuous
                fitted_ll = loglikelihood(fitted, U)
                @test isfinite(fitted_ll)
            end
        end
    end
    @test !isempty(seen_routes)
end
