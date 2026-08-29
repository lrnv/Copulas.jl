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

@testset "generic empirical EV estimators by dimension" begin
    checked = Set{Tuple{Method,Symbol,Symbol}}()
    selected = Set{Tuple{Method,Symbol,Symbol}}()
    for (U, dimension, kwargs) in ((_FIXTURE_DATA, :bivariate, (; grid=21)),
                                    (_FIXTURE_DATA3, :multivariate, (; degree=1)))
        for method in (:ols, :cfg, :pickands)
            route = (which(Copulas._fit,
                           Tuple{Type{ExtremeValueCopula},typeof(U),Val{method}}),
                     method, dimension)
            push!(selected, route)
            fitted = fit(ExtremeValueCopula, U; method,
                         vcov=false, derived_measures=false, kwargs...)
            @test fitted isa ExtremeValueCopula{size(U, 1)}
            push!(checked, route)
        end
    end
    @test checked == selected
end

const _FITTING_PATH_MODELS = Tuple(fixture.copula for fixture in ROUTING_COPULA_FIXTURES)
_has_fitting_parameters(C) =
    !(C isa Union{IndependentCopula,MCopula,WCopula}) && !isempty(params(C))
_check_parameter_roundtrip(C) =
    !(C isa EmpiricalEVCopula) && !(C isa FGMCopula && length(C) != 2)

@testset "advertised fitting routes beyond the primary family contract" begin
    selected_routes = Set{Any}()
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

        U = rand(StableRNG(30_000 + index), C, 12)
        for method in methods
            route = fitting_route_key(C, U, method)
            push!(selected_routes, route)
            route in PROVEN_FITTING_ROUTES && continue
            test_progress("routing fitting", case.name, method,
                          nameof(CT), d)
            # Routing only needs to exercise the empirical EV estimator. Its
            # high-resolution grid is validated in the fitting contract.
            route_kwargs = C isa EmpiricalEVCopula ?
                (d == 2 ? (; grid=21) : (; degree=1)) : (;)
            fitted = fit(CT, U, method; vcov=false,
                         derived_measures=false, route_kwargs...)
            @test fitted isa Copulas.Copula{d}
            prove_fitting_route!(C, U, method)
            if method === :mle && case.kind === :continuous
                fitted_ll = loglikelihood(fitted, U)
                @test isfinite(fitted_ll)
            end
        end
    end
    @test !isempty(selected_routes)
    @test selected_routes ⊆ PROVEN_FITTING_ROUTES
end
