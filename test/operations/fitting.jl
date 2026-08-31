# Fitting-operation proof: exercises representative Sklar, empirical, covariance,
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

@testset "extreme-value MLE accepts boundary starts" begin
    U = [0.10 0.25 0.40 0.55 0.70 0.85;
         0.15 0.20 0.45 0.60 0.75 0.90]
    for CT in (CuadrasAugeCopula, LogCopula)
        fitted = fit(CT, U, :mle; start=1.0)
        @test fitted isa Copulas.Copula
        @test all(isfinite, params(fitted))
    end
end

# A fitting route is the complete internal composition, not merely `_fit`.
# Generic fitting additionally depends on the example, parameter transform,
# and reconstruction methods selected for the concrete family.
function fitting_route_key(C, U, method)
    Base.@nospecialize C U method
    CT, d = typeof(C), length(C)
    components = Any[
        which(Copulas._available_fitting_methods, Tuple{Type{CT},Int}),
        which(Copulas._find_method, Tuple{Type{CT},Int,Symbol}),
        which(Copulas._fit, Tuple{Type{CT},typeof(U),Val{method}}),
    ]
    applicable(Copulas._example, CT, d) &&
        push!(components, which(Copulas._example, Tuple{Type{CT},Int}))
    bounded = params(C)
    topology = (keys(bounded), map(values(bounded)) do value
        value isa AbstractArray ? (typeof(value), size(value)) : typeof(value)
    end)
    component_type = C isa ArchimedeanCopula ? typeof(C.G) :
                     C isa ExtremeValueCopula ? typeof(C.tail) : nothing
    bounds = !isnothing(component_type) &&
             applicable(Copulas._θ_bounds, component_type, d) ?
             (which(Copulas._θ_bounds, Tuple{Type{component_type},Int}),
              Copulas._θ_bounds(component_type, d)) : nothing
    # Empirical EV fits reconstruct their non-parametric tail directly from
    # the observations; the generic EV forwarding method is technically
    # applicable but its parametric tail transform is not part of that route.
    # Multivariate FGM uses its dedicated constrained MLE directly; its
    # bivariate-only scalar transform is applicable by signature but rejects d>2.
    if !(C isa EmpiricalEVCopula) &&
       !(C isa FGMCopula && d != 2) && !isempty(bounded) &&
       applicable(Copulas._unbound_params, CT, d, bounded)
        unbound = Copulas._unbound_params(CT, d, bounded)
        push!(components,
              which(Copulas._unbound_params,
                    Tuple{Type{CT},Int,typeof(bounded)}))
        applicable(Copulas._rebound_params, CT, d, unbound) &&
            push!(components,
                  which(Copulas._rebound_params,
                        Tuple{Type{CT},Int,typeof(unbound)}))
        applicable(Copulas._fit_copula, CT, d, bounded, C) &&
            push!(components,
                  which(Copulas._fit_copula,
                        Tuple{Type{CT},Int,typeof(bounded),typeof(C)}))
    end
    # `which` already distinguishes genuinely dimension-specific dispatches:
    # adding the dimension itself would instead execute the same generic
    # algorithm once per representation. Parameter topology and bounds retain
    # the non-dispatch differences that affect generic reconstruction.
    return (Tuple(components), method, topology, bounds)
end

_has_fitting_parameters(C) =
    !(C isa Union{IndependentCopula,MCopula,WCopula}) && !isempty(params(C))
_check_parameter_roundtrip(C) =
    !(C isa EmpiricalEVCopula) && !(C isa FGMCopula && length(C) != 2)

@testset "all distinct advertised fitting routes" begin
    selected_routes = Set{Any}()
    tested_routes = Set{Any}()
    for (index, fixture) in enumerate(ROUTING_COPULA_FIXTURES)
        case, C = fixture.case, fixture.copula
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

        # Route selection depends on the matrix type, not on sampled values.
        # Delay sampling until an actually new route must be executed, so
        # families sharing the same fitting composition do not repeat it.
        route_data = fill(0.5, d, 2)
        for method in methods
            route = fitting_route_key(C, route_data, method)
            push!(selected_routes, route)
            route in tested_routes && continue
            test_progress("routing fitting", case.name, method,
                          nameof(CT), d)
            U = rand(StableRNG(30_000 + index), C, 12)
            # Routing only needs to exercise the empirical EV estimator. Its
            # high-resolution grid is validated in the fitting contract.
            route_kwargs = C isa EmpiricalEVCopula ?
                (d == 2 ? (; grid=21) : (; degree=1)) : (;)
            fitted = fit(CT, U, method; vcov=false,
                         derived_measures=false, route_kwargs...)
            @test fitted isa Copulas.Copula{d}
            push!(tested_routes, route)
            if method === :mle && is_absolutely_continuous(C)
                fitted_ll = loglikelihood(fitted, U)
                @test isfinite(fitted_ll)
                # source_ll = loglikelihood(C, U)
                # if isfinite(source_ll)
                #     @test fitted_ll >= source_ll - 1e-6
                # end
            end
        end
    end
    @test !isempty(selected_routes)
    @test tested_routes == selected_routes
end


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


# Fitting-operation proof for parameterizations. Public
# route availability and result interfaces are covered by this operation and
# the final routing inventory;
# this file checks that unconstrained coordinates map bijectively to the
# intended constrained parameter space.
@testset "asymmetric Mixed feasible fitting parameterization" begin
    for (i, z) in pairs(([-3.0, 3.0], [3.0, -3.0], [0.0, 0.5]))
        p = Copulas._rebound_params(Copulas.AsymMixedTail, 2, z)
        i == 1 && @test p.θ₂ > 0
        i == 2 && @test p.θ₂ < 0
        @test p.θ₁ >= 0
        @test p.θ₁ + p.θ₂ <= 1
        @test p.θ₁ + 2p.θ₂ <= 1
        @test p.θ₁ + 3p.θ₂ >= 0
        @test Copulas._unbound_params(Copulas.AsymMixedTail, 2, p) ≈ z
        @test Copulas.AsymMixedTail(p.θ₁, p.θ₂) isa Copulas.AsymMixedTail
    end

    # The reverse direction starts from independently chosen feasible model
    # parameters, so this is not merely a circular composition of one map.
    for p in ((; θ₁=0.25, θ₂=0.10), (; θ₁=1.20, θ₂=-0.30))
        restored = Copulas._rebound_params(Copulas.AsymMixedTail, 2,
            Copulas._unbound_params(Copulas.AsymMixedTail, 2, p))
        @test restored.θ₁ ≈ p.θ₁ atol=3e-11 rtol=3e-11
        @test restored.θ₂ ≈ p.θ₂ atol=3e-11 rtol=3e-11
    end

    example = Copulas._example(Copulas.AsymMixedCopula, 2)
    p = params(example)
    @test example isa Copulas.AsymMixedCopula
    @test keys(p) == (:θ₁, :θ₂)
    @test !iszero(p.θ₂)
end
