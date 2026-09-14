# Fitting-operation proof: exercises representative Sklar, empirical, covariance,
# optimizer, and model-result fitting routes beyond the universal fit contract.
struct PublicFitProtocolProbe <: Copulas.Copula{2} end
Copulas.fitting_methods(::Type{PublicFitProtocolProbe}, ::Val{2}) = (:probe,)
function Copulas.fit_copula(::Type{PublicFitProtocolProbe}, data, ::Val{:probe}; offset=0.0)
    estimate = Statistics.mean(data) + offset
    return IndependentCopula{2}(),
           (; free_parameters=(; estimate), fixed_parameters=(; offset),
              converged=true, iterations=1)
end

@testset "public downstream fitting protocol" begin
    U = [0.2 0.4 0.6; 0.3 0.5 0.7]
    fitted = fit(PublicFitProtocolProbe, U; method=:probe, offset=0.1)
    model = fit(CopulaModel, PublicFitProtocolProbe, U;
                method=:probe, offset=0.1)
    @test fitted isa IndependentCopula{2}
    @test StatsBase.coef(model) == [Statistics.mean(U) + 0.1]
    @test StatsBase.coefnames(model) == ["estimate"]
    @test StatsBase.dof(model) == 1
    @test model.method_details.fixed_parameters == (; offset=0.1)
    @test !("offset" in StatsBase.coefnames(model))
    @test model.iterations == 1
    @test_throws ArgumentError infer(model)
    inference = infer(model; method=:bootstrap, nresamples=3,
                      rng=StableRNG(48_099))
    @test inference isa CopulaInference
    @test size(StatsBase.vcov(inference)) == (StatsBase.dof(model),
                                              StatsBase.dof(model))
end

@testset "public Sklar fitting path" begin
    source = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    data = rand(StableRNG(111), source, 30)
    fitted = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                 copula_method=:itau, derived_measures=false)
    @test fitted isa SklarDist
    @test fitted.C isa ClaytonCopula{2}

    model = fit(CopulaModel,
        SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
        copula_method=:itau, derived_measures=false)
    @test fitteddistribution(model) isa SklarDist
    @test StatsBase.nobs(model) == size(data, 2)
    @test any(startswith("margin_"), StatsBase.coefnames(model))
    inference = infer(model; method=:bootstrap, nresamples=3,
                      rng=StableRNG(114))
    @test size(StatsBase.vcov(inference)) ==
          (StatsBase.dof(model), StatsBase.dof(model))
    @test size(StatsBase.vcov(inference; component=:copula), 1) == 1
    @test size(StatsBase.vcov(inference; component=:margins), 1) ==
          StatsBase.dof(model) - 1
    @test_throws ArgumentError infer(model; method=:hessian)

    ecdf_fit = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                   sklar_method=:ecdf, copula_method=:itau, derived_measures=false)
    @test ecdf_fit isa SklarDist

    default_model = fit(CopulaModel,
        SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
        derived_measures=false)
    @test default_model.method === :mle
    @test default_model.method_details.sklar_method === :ifm

    empirical_model = fit(CopulaModel,
        SklarDist{EmpiricalCopula,Tuple{Normal,Exponential}}, data;
        derived_measures=false)
    @test empirical_model.method === :deheuvels
end

@testset "MLE and MPL input semantics" begin
    X = [2.0 8.0 1.0 5.0 3.0 7.0;
         4.0 1.0 6.0 2.0 5.0 3.0]
    U = pseudos(X)

    default_fit = fit(CopulaModel, ClaytonCopula{2}, U;
        derived_measures=false)
    mle_fit = fit(CopulaModel, ClaytonCopula{2}, U; method=:mle,
        pseudo_values=true, derived_measures=false)
    mpl_fit = fit(CopulaModel, ClaytonCopula{2}, X; method=:mpl,
        pseudo_values=false, derived_measures=false)
    normalized_mpl = fit(CopulaModel, ClaytonCopula{2}, X; method=:mle,
        pseudo_values=false, derived_measures=false)
    normalized_mle = @test_logs (:warn, r"method=:mpl requires raw observations") fit(
        CopulaModel, ClaytonCopula{2}, U; method=:mpl, pseudo_values=true,
        derived_measures=false)

    @test default_fit.method === :mle
    @test mle_fit.method === :mle
    @test mpl_fit.method === :mpl
    @test normalized_mpl.method === :mpl
    @test normalized_mle.method === :mle
    @test params(default_fit.result) == params(mle_fit.result)
    @test params(mpl_fit.result) == params(mle_fit.result)
    @test params(normalized_mpl.result) == params(mpl_fit.result)
    @test params(normalized_mle.result) == params(mle_fit.result)
    @test mpl_fit.method_details.requested_method === :mpl
    @test normalized_mpl.method_details.requested_method === :mle
    @test normalized_mle.method_details.requested_method === :mpl
    @test mpl_fit.method_details.pseudo_values === false
    @test mpl_fit.method_details.fitting_data_pseudo_values === true
    @test normalized_mle.method_details.pseudo_values === true
    @test mpl_fit.method_details.U == U
    @test_throws ArgumentError Copulas._find_method(EmpiricalCopula, 2, :mpl)
    @test_throws ArgumentError Copulas._find_method(ClaytonCopula{2}, 2, :mpl)
end

@testset "estimation and inference are separate" begin
    U = rand(StableRNG(112), ClaytonCopula{2}(1.0), 8)
    fitted = fit(ClaytonCopula{2}, U; method=:itau)
    model = fit(CopulaModel, ClaytonCopula{2}, U; method=:itau,
                derived_measures=false)
    @test fitted isa ClaytonCopula{2}
    @test !(fitted isa CopulaModel)
    @test params(fitted) == params(fitteddistribution(model))
    inference = infer(model; method=:bootstrap, nresamples=3,
                      rng=StableRNG(113))
    @test inference isa CopulaInference
    @test size(StatsBase.vcov(inference)) ==
          (StatsBase.dof(model), StatsBase.dof(model))
    @test Copulas.inference_diagnostics(inference).nresamples == 3
    @test !applicable(StatsBase.vcov, model)
    @test_throws ArgumentError fit(CopulaModel, ClaytonCopula{2}, U;
        method=:itau, vcov=true, derived_measures=false)
    @test_throws ArgumentError fit(CopulaModel, ClaytonCopula{2}, U;
        method=:itau, vcov_method=:bootstrap, derived_measures=false)
    @test_throws ArgumentError infer(model; method=:invalid)
end

@testset "bivariate rotation fitting preserves the requested type" begin
    U = [0.12 0.24 0.41 0.63 0.78 0.91;
         0.73 0.88 0.42 0.59 0.11 0.26]
    targets = (Rotated90Copula{2,ClaytonCopula{2}},
               Rotated180Copula{2,ClaytonCopula{2}},
               Rotated270Copula{2,ClaytonCopula{2}})
    families = (Rotated90Copula, Rotated180Copula, Rotated270Copula)
    masks = ((true, false), (true, true), (false, true))
    for (target, family, mask) in zip(targets, families, masks)
        fitted = fit(target, U; method=:itau, derived_measures=false)
        @test fitted isa family
        @test Copulas.basecopula(fitted) isa ClaytonCopula{2}
        @test Copulas.flipmask(fitted) == mask
    end
end

@testset "bivariate Student rank matching" begin
    source = TCopula{2}(4.0, [1.0 0.55; 0.55 1.0])
    U = rand(StableRNG(316), source, 2_000)
    fitted = fit(TCopula{2}, U; method=:itau_irho, derived_measures=false)
    @test fitted isa TCopula{2}
    @test fitted.Σ[1, 2] ≈ sinpi(StatsBase.corkendall(U')[1, 2] / 2)
    @test Copulas.ρ(fitted) ≈ StatsBase.corspearman(U')[1, 2] atol=2e-3
end

@testset "Gaussian MLE maximizes the copula likelihood" begin
    # Regression test for #477.
    z1 = [
        -0.789121, -0.167787,  1.487925,  0.393974,  1.120231,
         0.777104, -0.436464,  0.741952, -0.116595, -0.122389,
         0.297167, -1.325930,  1.392166, -0.471090,  1.200269,
         0.336321,  1.732033, -0.459969, -0.111795,  0.537769,
    ]
    z2 = [
        -2.702032,  0.644028,  2.185593,  1.223794,  1.214885,
         0.142574,  0.963670,  1.345007,  0.234323, -0.156080,
        -0.313183, -1.993197,  1.822312, -2.507589,  0.129987,
         0.232541,  1.625898,  1.431677,  0.837426, -0.122637,
    ]
    N01 = Normal()
    U = Matrix{Float64}(undef, 2, length(z1))
    U[1, :] .= cdf.(N01, z1)
    U[2, :] .= cdf.(N01, z2)
    fitted = fit(GaussianCopula, U; method=:mle, derived_measures=false,)
    @test fitted isa GaussianCopula{2}
    @test fitted.Σ[1, 2] ≈ 0.5561662371678145 atol=1e-8
    @test loglikelihood(fitted, U) ≈ 5.140188516707351 atol=1e-10
end

@testset "Student MLE profiles degrees of freedom" begin
    d = 3
    ρ = 0.55
    Σ = [ρ^abs(i - j) for i in 1:d, j in 1:d]
    source = TCopula(4.0, copy(Σ))
    U = rand(StableRNG(477), source, 1_000)
    model = fit(CopulaModel, TCopula, U; method=:mle, derived_measures=false,)
    fitted = fitteddistribution(model)
    θ = params(fitted)
    @test model.converged
    @test fitted isa TCopula{3}
    @test θ.ν > 0
    @test isfinite(θ.ν)
    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(θ.Σ),)
    @test maximum(abs.(LinearAlgebra.diag(θ.Σ) .- 1),) < 1e-12
    # The Student profile contains the Gaussian copula as ν = Inf,
    # so its fitted likelihood must not be worse than that endpoint.
    gaussian = fit(GaussianCopula,U; method=:mle, derived_measures=false,)
    @test loglikelihood(fitted, U) >= loglikelihood(gaussian, U) - 1e-8
    @test model.method_details.profile_upper >= 0.5
    @test model.method_details.profile_expansions >= 0
end

@testset "Student MLE can estimate ν below two" begin
    d = 3
    ρ = 0.5
    Σ = [ρ^abs(i - j) for i in 1:d, j in 1:d]
    source = TCopula(1.0, copy(Σ))
    U = rand(StableRNG(478), source, 1_500)
    model = fit(CopulaModel, TCopula, U; method=:mle, derived_measures=false,)
    fitted = fitteddistribution(model)
    @test model.converged
    @test 0 < params(fitted).ν < 2
    @test model.method_details.profile_expansions >= 1
    @test model.method_details.profile_upper > 0.5
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
                         derived_measures=false, kwargs...)
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
function fitting_execution_route_key(C, U, method)
    Base.@nospecialize C U method

    CT = typeof(C)
    d = length(C)
    fit_method = which(Copulas._fit, Tuple{Type{CT},typeof(U),Val{method}})

    parameter_dof = method === :mle ? Copulas._parameter_dof(params(C)) : nothing

    dimension = d == 2 ? :bivariate : :multivariate

    return (
        fit_method,
        method,
        dimension,
        parameter_dof,
    )
end

_has_fitting_parameters(C) =
    !(C isa Union{IndependentCopula,MCopula,WCopula}) && !isempty(params(C))
_check_parameter_roundtrip(C) =
    !(C isa EmpiricalEVCopula) && !(C isa FGMCopula && length(C) != 2)

function test_mle_parameter_plumbing(C)
    Base.@nospecialize C

    CT = typeof(C)
    d = length(C)
    bounded = params(C)
    unbounded = Copulas._unbound_params(CT, d, bounded)
    restored = Copulas._rebound_params(CT, d, unbounded)

    @test keys(restored) == keys(bounded)

    @test all(key -> getfield(bounded, key) ≈ getfield(restored, key), keys(bounded))

    if applicable(Copulas._example, CT, d)
        example = Copulas._example(CT, d)
        @test example isa Copulas.Copula{d}
    end

    return nothing
end

@testset "MLE parameter plumbing for every advertised family" begin
    for i in eachindex(BASE_COPULA_CASES)
        fixture = COPULA_FIXTURES[i]
        case, C = fixture.case, fixture.copula
        CT = typeof(C)
        d = length(C)

        methods = Copulas._available_fitting_methods(CT, d)

        :mle in methods || continue
        _has_fitting_parameters(C) || continue
        _check_parameter_roundtrip(C) || continue

        @testset "$(case.name)" begin
            test_mle_parameter_plumbing(C)
        end
    end
end

@testset "one execution per fitting engine" begin
    selected_routes = Set{Any}()
    tested_routes = Set{Any}()

    for (index, fixture) in enumerate(COPULA_FIXTURES)
        case, C = fixture.case, fixture.copula
        CT = typeof(C)
        d = length(C)

        methods = Copulas._available_fitting_methods(CT, d)

        # The route depends on the matrix type, not on its values.
        route_data = fill(0.5, d, 2)

        for method in methods
            route = fitting_execution_route_key(C, route_data, method)
            push!(selected_routes, route)
            route in tested_routes && continue

            U = rand(StableRNG(30_000 + index), C, 12)
            route_kwargs = C isa EmpiricalEVCopula ? (d == 2 ? (; grid=21) : (; degree=1)) :
                C isa SurvivalCopula ? (; flips=C.flipmask) : (;)
            fitted = fit( CT, U, method; derived_measures=false, route_kwargs...)
            @test fitted isa Copulas.Copula{d}

            if method === :mle && is_absolutely_continuous(C)
                fitted_ll = loglikelihood(fitted, U)
                @test isfinite(fitted_ll)
            end

            push!(tested_routes, route)
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
            @test Copulas.fitting_methods(family, Val(d)) == methods
            @test methods isa Tuple
            @test all(method -> method isa Symbol, methods)
            @test length(unique(methods)) == length(methods)
            if isempty(methods)
                @test_throws ArgumentError Copulas._find_method(family, d, :default)
            else
                @test Copulas._find_method(family, d, :default) in methods
                @test all(method -> Copulas._find_method(family, d, method) === method,
                          methods)
                if :mle in methods
                    @test Copulas._default_fitting_method(family, d) === :mle
                else
                    @test Copulas._default_fitting_method(family, d) === first(methods)
                end
            end
        end
    end
end

@testset "positional fitting adapters" begin
    U = rand(StableRNG(20_050), ClaytonCopula{2}(1.0), 12)
    @test fit(ClaytonCopula{2}, U, :itau; derived_measures=false) isa ClaytonCopula{2}
    @test fit(CopulaModel, ClaytonCopula{2}, U, :itau; derived_measures=false) isa CopulaModel

    D = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    X = rand(StableRNG(20_051), D, 12)
    family = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
    @test fit(family, X, :itau; derived_measures=false) isa SklarDist
    @test fit(CopulaModel, family, X, :itau; derived_measures=false) isa CopulaModel
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
                     derived_measures=false)
        expected = direct()
        @test typeof(fitted) == typeof(expected)
        @test params(fitted) == params(expected)
        @test cdf(fitted, point) ≈ cdf(expected, point)
    end

    U3 = _FIXTURE_DATA3
    fitted3 = fit(EmpiricalEVCopula, U3; method=:cfg, degree=1,
                  derived_measures=false)
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
    @test fit(nested, nested_data; derived_measures=false) isa
          NestedArchimedeanCopula{4}

    generic_data = rand(StableRNG(20_102), ClaytonCopula{2}(1.0), 64)
    @test fit(ArchimedeanCopula, generic_data; method=:gnz2011, derived_measures=false) isa ArchimedeanCopula{2}
    @test fit(ExtremeValueCopula, generic_data; method=:ols,
              derived_measures=false) isa ExtremeValueCopula{2}

    non_fittable = (
        LiouvilleCopula{2}(Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
        ExtremeValueCopula{2}(Copulas.DiscreteSpectralTail([0.7 0.3; 0.2 0.8])),
    )
    for C in non_fittable
        U = rand(StableRNG(20_101), C, 4)
        @test_throws Exception fit(typeof(C), U; derived_measures=false)
    end
end

@testset "complete StatsBase model-result interface" begin
    C = ClaytonCopula{2}(1.5)
    U = [0.2 0.4 0.7 0.8; 0.3 0.6 0.5 0.9]
    M = CopulaModel(C, 4, loglikelihood(C, U), :fixture;
        method_details=(free_parameters=(θ=1.5,), U=U, null_ll=0.0))
    @test StatsBase.isfitted(M)
    @test StatsBase.nobs(M) == 4
    @test StatsBase.coef(M) == [1.5]
    @test StatsBase.coefnames(M) == ["θ"]
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
    @test_throws ArgumentError infer(M0)
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


@testset "dimension-specialized fitting reconstruction" begin
    C = ClaytonCopula{3}(2.0)
    CT = typeof(C)
    example = C

    f(x) = begin
        θ = (; θ=x)
        Cx = Copulas._fit_copula(
            CT,
            Val(3),
            θ,
            example,
        )
        return Distributions.params(Cx).θ
    end

    @test ForwardDiff.derivative(f, 2.0) ≈ 1.0
end

@testset "Gaussian copula multivariate MLE" begin
    d = 5
    n = 2_000
    ρ = 0.6
    R = [ρ^abs(i-j) for i in 1:d, j in 1:d]
    source = GaussianCopula(R)
    U = rand(StableRNG(477), source, n)
    model = fit(CopulaModel, GaussianCopula, U; method=:mle, derived_measures=false,)
    fitted = fitteddistribution(model)
    R̂ = params(fitted).Σ
    @test model.converged
    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(R̂))
    @test maximum(abs.(diag(R̂) .- 1)) < 1e-12
    @test maximum(abs.(R̂ - R)) < 0.05
end
