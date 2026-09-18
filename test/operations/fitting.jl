# Fitting-operation proof: exercises representative Sklar, empirical, covariance,
# optimizer, and model-result fitting routes beyond the universal fit contract.
struct PublicFitProtocolProbe <: Copulas.Copula{2} end
Copulas._available_fitting_methods(::Type{PublicFitProtocolProbe}, ::Int) = (:probe,)
function Copulas._fit(::Type{PublicFitProtocolProbe}, data, ::Val{:probe}; offset=0.0)
    estimate = Statistics.mean(data) + offset
    return ClaytonCopula{2}(estimate)
end

struct PublicMPLProtocolProbe <: Copulas.Copula{2} end
Copulas._available_fitting_methods(::Type{PublicMPLProtocolProbe}, ::Int) = (:mle,)
function Copulas._fit(::Type{PublicMPLProtocolProbe}, data, ::Val{:mle}; kwargs...)
    return IndependentCopula{2}()
end

@testset "internal fitting dispatch" begin
    U = [0.2 0.4 0.6; 0.3 0.5 0.7]
    fitted = fit(PublicFitProtocolProbe, U; method=:probe, offset=0.1)
    model = fit(CopulaModel, PublicFitProtocolProbe, U;
                method=:probe, offset=0.1)
    @test fitted isa ClaytonCopula{2}
    @test StatsBase.coef(model) == [Statistics.mean(U) + 0.1]
    @test StatsBase.coefnames(model) == ["θ"]
    @test StatsBase.dof(model) == 1
    @test !("offset" in StatsBase.coefnames(model))
    @test_throws ArgumentError infer(model)
    inference = infer(model; method=:bootstrap, nresamples=3,
                      rng=StableRNG(48_099))
    @test inference isa Copulas.CopulaInference
    @test size(StatsBase.vcov(inference)) == (StatsBase.dof(model),
                                              StatsBase.dof(model))

    raw = [2.0 8.0 1.0 5.0; 4.0 1.0 6.0 2.0]
    mpl = fit(CopulaModel, PublicMPLProtocolProbe, raw;
              method=:mpl, pseudo_values=false)
    @test fitted_distribution(mpl) isa IndependentCopula{2}
    @test :mpl ∉ Copulas._available_fitting_methods(PublicMPLProtocolProbe, 2)
end

@testset "public Sklar fitting path" begin
    source = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    data = rand(StableRNG(111), source, 30)
    fitted = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                 copula_method=:itau)
    @test fitted isa SklarDist
    @test fitted.C isa ClaytonCopula{2}

    model = fit(CopulaModel,
        SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
        copula_method=:itau)
    @test fitted_distribution(model) isa SklarDist
    @test StatsBase.nobs(model) == size(data, 2)
    @test any(startswith("margin_"), StatsBase.coefnames(model))
    displayed = sprint(show, model)
    copula_section = split(split(displayed, "[ Copula parameters ]")[2],
                           "[ Marginals ]")[1]
    @test occursin("copula_", copula_section)
    @test !occursin("margin_", copula_section)
    inference = infer(model; method=:bootstrap, nresamples=3,
                      rng=StableRNG(114))
    @test size(StatsBase.vcov(inference)) ==
          (StatsBase.dof(model), StatsBase.dof(model))
    @test size(StatsBase.vcov(inference; component=:copula), 1) == 1
    @test size(StatsBase.vcov(inference; component=:margins), 1) ==
          StatsBase.dof(model) - 1
    @test_throws ArgumentError infer(model; method=:hessian)

    ecdf_fit = fit(SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data;
                   sklar_method=:ecdf, copula_method=:itau)
    @test ecdf_fit isa SklarDist

    default_model = fit(CopulaModel,
        SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}, data)
    @test Copulas.fitting_method(default_model) === :mle
    @test default_model.recipe.kwargs.sklar_method === :ifm

    empirical_model = fit(CopulaModel,
        SklarDist{EmpiricalCopula,Tuple{Normal,Exponential}}, data)
    @test Copulas.fitting_method(empirical_model) === :deheuvels
end

@testset "MLE and MPL input semantics" begin
    X = [2.0 8.0 1.0 5.0 3.0 7.0;
         4.0 1.0 6.0 2.0 5.0 3.0]
    U = pseudos(X)

    default_fit = fit(CopulaModel, ClaytonCopula{2}, U)
    mle_fit = fit(CopulaModel, ClaytonCopula{2}, U; method=:mle,
        pseudo_values=true)
    mpl_fit = fit(CopulaModel, ClaytonCopula{2}, X; method=:mpl,
        pseudo_values=false)
    normalized_mpl = fit(CopulaModel, ClaytonCopula{2}, X; method=:mle,
        pseudo_values=false)
    normalized_mle = @test_logs (:warn, r"method=:mpl requires raw observations") fit(
        CopulaModel, ClaytonCopula{2}, U; method=:mpl, pseudo_values=true)

    @test Copulas.fitting_method(default_fit) === :mle
    @test Copulas.fitting_method(mle_fit) === :mle
    @test Copulas.fitting_method(mpl_fit) === :mpl
    @test Copulas.fitting_method(normalized_mpl) === :mpl
    @test Copulas.fitting_method(normalized_mle) === :mle
    @test params(default_fit.result) == params(mle_fit.result)
    @test params(mpl_fit.result) == params(mle_fit.result)
    @test params(normalized_mpl.result) == params(mpl_fit.result)
    @test params(normalized_mle.result) == params(mle_fit.result)
    @test mpl_fit.data === X
    @test Copulas._copula_data(mpl_fit) == U
    @test_throws ArgumentError Copulas._find_method(EmpiricalCopula, 2, :mpl)
    @test_throws ArgumentError Copulas._find_method(ClaytonCopula{2}, 2, :mpl)
end

@testset "estimation and inference are separate" begin
    U = rand(StableRNG(112), ClaytonCopula{2}(1.0), 8)
    fitted = fit(ClaytonCopula{2}, U; method=:itau)
    model = fit(CopulaModel, ClaytonCopula{2}, U; method=:itau)
    @test fitted isa ClaytonCopula{2}
    @test !(fitted isa CopulaModel)
    @test params(fitted) == params(fitted_distribution(model))
    inference = infer(model; method=:bootstrap, nresamples=3,
                      rng=StableRNG(113))
    @test inference isa Copulas.CopulaInference
    @test size(StatsBase.vcov(inference)) ==
          (StatsBase.dof(model), StatsBase.dof(model))
    @test !applicable(StatsBase.vcov, model)
    @test_throws ArgumentError fit(CopulaModel, ClaytonCopula{2}, U;
        method=:itau, vcov=true)
    @test_throws ArgumentError fit(CopulaModel, ClaytonCopula{2}, U;
        method=:itau, vcov_method=:bootstrap)
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
        fitted = fit(target, U; method=:itau)
        @test fitted isa family
        @test Copulas.basecopula(fitted) isa ClaytonCopula{2}
        @test Copulas.flipmask(fitted) == mask
    end
end

@testset "bivariate Student rank matching" begin
    source = TCopula{2}(4.0, [1.0 0.55; 0.55 1.0])
    U = rand(StableRNG(316), source, 2_000)
    fitted = fit(TCopula{2}, U; method=:itau_irho)
    @test fitted isa TCopula{2}
    @test fitted.Σ[1, 2] ≈ sinpi(StatsBase.corkendall(U')[1, 2] / 2)
    @test Copulas.ρ(fitted) ≈ StatsBase.corspearman(U')[1, 2] atol=2e-3
end

@testset "Gaussian rank inversions are closed form" begin
    source = GaussianCopula(2, 0.6)
    U = rand(StableRNG(317), source, 2_000)
    τ̂ = StatsBase.corkendall(U')[1, 2]
    ρ̂ = StatsBase.corspearman(U')[1, 2]
    itau = fit(GaussianCopula, U; method=:itau)
    irho = fit(GaussianCopula, U; method=:irho)
    @test itau isa GaussianCopula{2}
    @test irho isa GaussianCopula{2}
    @test itau.Σ[1, 2] == sinpi(τ̂ / 2)
    @test irho.Σ[1, 2] == 2 * sinpi(ρ̂ / 6)
    @test Copulas.τ(itau) ≈ τ̂ atol=1e-14
    @test Copulas.ρ(irho) ≈ ρ̂ atol=1e-14

    # Above dimension 2 the inversion is pairwise and the matrix must stay
    # a valid correlation matrix.
    R = [1.0 0.5 0.3 0.1; 0.5 1.0 0.4 0.2; 0.3 0.4 1.0 0.6; 0.1 0.2 0.6 1.0]
    U4 = rand(StableRNG(318), GaussianCopula(R), 5_000)
    for method in (:itau, :irho)
        fitted = fit(GaussianCopula, U4; method)
        @test fitted isa GaussianCopula{4}
        @test maximum(abs.(fitted.Σ .- R)) < 0.03
        @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(fitted.Σ))
    end
    τ̂4 = StatsBase.corkendall(U4')
    @test fit(GaussianCopula, U4; method=:itau).Σ == sinpi.(τ̂4 ./ 2)

    # Pairwise Kendall coefficients need not be jointly consistent: the
    # inversion of (0.9, 0.9, -0.9) is not positive definite and is repaired.
    repaired = Copulas._nearest_correlation([1.0 0.9 0.9; 0.9 1.0 -0.9; 0.9 -0.9 1.0])
    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(repaired))
    @test maximum(abs.(LinearAlgebra.diag(repaired) .- 1)) == 0
    @test repaired[1, 2] == repaired[1, 3] == -repaired[2, 3]
    @test 0 < repaired[1, 2] < 0.9
    R0 = [1.0 0.5 0.3; 0.5 1.0 0.4; 0.3 0.4 1.0]
    @test Copulas._nearest_correlation(R0) == R0
end

@testset "Student Kendall inversion profiles degrees of freedom" begin
    source = TCopula{2}(4.0, [1.0 0.6; 0.6 1.0])
    U = rand(StableRNG(319), source, 5_000)
    fitted = fit(TCopula, U; method=:itau)
    @test fitted isa TCopula{2}
    @test fitted.Σ[1, 2] == sinpi(StatsBase.corkendall(U')[1, 2] / 2)
    @test 3 <= fitted.df <= 6
    # With the correlation held at the Kendall inversion the profile is a
    # restriction of the MLE profile, so its likelihood cannot exceed the MLE.
    mle = fit(TCopula, U; method=:mle)
    @test loglikelihood(fitted, U) <= loglikelihood(mle, U) + 1e-8
    @test loglikelihood(fitted, U) >= loglikelihood(GaussianCopula(fitted.Σ), U) - 1e-8
    @test isapprox(fitted.df, mle.df; rtol=0.1)

    d = 3
    R = [0.55^abs(i - j) for i in 1:d, j in 1:d]
    U3 = rand(StableRNG(320), TCopula(4.0, copy(R)), 2_000)
    model = fit(CopulaModel, TCopula, U3; method=:itau)
    fitted3 = fitted_distribution(model)
    @test fitted3 isa TCopula{3}
    @test Copulas.fitting_method(model) === :itau
    @test fitted3.Σ == sinpi.(StatsBase.corkendall(U3') ./ 2)
    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(fitted3.Σ))
    @test 3 <= fitted3.df <= 6
    @test :itau in Copulas._available_fitting_methods(TCopula, 3)
    @test :itau_irho ∉ Copulas._available_fitting_methods(TCopula, 3)
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
    fitted = fit(GaussianCopula, U; method=:mle,)
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
    model = fit(CopulaModel, TCopula, U; method=:mle,)
    fitted = fitted_distribution(model)
    θ = params(fitted)
    @test fitted isa TCopula{3}
    @test θ.ν > 0
    @test isfinite(θ.ν)
    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(θ.Σ),)
    @test maximum(abs.(LinearAlgebra.diag(θ.Σ) .- 1),) < 1e-12
    # The Student profile contains the Gaussian copula as ν = Inf,
    # so its fitted likelihood must not be worse than that endpoint.
    gaussian = fit(GaussianCopula,U; method=:mle,)
    @test loglikelihood(fitted, U) >= loglikelihood(gaussian, U) - 1e-8
end

@testset "Student MLE can estimate ν below two" begin
    d = 3
    ρ = 0.5
    Σ = [ρ^abs(i - j) for i in 1:d, j in 1:d]
    source = TCopula(1.0, copy(Σ))
    U = rand(StableRNG(478), source, 1_500)
    model = fit(CopulaModel, TCopula, U; method=:mle,)
    fitted = fitted_distribution(model)
    @test 0 < params(fitted).ν < 2
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
                         kwargs...)
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
            if C isa Union{
                    EmpiricalCopula,BetaCopula,BernsteinCopula,CheckerboardCopula,
                }
                U = pseudos(U)
            end
            route_kwargs = C isa EmpiricalEVCopula ? (d == 2 ? (; grid=21) : (; degree=1)) :
                C isa SurvivalCopula ? (; flips=C.flipmask) : ()
            fitted = fit( CT, U, method; route_kwargs...)
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
    @test fit(ClaytonCopula{2}, U, :itau) isa ClaytonCopula{2}
    @test fit(CopulaModel, ClaytonCopula{2}, U, :itau) isa CopulaModel

    D = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    X = rand(StableRNG(20_051), D, 12)
    family = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
    @test fit(family, X, :itau) isa SklarDist
    @test fit(CopulaModel, family, X, :itau) isa CopulaModel
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
        fitted = fit(family, U; method=method, kwargs...)
        expected = direct()
        @test typeof(fitted) == typeof(expected)
        @test params(fitted) == params(expected)
        @test cdf(fitted, point) ≈ cdf(expected, point)
    end

    U3 = _FIXTURE_DATA3
    fitted3 = fit(EmpiricalEVCopula, U3; method=:cfg, degree=1)
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
    @test fit(nested, nested_data) isa
          NestedArchimedeanCopula{4}

    generic_data = rand(StableRNG(20_102), ClaytonCopula{2}(1.0), 64)
    @test fit(ArchimedeanCopula, generic_data; method=:gnz2011) isa ArchimedeanCopula{2}
    @test fit(ExtremeValueCopula, generic_data; method=:ols) isa ExtremeValueCopula{2}

    non_fittable = (
        LiouvilleCopula{2}(Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
        ExtremeValueCopula{2}(Copulas.DiscreteSpectralTail([0.7 0.3; 0.2 0.8])),
    )
    for C in non_fittable
        U = rand(StableRNG(20_101), C, 4)
        @test_throws Exception fit(typeof(C), U)
    end
end

@testset "complete StatsBase model-result interface" begin
    C = ClaytonCopula{2}(1.5)
    U = [0.2 0.4 0.7 0.8; 0.3 0.6 0.5 0.9]
    M = CopulaModel(C, U, loglikelihood(C, U),
        Copulas._CopulaFitSpec(ClaytonCopula{2}, :fixture, (;)))
    @test StatsBase.isfitted(M)
    @test StatsBase.nobs(M) == 4
    @test StatsBase.coef(M) == [1.5]
    @test StatsBase.coefnames(M) == ["θ"]
    @test StatsBase.nullloglikelihood(M) == 0
    @test StatsBase.nulldeviance(M) == 0
    @test size(StatsBase.residuals(M)) == size(U)
    @test size(StatsBase.residuals(M; transform=:normal)) == size(U)
    @test_throws ArgumentError StatsBase.residuals(M; transform=:invalid)
    M0 = CopulaModel(EmpiricalCopula(U), U, 0.0,
        Copulas._CopulaFitSpec(EmpiricalCopula, :empirical, (;)))
    @test StatsBase.dof(M0) == 0
    @test isempty(StatsBase.coef(M0))
    @test isempty(StatsBase.coefnames(M0))
    @test_throws ArgumentError infer(M0)
    @test StatsBase.aic(M0) == StatsBase.bic(M0) == 0

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
        Cx = Copulas._construct_fitted_copula(
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
    model = fit(CopulaModel, GaussianCopula, U; method=:mle,)
    fitted = fitted_distribution(model)
    R̂ = params(fitted).Σ
    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(R̂))
    @test maximum(abs.(diag(R̂) .- 1)) < 1e-12
    @test maximum(abs.(R̂ - R)) < 0.05
end

@testset "observation weights" begin
    # Fitting with `weights` maximizes a weighted pseudo-likelihood whose
    # weights are normalized to sum to n. Unit weights, at any scale, must
    # reproduce the unweighted estimator bit for bit in every family that
    # implements `:mle`, including the families with a hand-written objective.
    n = 200
    families = [
        (ClaytonCopula, ClaytonCopula{2}(2.0)),
        (GumbelCopula, GumbelCopula{3}(1.6)),
        (FrankCopula, FrankCopula{2}(3.0)),
        (AMHCopula, AMHCopula{2}(0.5)),
        (BB1Copula, BB1Copula{2}(1.2, 1.5)),
        (GaussianCopula, GaussianCopula([1.0 0.5; 0.5 1.0])),
        (GaussianCopula, GaussianCopula([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0])),
        (TCopula, TCopula(4.0, [1.0 0.5; 0.5 1.0])),
        (TCopula, TCopula(5.0, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0])),
        (FGMCopula, FGMCopula(2, 0.4)),
        (FGMCopula, FGMCopula(3, [0.2, 0.1, 0.1, 0.05])),
        (GalambosCopula, GalambosCopula(2, 1.5)),
        (SurvivalCopula{2,ClaytonCopula{2}}, SurvivalCopula(ClaytonCopula{2}(2.0))),
        (Rotated90Copula{2,ClaytonCopula{2}}, Rotated90Copula(ClaytonCopula{2}(2.0))),
        (IndependentCopula, IndependentCopula(2)),
    ]
    @testset "unit weights are the unweighted fit: $(CT)" for (CT, C) in families
        U = rand(StableRNG(516), C, n)
        unweighted = fit(CopulaModel, CT, U; method=:mle)
        for weights in (ones(n), fill(3, n), fill(2.5, n))
            weighted = fit(CopulaModel, CT, U; method=:mle, weights)
            @test coef(weighted) == coef(unweighted)
            @test loglikelihood(weighted) == loglikelihood(unweighted)
            @test nobs(weighted) == n
            @test aic(weighted) == aic(unweighted)
            @test bic(weighted) == bic(unweighted)
        end
        @test Copulas._model_weights(unweighted) === nothing
    end
    let U = rand(StableRNG(516), ClaytonCopula{2}(2.0), n)
        @test occursin("Observation weights", sprint(show, fit(CopulaModel, ClaytonCopula, U; weights=ones(n))))
        @test !occursin("Observation weights", sprint(show, fit(CopulaModel, ClaytonCopula, U)))
    end

    @testset "integer weights are replication counts" begin
        # Weights that are counts summing to n are left untouched by the
        # normalization, so the weighted fit is the fit of the sample in which
        # column j is repeated weights[j] times. The two objectives agree up
        # to summation order, so the optimizers stop within their tolerance
        # of each other (Brent's default relative tolerance is sqrt(eps)).
        counts = zeros(Int, n)
        for slot in rand(StableRNG(517), 1:n, n)
            counts[slot] += 1
        end
        @test sum(counts) == n && any(iszero, counts)
        kept = findall(>(0), counts)
        for (CT, C) in ((ClaytonCopula, ClaytonCopula{2}(2.0)),
                        (GaussianCopula, GaussianCopula([1.0 0.5; 0.5 1.0])),
                        (TCopula, TCopula(4.0, [1.0 0.5; 0.5 1.0])),
                        (FGMCopula, FGMCopula(2, 0.4)),
                        (GalambosCopula, GalambosCopula(2, 1.5)))
            U = rand(StableRNG(518), C, n)
            Urep = hcat((repeat(U[:, j], 1, counts[j]) for j in kept)...)
            replicated = fit(CopulaModel, CT, Urep; method=:mle)
            weighted = fit(CopulaModel, CT, U; method=:mle, weights=counts)
            @test coef(weighted) ≈ coef(replicated) rtol=1e-6
            @test loglikelihood(weighted) ≈ loglikelihood(replicated) rtol=1e-8
            @test nobs(weighted) == nobs(replicated) == n
            @test bic(weighted) ≈ bic(replicated) rtol=1e-8
            # The estimator is invariant to the scale of the weights.
            rescaled = fit(CopulaModel, CT, U; method=:mle, weights=0.25 .* counts)
            @test coef(rescaled) ≈ coef(weighted) rtol=1e-8
            @test loglikelihood(rescaled) ≈ loglikelihood(weighted) rtol=1e-8
        end
    end

    @testset "a zero weight removes its observation" begin
        U = rand(StableRNG(519), ClaytonCopula{2}(2.0), n)
        weights = ones(n)
        weights[1] = 0
        # Even where the removed observation has a vanishing density.
        U[:, 1] .= 1e-300
        removed = fit(ClaytonCopula, U[:, 2:end])
        @test params(fit(ClaytonCopula, U; weights)).θ ≈ params(removed).θ rtol=1e-6
        @test isfinite(loglikelihood(fit(CopulaModel, ClaytonCopula, U; weights)))
        C = ClaytonCopula{2}(2.0)
        @test Copulas._weighted_loglikelihood(C, U, weights) ==
              Copulas._weighted_loglikelihood(C, U[:, 2:end], ones(n - 1))
        @test Copulas._weighted_loglikelihood(C, U[:, 2:end], ones(n - 1)) ==
              loglikelihood(C, U[:, 2:end])
        # The removed observation may sit on the boundary of the hypercube,
        # where the elliptical scores are infinite: it is dropped before the
        # engine forms its weighted cross-product, so no `0 * Inf` reaches it.
        for (CT, C) in ((GaussianCopula, GaussianCopula([1.0 0.5; 0.5 1.0])),
                        (TCopula, TCopula(4.0, [1.0 0.5; 0.5 1.0])),
                        (ClaytonCopula, ClaytonCopula{2}(2.0)))
            V = rand(StableRNG(528), C, n)
            for boundary in (0.0, 1.0)
                V[:, 1] .= boundary
                removed = fit(CopulaModel, CT, V[:, 2:end])
                kept = fit(CopulaModel, CT, V; weights)
                @test coef(kept) ≈ coef(removed) rtol=1e-6
                # The kept weights are normalized to sum to n over n - 1 columns.
                @test loglikelihood(kept) ≈ loglikelihood(removed) * n / (n - 1) rtol=1e-6
                @test nobs(kept) == n
                # The model records every column and its normalized weight.
                @test Copulas._model_weights(kept) == weights .* (n / (n - 1))
            end
        end
        # The rank inversions are given the same filtered pair: the weighted
        # measure of the n - 1 kept columns at weight n / (n - 1) each is the
        # unweighted measure of those columns, up to the scale's rounding.
        V = rand(StableRNG(529), GaussianCopula([1.0 0.5; 0.5 1.0]), n)
        for boundary in (0.0, 1.0), method in (:itau, :irho, :ibeta)
            V[:, 1] .= boundary
            removed = fit(CopulaModel, GaussianCopula, V[:, 2:end]; method)
            kept = fit(CopulaModel, GaussianCopula, V; method, weights)
            @test coef(kept) ≈ coef(removed) rtol=1e-12
            @test loglikelihood(kept) ≈ loglikelihood(removed) * n / (n - 1) rtol=1e-6
        end
        @test Copulas._weighted_sample(U, weights) == (U[:, 2:end], ones(n - 1))
        let w = ones(n)
            @test Copulas._weighted_sample(U, w) === (U, w)
        end
        @test Copulas._weighted_sample(U, nothing) === (U, nothing)
    end

    @testset "weighted maximum pseudo-likelihood ranks by weight" begin
        X = randn(StableRNG(520), 2, n)
        weights = rand(StableRNG(521), n)
        model = fit(CopulaModel, ClaytonCopula, X; pseudo_values=false, weights)
        @test Copulas.fitting_method(model) === :mpl
        direct = fit(CopulaModel, ClaytonCopula, pseudos(X; weights); weights)
        @test coef(model) == coef(direct)
        @test loglikelihood(model) == loglikelihood(direct)
        @test Copulas._copula_data(model) == pseudos(X; weights)
        @test nullloglikelihood(model) == 0
        @test residuals(model) == residuals(direct)
        unweighted = fit(CopulaModel, ClaytonCopula, X; pseudo_values=false)
        @test coef(fit(CopulaModel, ClaytonCopula, X; pseudo_values=false, weights=ones(n))) ==
              coef(unweighted)
    end

    @testset "selection and nested Archimedean fits take weights" begin
        U = rand(StableRNG(522), ClaytonCopula{2}(2.0), n)
        weights = rand(StableRNG(523), n)
        selection = fit(CopulaModel, Copulas.Copula, U;
                        candidates=[ClaytonCopula, GumbelCopula], weights)
        @test Copulas._model_weights(selected_model(selection)) !== nothing
        @test all(row.status === :ok for row in selection_table(selection))

        C0 = NestedArchimedeanCopula(Copulas.ClaytonGenerator(1.0);
                                     children=[ClaytonCopula{2}(2.0), ClaytonCopula{2}(3.0)])
        Un = rand(StableRNG(524), C0, n)
        unweighted = fit(CopulaModel, C0, Un)
        weighted = fit(CopulaModel, C0, Un; weights=fill(2, n))
        @test coef(weighted) == coef(unweighted)
        @test loglikelihood(weighted) == loglikelihood(unweighted)
        @test params(fit(C0, Un; weights=ones(n))) == params(fit(C0, Un))
        reparam(α) = NestedArchimedeanCopula(Copulas.ClaytonGenerator(exp(α[1]));
            children=[ClaytonCopula{2}(exp(α[1]) + exp(α[2])),
                      ClaytonCopula{2}(exp(α[1]) + exp(α[3]))])
        custom = fit(CopulaModel, reparam, zeros(3), Un)
        custom_weighted = fit(CopulaModel, reparam, zeros(3), Un; weights=ones(n))
        @test coef(custom_weighted) == coef(custom)
        @test_throws ArgumentError fit(CopulaModel, C0, Un; weights=-ones(n))
        # A resample of the weighted nested fit is refitted unweighted; the
        # reparameterized model keeps every child θ above its parent's by
        # construction.
        @test infer(custom_weighted; method=:bootstrap, nresamples=2, rng=StableRNG(528)).method === :bootstrap
    end

    @testset "weighted rank measures" begin
        # Each weighted measure is the measure of the sample in which
        # observation j is repeated w[j] times. Kendall's tau-b reproduces the
        # integer arithmetic of StatsBase.corkendall, so unit weights are bit
        # identical; Blomqvist's beta counts the same way; Spearman's rho is
        # the weighted Pearson correlation of the weighted average ranks, a
        # different reduction from StatsBase's, within an ulp of it.
        U = rand(StableRNG(529), GaussianCopula([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]), n)
        Ut = round.(U; digits=1)
        counts = zeros(Int, n)
        for slot in rand(StableRNG(530), 1:n, n)
            counts[slot] += 1
        end
        kept = findall(>(0), counts)
        replicate(X) = hcat((repeat(X[:, j], 1, counts[j]) for j in kept)...)
        measures = ((Copulas._weighted_corkendall, corkendall),
                    (Copulas._weighted_corspearman, corspearman),
                    (Copulas._weighted_corblomqvist, Copulas.corblomqvist))
        for (weighted, reference) in measures, X in (U, Ut)
            # The kernels take the weights `_fit_weights` has normalized, so
            # unit weights reach them as exactly one.
            if weighted === Copulas._weighted_corspearman
                @test weighted(X, ones(n)) ≈ reference(X') atol=1e-15
            else
                @test weighted(X, ones(n)) == reference(X')
            end
            @test weighted(X, fill(3.0, n)) ≈ reference(X') atol=1e-15
            @test weighted(X, Float64.(counts)) ≈ reference(replicate(X)') atol=1e-15
            @test weighted(X, 0.5 .* counts) ≈ weighted(X, Float64.(counts)) atol=1e-15
            @test weighted(X, Float64.(counts))[1, 2] == weighted(X, Float64.(counts))[2, 1]
            @test all(isone, LinearAlgebra.diag(weighted(X, ones(n))))
        end
        @test Copulas._weighted_β(U, ones(n)) == Copulas.β(U)
        @test Copulas._weighted_β(U, Float64.(counts)) ≈ Copulas.β(replicate(U)) atol=1e-15
        # A zero weight removes its observation, and a NaN column poisons its pair.
        w0 = ones(n); w0[1] = 0
        for (weighted, reference) in measures
            @test weighted(U, w0)[1, 2] ≈ reference(U[:, 2:end]')[1, 2] atol=1e-15
        end
        Un = copy(U); Un[1, 3] = NaN
        for (weighted, _) in measures
            @test isnan(weighted(Un, ones(n))[1, 2]) && !isnan(weighted(Un, ones(n))[2, 3])
        end
        # The measure's element type is promoted from the sample and the
        # weights, never forced.
        for (weighted, _) in measures
            @test eltype(weighted(Float32.(U), ones(Float32, n))) === Float32
            @test eltype(weighted(Float32.(U), ones(n))) === Float64
            @test eltype(weighted(BigFloat.(U), ones(n))) === BigFloat
            @test Float64.(weighted(BigFloat.(U), ones(n))) ≈ weighted(U, ones(n)) atol=1e-15
        end
    end

    @testset "rank inversions take weights: $(nameof(CT)) d=$(length(C)) $method" for (CT, C, methods) in [
            (ClaytonCopula, ClaytonCopula{2}(2.0), (:itau, :irho, :ibeta)),
            (GumbelCopula, GumbelCopula{3}(1.6), (:itau, :irho, :ibeta)),
            (GaussianCopula, GaussianCopula([1.0 0.5; 0.5 1.0]), (:itau, :irho, :ibeta)),
            (GaussianCopula, GaussianCopula([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]), (:itau, :irho, :ibeta)),
            (TCopula, TCopula(4.0, [1.0 0.5; 0.5 1.0]), (:itau, :itau_irho)),
            (TCopula, TCopula(5.0, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]), (:itau,)),
            (FGMCopula, FGMCopula(2, 0.4), (:itau, :irho, :ibeta)),
            (GalambosCopula{2}, GalambosCopula(2, 1.5), (:itau, :irho, :ibeta)),
            (ArchimaxCopula{2,Copulas.IndependentGenerator,Copulas.GalambosTail},
             ArchimaxCopula{2}(Copulas.IndependentGenerator(), Copulas.GalambosTail(1.5)), (:itau, :irho, :ibeta)),
            (SurvivalCopula{2,ClaytonCopula{2}}, SurvivalCopula(ClaytonCopula{2}(2.0)), (:itau, :irho, :ibeta)),
            (IndependentCopula, IndependentCopula(2), (:itau, :irho, :ibeta)),
            ], method in methods
        # The weighted inversion of unit weights is the unweighted one, and
        # integer weights summing to n invert the measure of the replicated
        # sample: exactly for a closed-form or root-finding inversion, to
        # Brent's tolerance for the Student profile over the degrees of freedom.
        U = rand(StableRNG(531), C, n)
        unweighted = fit(CopulaModel, CT, U; method)
        for weights in (ones(n), fill(3, n), fill(2.5, n))
            weighted = fit(CopulaModel, CT, U; method, weights)
            @test coef(weighted) == coef(unweighted)
            @test loglikelihood(weighted) == loglikelihood(unweighted)
            @test Copulas.fitting_method(weighted) === method
        end
        counts = zeros(Int, n)
        for slot in rand(StableRNG(532), 1:n, n)
            counts[slot] += 1
        end
        kept = findall(>(0), counts)
        Urep = hcat((repeat(U[:, j], 1, counts[j]) for j in kept)...)
        replicated = fit(CopulaModel, CT, Urep; method)
        weighted = fit(CopulaModel, CT, U; method, weights=counts)
        tol = C isa TCopula && method === :itau ? 1e-6 : 1e-12
        @test coef(weighted) ≈ coef(replicated) rtol=tol
        @test loglikelihood(weighted) ≈ loglikelihood(replicated) rtol=1e-8
        @test Copulas._model_weights(weighted) !== nothing
    end

    @testset "the Sklar route takes weights" begin
        # Every step reads the same weights: the margins through
        # Distributions.fit(D, x, w), the :ecdf ranks through the weighted
        # pseudos, the copula through fit(CT, U; weights), the log-likelihood
        # through the weighted sum. Distributions.jl reduces its weighted
        # sufficient statistics in another order than its unweighted ones, so
        # unit weights reproduce the margins up to rounding, not bit for bit.
        S0 = SklarDist(ClaytonCopula{2}(2.0), (Normal(1.0, 2.0), Exponential(3.0)))
        X = rand(StableRNG(533), S0, n)
        T = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
        counts = zeros(Int, n)
        for slot in rand(StableRNG(534), 1:n, n)
            counts[slot] += 1
        end
        kept = findall(>(0), counts)
        Xrep = hcat((repeat(X[:, j], 1, counts[j]) for j in kept)...)
        for sklar_method in (:ifm, :ecdf)
            unweighted = fit(CopulaModel, T, X; sklar_method)
            for weights in (ones(n), fill(3, n))
                weighted = fit(CopulaModel, T, X; sklar_method, weights)
                # Under :ifm the margins' rounding is the copula step's input,
                # so its parameter lands within optimizer tolerance.
                @test coef(weighted) ≈ coef(unweighted) rtol=1e-6
                Sw, Su = fitted_distribution(weighted), fitted_distribution(unweighted)
                for i in 1:2
                    @test collect(params(Sw.m[i])) ≈ collect(params(Su.m[i])) rtol=1e-10
                end
                @test loglikelihood(weighted) ≈ loglikelihood(unweighted) rtol=1e-10
                @test nobs(weighted) == n
            end
            replicated = fit(CopulaModel, T, Xrep; sklar_method)
            weighted = fit(CopulaModel, T, X; sklar_method, weights=counts)
            @test coef(weighted) ≈ coef(replicated) rtol=1e-6
            @test loglikelihood(weighted) ≈ loglikelihood(replicated) rtol=1e-8
            @test bic(weighted) ≈ bic(replicated) rtol=1e-8
            @test nobs(weighted) == nobs(replicated) == n
            @test Copulas._model_weights(weighted) == Copulas._fit_weights(counts, n)
            @test size(Copulas._copula_data(weighted)) == size(X)
            sklar_method === :ecdf &&
                @test Copulas._copula_data(weighted) == pseudos(X; weights=counts)
            @test params(fit(T, X; sklar_method, weights=counts).C) == params(fitted_distribution(weighted).C)
            # The copula step may be a rank inversion.
            rank = fit(T, X; sklar_method, copula_method=:itau, weights=counts)
            @test params(rank.C) == params(fit(ClaytonCopula, pseudos(X; weights=counts); method=:itau, weights=counts))
            @test occursin("Observation weights", sprint(show, weighted))
        end
        # A zero-weight observation is dropped before the margin sees it. It
        # may sit on the boundary of the margin's support, where a weighted
        # sufficient statistic is `0 * -Inf`: the Gamma statistic carries
        # `log(x)`, and `Distributions.fit(Gamma, x, w)` with `x[1] = 0`,
        # `w[1] = 0` raises a `DomainError` on the `NaN` shape.
        SG = SklarDist{ClaytonCopula,Tuple{Normal,Gamma}}
        Z = rand(StableRNG(536), SklarDist(ClaytonCopula{2}(2.0), (Normal(), Gamma(2.0, 3.0))), n)
        Z[2, 1] = 0.0
        zero_weight = ones(n)
        zero_weight[1] = 0
        @test_throws DomainError Distributions.fit(Gamma, Z[2, :], zero_weight)
        for sklar_method in (:ifm, :ecdf)
            removed = fit(CopulaModel, SG, Z[:, 2:end]; sklar_method)
            kept = fit(CopulaModel, SG, Z; sklar_method, weights=zero_weight)
            # The :ecdf ranks of the kept columns divide by n + 1 with n
            # counting the removed column, so they differ from the ranks of
            # the n - 1 columns alone by a factor n² / (n² - 1).
            tol = sklar_method === :ifm ? 1e-6 : 1e-3
            @test coef(kept) ≈ coef(removed) rtol=tol
            @test loglikelihood(kept) ≈ loglikelihood(removed) * n / (n - 1) rtol=tol
            @test nobs(kept) == n
            @test size(Copulas._copula_data(kept)) == size(Z)
        end
        # Refusals: a margin family without a weighted fit, by name; weights
        # through copula_kwargs.
        Y = copy(X); Y[2, :] .= rand(StableRNG(535), n)
        @test_throws ArgumentError fit(SklarDist{ClaytonCopula,Tuple{Normal,Beta}}, Y; weights=ones(n))
        @test_throws ArgumentError fit(SklarDist{ClaytonCopula,Tuple{Normal,Cauchy}}, X; weights=ones(n))
        err = try fit(SklarDist{ClaytonCopula,Tuple{Normal,Cauchy}}, X; weights=ones(n)) catch e; e end
        @test occursin("Cauchy", sprint(showerror, err))
        @test_throws ArgumentError fit(T, X; copula_kwargs=(; weights=ones(n)))
        @test_throws ArgumentError fit(T, X; weights=-ones(n))
    end

    @testset "the template fit stays in the certified nesting region" begin
        # `_nested_rebound` skips the certificate, so the loss is `Inf` on a
        # tree that fails one: a child Clayton θ below its parent's, or below
        # zero, where the composed generator is not defined. Without the
        # barrier the bootstrap refits below raise a `DomainError` from
        # `composition_taylor` on three of these four seeds.
        C0 = NestedArchimedeanCopula(Copulas.ClaytonGenerator(1.0);
                                     children=[ClaytonCopula{2}(2.0), ClaytonCopula{2}(3.0)])
        recon = Base.Fix1(Copulas._nested_rebound, C0)
        @test Copulas._nested_certified(recon(Copulas._nested_unbound(C0)))
        @test !Copulas._nested_certified(recon([log(2.0), log(1.5), log(4.0)]))
        @test !Copulas._nested_certified(recon([log(2.0), log(0.5), log(4.0)]))
        Un = rand(StableRNG(524), C0, n)
        M = fit(CopulaModel, C0, Un)
        for seed in (528, 529, 530, 531)
            I = infer(M; method=:bootstrap, nresamples=2, rng=StableRNG(seed))
            @test all(isfinite, vcov(I))
        end
    end

    @testset "refusals" begin
        U = rand(StableRNG(525), ClaytonCopula{2}(2.0), n)
        @test_throws DimensionMismatch fit(ClaytonCopula, U; weights=ones(n - 1))
        @test_throws ArgumentError fit(ClaytonCopula, U; weights=-ones(n))
        @test_throws ArgumentError fit(ClaytonCopula, U; weights=zeros(n))
        @test_throws ArgumentError fit(ClaytonCopula, U; weights=[NaN; ones(n - 1)])
        @test_throws ArgumentError fit(ClaytonCopula, U; weights=[Inf; ones(n - 1)])
        @test_throws ArgumentError fit(ClaytonCopula, U; weights=ones(n, 1))
        @test_throws ArgumentError fit(GalambosCopula, U; method=:iupper, weights=ones(n))
        @test_throws ArgumentError fit(BetaCopula, U; weights=ones(n))
        @test_throws ArgumentError fit(EmpiricalCopula, U; weights=ones(n))

        weighted = fit(CopulaModel, ClaytonCopula, U; weights=rand(StableRNG(527), n))
        @test_throws ArgumentError infer(weighted; method=:jackknife)
        @test_throws ArgumentError GOFCopulaTest(weighted; N=2)
        @test_throws ArgumentError GOFCopulaTest(weighted, U; N=2)
    end
end
