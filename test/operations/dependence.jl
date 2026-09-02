# Complete operation proof for scalar and pairwise dependence measures:
# family-wide contracts, independent defining identities, specialization
# equivalence, inverse behaviour, and exhaustive dispatch execution.

function dependence_operation_route_key(measure, C)
    Base.@nospecialize measure C
    method = which(measure, Tuple{typeof(C)})
    return (method, length(C) == 2 ? :bivariate : :multivariate)
end

@testset "dependence measures agree with their definitions" begin
    C = FGMCopula{2}(0.4)
    integral, _ = HCubature.hcubature(u -> cdf(C, u), zeros(2), ones(2);
                                      rtol=2e-5)
    @test Copulas.ρ(C) ≈ 12integral - 3 atol=2e-4
    @test Copulas.β(C) ≈ 4cdf(C, [0.5, 0.5]) - 1

    @test Copulas.τ(IndependentCopula{2}()) == 0
    @test Copulas.ρ(IndependentCopula{2}()) == 0
    @test Copulas.β(IndependentCopula{2}()) == 0
    @test Copulas.γ(IndependentCopula{2}()) == 0
    @test Copulas.τ(MCopula{2}()) == 1
    @test Copulas.ρ(MCopula{2}()) == 1
    @test Copulas.τ(WCopula{2}()) == -1
    @test Copulas.ρ(WCopula{2}()) == -1

    for C in (IndependentCopula{2}(), IndependentCopula{3}(),
              MCopula{2}(), MCopula{3}(), WCopula{2}())
        for measure in SCALAR_DEPENDENCE_MEASURES
            if applicable(measure, C) &&
               !(measure in (Copulas.ι,) && C isa WCopula)
                value = measure(C)
                @test value isa Real
                prove_dependence_route!(measure, C)
            end
        end
    end
end

function test_dependence_contract(C)
    Base.@nospecialize C
    # Expensive generic measures compose primitives proved by other operation
    # suites. Applicability is therefore checked for every family, while each
    # selected numerical implementation is executed only once below.
    for measure in SCALAR_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, C) || continue
        @test applicable(measure, C)
    end
    for (measure, _) in PAIRWISE_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, C) || continue
        @test applicable(measure, C)
    end
end

function test_scalar_dependence_result(measure, C)
    Base.@nospecialize measure C
    value = measure(C)
    @test value isa Real
    @test !isnan(value)
    if measure !== Copulas.ι
        @test -1 <= value <= 1
    end
end

function test_pairwise_dependence_result(measure, diagonal, C)
    Base.@nospecialize measure diagonal C
    d = length(C)
    matrix = measure(C)
    @test size(matrix) == (d, d)
    @test matrix ≈ transpose(matrix)
    @test diag(matrix) == fill(diagonal, d)
    @test all(x -> x isa Real && !isnan(x), matrix)
end

@testset verbose=true "public dependence-measure contract" begin
    @testset "$(fixture.case.name)" for fixture in COPULA_FIXTURES
        test_progress("operations", "dependence", fixture.case.name, "contract")
        test_dependence_contract(fixture.copula)
    end
end

@testset verbose=true "one execution per dependence-measure dispatch" begin
    # Prefer cheap closed-form representatives when several families select
    # the same route. Applicability remains checked for every family above.
    route_cost(case) = case.family in (BernsteinCopula, FGMCopula) ? 0 :
                       case.family === ClaytonCopula ? 1 : 2
    models = sort(collect(ROUTING_COPULA_FIXTURES); by=x -> route_cost(x.case))

    @testset verbose=true "$(nameof(measure))" for measure in SCALAR_DEPENDENCE_MEASURES
        selected_routes = Set(dependence_operation_route_key(measure, fixture.copula)
                              for fixture in models
                              if _dependence_is_defined(measure, fixture.copula))
        tested_routes = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, copula) || continue
            key = dependence_operation_route_key(measure, copula)
            key in tested_routes && continue
            test_progress("operations", "dependence", nameof(measure), case.name)
            test_scalar_dependence_result(measure, copula)
            push!(tested_routes, key)
        end
        @test tested_routes == selected_routes
    end

    @testset verbose=true "$(nameof(first(entry)))" for entry in PAIRWISE_DEPENDENCE_MEASURES
        measure, diagonal = entry
        selected_routes = Set(dependence_operation_route_key(measure, fixture.copula)
                              for fixture in models
                              if _dependence_is_defined(measure, fixture.copula))
        tested_routes = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, copula) || continue
            key = dependence_operation_route_key(measure, copula)
            key in tested_routes && continue
            test_progress("operations", "dependence", nameof(measure), case.name)
            test_pairwise_dependence_result(measure, diagonal, copula)
            push!(tested_routes, key)
        end
        @test tested_routes == selected_routes
    end
end



# Dependence-operation equivalence proofs for scalar and pairwise routes.

function _unique_dependence_routes(operation, predicate)
    Base.@nospecialize operation predicate
    seen = Set{Method}()
    routes = NamedTuple[]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 || continue
        predicate(case, C) || continue
        method = operation(case, C)
        method in seen && continue
        push!(seen, method)
        push!(routes, (; case, C, method))
    end
    return routes
end

@testset verbose=true "specialized dependence measures agree with generic definitions" begin
    # Entropy and Gini's gamma use substantially more expensive multidimensional
    # expectations and are covered by their independent identities in
    # correctness/. Kendall's generic definition is stochastic, so singular
    # Kendall formulas keep their exact family identities instead of a noisy,
    # repeated 10_000-observation comparison here. The CDF-only definitions of
    # rho, beta and tail dependence remain valid for singular and mixed laws.
    @testset verbose=true "$(nameof(SCALAR_DEPENDENCE_MEASURES[index]))" for index in (1, 2, 3, 6, 7)
        measure = SCALAR_DEPENDENCE_MEASURES[index]
        routes = _unique_dependence_routes(
            (_, C) -> which(measure, Tuple{typeof(C)}),
            (_, C) -> measure === Copulas.τ ?
                is_absolutely_continuous(C) : true,
        )
        generic_method = which(measure, Tuple{Copulas.Copula{2}})
        for (; case, C, method) in routes
            if method === generic_method
                # The bivariate generic mechanism is proved independently by
                # PolynomialOracleCopula in correctness/mathematical.jl.
                @test dependence_route_key(measure, C) in
                      PROVEN_DEPENDENCE_ROUTES[measure]
                continue
            end
            @testset "$(case.name)" begin
                test_progress("equivalence", nameof(measure), case.name)
                if measure === Copulas.τ &&
                   (C isa GaussianCopula || C isa TCopula)
                    # Kendall's tau is invariant over the radial distribution
                    # of an elliptical copula.  At ρ = 1/2, the exact identity
                    # 2asin(ρ)/π = 1/3 validates both elliptical
                    # specializations without repeatedly evaluating their
                    # expensive numerical CDFs inside a cubature.
                    reference = C isa GaussianCopula ?
                        GaussianCopula{2}(0.5) :
                        TCopula{2}(C.df, [1.0 0.5; 0.5 1.0])
                    @test Copulas.τ(reference) ≈ 1 / 3 atol=2e-15
                elseif measure === Copulas.ρ && C isa GaussianCopula
                    # The bivariate Gaussian identity avoids nesting the
                    # numerical normal CDF inside the generic rho cubature.
                    reference = GaussianCopula{2}(0.5)
                    @test Copulas.ρ(reference) ≈ 6asin(0.25) / π atol=2e-15
                else
                    expected = measure === Copulas.τ ?
                        4 * HCubature.hcubature(u -> cdf(C, u) * pdf(C, u),
                                               zeros(2), ones(2); rtol=1e-5)[1] - 1 :
                        invoke(measure, Tuple{Copulas.Copula}, C)
                    @test isapprox(measure(C), expected; atol=3e-4, rtol=3e-4)
                end
            end
            prove_dependence_route!(measure, C)
        end
    end
end

@testset "limit and subset dependence routes agree with independent identities" begin
    independence = IndependentCopula{2}()
    @test Copulas.γ(independence) == 0
    @test Copulas.ι(independence) == 0
    @test Copulas.γ(MCopula{2}()) == 1
    @test Copulas.ι(MCopula{2}()) == -Inf

    parent = ClaytonCopula{2}(1.5)
    subset = subsetdims(parent, (2, 1))
    # Gamma and entropy use stochastic generic expectations. Their forwarding
    # dispatches are inventoried below; exact value equality is meaningful only
    # for the deterministic measures.
    for measure in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λₗ, Copulas.λᵤ)
        @test measure(subset) == measure(parent)
    end
end

_spectral_matrix(tail::Copulas.DiscreteSpectralTail) = tail.B
_spectral_matrix(tail::Union{Copulas.BC2Tail,Copulas.MOTail}) = tail.spectral.B

function _spectral_curvature_tau(tail)
    # If A(t) = sum_k max(B[1,k]t, B[2,k](1-t)), its second derivative is
    # the discrete measure placing mass B[1,k] + B[2,k] at the corresponding
    # kink.  This is the distributional version of the defining EV Kendall
    # integral and, unlike a sample-concordance check, is exact and noiseless.
    B = _spectral_matrix(tail)
    total = zero(eltype(B))
    for k in axes(B, 2)
        mass = B[1, k] + B[2, k]
        iszero(mass) && continue
        kink = B[2, k] / mass
        total += mass * kink * (1 - kink) / Copulas.A(tail, kink)
    end
    return total
end

_singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.BC2Tail}) =
    _spectral_curvature_tau(C.tail)

function _singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.MOTail})
    # Classical competing-shocks identity. The public bivariate constructor
    # stores private shocks in subset order ([2], [1], [1,2]).
    λ₁, λ₂, λ₁₂ = C.tail.λ[2], C.tail.λ[1], C.tail.λ[3]
    a = λ₁ / (λ₁ + λ₁₂)
    b = λ₂ / (λ₂ + λ₁₂)
    return a * b / (a + b - a * b)
end

_singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.DiscreteSpectralTail}) =
    _spectral_curvature_tau(C.tail)

function _singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.CuadrasAugeTail})
    # Its Pickands function has one kink at 1/2 with slope jump 2θ.
    kink = 0.5
    return 2C.tail.θ * kink * (1 - kink) / Copulas.A(C.tail, kink)
end

# This is the classical bivariate Raftery identity, independently obtained
# from its common-factor mixture representation.
_singular_tau_oracle(C::RafteryCopula{2}) = 2C.θ / (3 - C.θ)
_singular_tau_oracle(::MCopula{2}) = 1
_singular_tau_oracle(::WCopula{2}) = -1
# The order-two Williamson transform of a Dirac radial is
# ϕ(t) = (1 - t/r)₊, hence its bivariate copula is the lower
# Fréchet--Hoeffding bound independently of the positive radius r.
_singular_tau_oracle(
    ::ArchimedeanCopula{2,<:WilliamsonGenerator{<:Dirac}},
) = -1

_singular_limit_kind(C::ArchimedeanCopula{2}) =
    Copulas.limit_kind(C.G, Val(2))
_singular_limit_kind(C::ExtremeValueCopula{2}) =
    Copulas.limit_kind(C.tail, Val(2))
_singular_limit_kind(C::ArchimaxCopula{2}) =
    Copulas._archimax_limit_kind(C)
_singular_limit_kind(::Copulas.Copula{2}) = Copulas.NO_LIMIT

function _singular_tau_oracle_with_limits(C::Copulas.Copula{2})
    kind = _singular_limit_kind(C)
    kind === Copulas.M_LIMIT && return 1
    kind === Copulas.W_LIMIT && return -1
    return _singular_tau_oracle(C)
end

@testset verbose=true "singular Kendall routes agree with deterministic identities" begin
    routes = _unique_dependence_routes(
        (_, C) -> which(Copulas.τ, Tuple{typeof(C)}),
        (_, C) -> !is_absolutely_continuous(C),
    )
    generic_method = which(Copulas.τ, Tuple{Copulas.Copula{2}})
    compared = 0
    for route in routes
        (; case, C, method) = route
        method === generic_method && continue
        @testset "$(case.name)" begin
            test_progress("equivalence", "singular Kendall", case.name)
            expected = _singular_tau_oracle_with_limits(C)
            @test Copulas.τ(C) ≈ expected atol=2e-12 rtol=2e-12
        end
        prove_dependence_route!(Copulas.τ, C)
        compared += 1
    end
    @test compared > 0
end

@testset "all gamma and entropy dispatches have an independent proof" begin
    parent = ClaytonCopula{2}(1.5)
    subset = subsetdims(parent, (2, 1))
    candidates = Any[]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 && push!(candidates, C)
    end
    push!(candidates, subset)

    for (measure, checked) in (
        (Copulas.γ, (PolynomialOracleCopula(0.4), IndependentCopula{2}(),
                     MCopula{2}(), subset)),
        (Copulas.ι, (PolynomialOracleCopula(0.4), IndependentCopula{2}(),
                     MCopula{2}(), subset)),
    )
        selected_methods = Set(which(measure, Tuple{typeof(C)}) for C in candidates)
        checked_methods = Set(which(measure, Tuple{typeof(C)}) for C in checked)
        @test selected_methods == checked_methods
        for C in checked
            prove_dependence_route!(measure, C)
        end
    end
end

@testset "every pairwise dependence route reduces to bivariate margins" begin
    scalar = Dict(
        StatsBase.corkendall => Copulas.τ,
        StatsBase.corspearman => Copulas.ρ,
        Copulas.corblomqvist => Copulas.β,
        Copulas.corgini => Copulas.γ,
        Copulas.corentropy => Copulas.ι,
        Copulas.corlowertail => Copulas.λₗ,
        Copulas.coruppertail => Copulas.λᵤ,
    )
    for (pairwise, diagonal) in PAIRWISE_DEPENDENCE_MEASURES
        selected = Set((which(pairwise, Tuple{typeof(fixture.copula)}),
                        length(fixture.copula) == 2 ? :bivariate : :multivariate)
                       for fixture in ROUTING_COPULA_FIXTURES
                       if _dependence_is_defined(pairwise, fixture.copula))
        checked = Set{Any}()
        for fixture in ROUTING_COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            _dependence_is_defined(pairwise, C) || continue
            key = (which(pairwise, Tuple{typeof(C)}),
                   length(C) == 2 ? :bivariate : :multivariate)
            key in checked && continue
            # Generic gamma and entropy estimators sample internally. Reusing
            # the same RNG state makes this an exact forwarding test instead
            # of comparing two independent Monte Carlo estimates.
            seed = 0x51a7 + hash((pairwise, key))
            Random.seed!(seed)
            observed = pairwise(C)
            Random.seed!(seed)
            if C isa EmpiricalCopula &&
               pairwise in (StatsBase.corkendall, StatsBase.corspearman)
                expected = pairwise(transpose(C.u))
            else
                d = length(C)
                expected = Matrix{Float64}(I, d, d) .* diagonal
                for i in 1:d, j in 1:(i - 1)
                    value = scalar[pairwise](subsetdims(C, (i, j)))
                    expected[i, j] = expected[j, i] = value
                end
            end
            @test observed ≈ expected atol=1e-8
            push!(checked, key)
        end
        @test checked == selected
    end
end

@testset "multivariate Archimedean and Raftery dependence identities" begin
    # These closed forms are dimension-dependent dispatch routes and therefore
    # cannot be represented by the bivariate specialization comparison above.
    clayton = ClaytonCopula{3}(1.5)
    @test Copulas.τ(clayton) ≈ 3 / 7
    # The generator specialization is dimension invariant; its bivariate
    # value is independently checked against the generic integral above.
    @test Copulas.ρ(clayton) == Copulas.ρ(ClaytonCopula{2}(1.5))
    prove_dependence_route!(Copulas.τ, clayton)
    prove_dependence_route!(Copulas.ρ, clayton)

    raftery = RafteryCopula{3}(0.5)
    @test Copulas.τ(raftery) ≈ 0.4
    @test Copulas.ρ(raftery) ≈ 13 / 27
    prove_dependence_route!(Copulas.τ, raftery)
    prove_dependence_route!(Copulas.ρ, raftery)
end



# Dependence-operation proof: verifies each public measure inverse on
# representative supported families and both type- and instance-based dispatch.
const _INVERSE_PAIRS = (
    (Copulas.τ, Copulas.τ⁻¹), (Copulas.ρ, Copulas.ρ⁻¹),
    (Copulas.β, Copulas.β⁻¹), (Copulas.λᵤ, Copulas.λᵤ⁻¹),
)
const _DEPENDENCE_INVERSES = last.(_INVERSE_PAIRS)
const _CHECKED_INVERSE_METHODS =
    Dict(inverse => Set{Method}() for inverse in _DEPENDENCE_INVERSES)

has_scalar_parameter(object) = length(params(object)) == 1
supports_inverse(object, inverse) = has_scalar_parameter(object) &&
    hasmethod(inverse, Tuple{Type{typeof(object)},Float64})
supports_inverse(C::ArchimedeanCopula, inverse) =
    has_scalar_parameter(C) &&
    hasmethod(inverse, Tuple{Type{typeof(C.G)},Float64})
function supports_inverse(C::ExtremeValueCopula, inverse)
    has_scalar_parameter(C) || return false
    # The generic EV Kendall inverse forwards to the tail.  Its signature is
    # therefore present for every EV copula even when that tail provides no
    # inverse (for example DiscreteSpectralTail).
    inverse === Copulas.τ⁻¹ && return hasmethod(
        inverse, Tuple{Type{typeof(C.tail)},Float64})
    return hasmethod(inverse, Tuple{Type{typeof(C)},Float64})
end

const _COPULA_INVERSE_CASES = Tuple(unique(typeof,
    [fixture.copula for fixture in ROUTING_COPULA_FIXTURES
     if length(fixture.copula) == 2 && has_scalar_parameter(fixture.copula)]))
const _GENERATOR_INVERSE_CASES = Tuple(unique(typeof,
    [G for G in GENERATOR_CASES if has_scalar_parameter(G)]))
const _TAIL_INVERSE_CASES = Tuple(unique(typeof,
    [tail for (tail, d) in TAIL_CASES
     if d == 2 && has_scalar_parameter(tail)]))

@testset "dependence-measure numerical anchors and boundary regimes" begin
    @test Copulas.Debye(0.5, 1) ≈ 0.8819271567906056
    @test Copulas.τ⁻¹(FrankCopula, 0.6) ≈ 7.929642284264058
    @test Copulas.τ⁻¹(GumbelCopula, 0.5) ≈ 2.0
    @test Copulas.τ⁻¹(ClaytonCopula, 1 / 3) ≈ 1.0
    @test Copulas.τ⁻¹(AMHCopula, 1 / 4) ≈ 0.8384520912688538
    @test Copulas.τ⁻¹(AMHCopula, 0.0) ≈ 0.0
    @test Copulas.τ⁻¹(AMHCopula, 1 / 3 + 0.0001) ≈ 1.0
    @test Copulas.τ⁻¹(AMHCopula, -2 / 11) ≈ -1.0
    @test Copulas.τ⁻¹(AMHCopula, -0.1505) ≈ -0.8 atol=1e-3
    @test Copulas.τ⁻¹(FrankCopula, -0.3881) ≈ -4.0 atol=1e-3
    @test Copulas.τ⁻¹(ClaytonCopula, -1 / 3) ≈ -0.5 atol=1e-5

    @test Copulas.ρ⁻¹(ClaytonCopula, 1 / 3) ≈ 0.58754 atol=1e-5
    @test Copulas.ρ⁻¹(ClaytonCopula, 0.01) ≈ 0.0 atol=1e-1
    @test Copulas.ρ⁻¹(ClaytonCopula, -0.4668) ≈ -0.5 atol=1e-3
    @test Copulas.ρ⁻¹(ClaytonCopula, 1.0) == Inf

    @test Copulas.ρ⁻¹(GumbelCopula, 0.5) ≈ 1.5410704204332681
    ρweak = 1e-4
    θweak = Copulas.ρ⁻¹(GumbelCopula, ρweak)
    @test 1 < θweak < 1.01
    @test Copulas.ρ(GumbelCopula{2}(θweak)) ≈ ρweak atol=1e-7

    @test Copulas.ρ⁻¹(FrankCopula, 1 / 3) ≈ 2.116497 atol=1e-5
    @test Copulas.ρ⁻¹(FrankCopula, -0.5572) ≈ -4.0 atol=1e-3

    @test Copulas.ρ⁻¹(AMHCopula, 0.2) ≈ 0.5168580913147318
    @test Copulas.ρ⁻¹(AMHCopula, 0.0) ≈ 0.0 atol=1e-4
    @test Copulas.ρ⁻¹(AMHCopula, 0.49) ≈ 1 atol=1e-4
    @test Copulas.ρ⁻¹(AMHCopula, -0.273) ≈ -1 atol=1e-4
    @test Copulas.ρ⁻¹(AMHCopula, -0.2246) ≈ -0.8 atol=1e-3
end

function _record_inverse_route!(inverse, argument_type)
    Base.@nospecialize inverse argument_type
    push!(_CHECKED_INVERSE_METHODS[inverse],
          which(inverse, Tuple{Type{argument_type},Float64}))
end

rebuild_inverse_case(C::ExtremeValueCopula{2}, parameter) =
    ExtremeValueCopula{2}(typeof(C.tail)(parameter))
rebuild_inverse_case(C::FGMCopula{d}, parameter) where {d} =
    FGMCopula{d}(parameter)
rebuild_inverse_case(C::Copulas.Copula, parameter) = typeof(C)(parameter)
rebuild_inverse_case(G::Copulas.Generator, parameter) = typeof(G)(parameter)

@testset "dispatched copula dependence-measure inverses" begin
    for C in _COPULA_INVERSE_CASES
        for (measure, inverse) in _INVERSE_PAIRS
            supports_inverse(C, inverse) || continue
            value = measure(C)
            rebuilt = rebuild_inverse_case(C, inverse(typeof(C), value))
            _record_inverse_route!(inverse, typeof(C))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end

@testset "dispatched generator dependence-measure inverses" begin
    for G in _GENERATOR_INVERSE_CASES
        for (measure, inverse) in _INVERSE_PAIRS[1:2]
            supports_inverse(G, inverse) || continue
            value = measure(G)
            rebuilt = rebuild_inverse_case(G, inverse(typeof(G), value))
            _record_inverse_route!(inverse, typeof(G))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end

@testset "dispatched tail Kendall inverses" begin
    for tail in _TAIL_INVERSE_CASES
        supports_inverse(tail, Copulas.τ⁻¹) || continue
        C = ExtremeValueCopula{2}(tail)
        value = Copulas.τ(C)
        rebuilt = ExtremeValueCopula{2}(
            typeof(tail)(Copulas.τ⁻¹(typeof(tail), value)))
        _record_inverse_route!(Copulas.τ⁻¹, typeof(tail))
        @test Copulas.τ(rebuilt) ≈ value atol=2e-6
    end
end

@testset "every dispatched dependence inverse method has an oracle" begin
    reachable = Dict(inverse => Set{Method}() for inverse in _DEPENDENCE_INVERSES)
    for C in _COPULA_INVERSE_CASES, (_, inverse) in _INVERSE_PAIRS
        supports_inverse(C, inverse) || continue
        push!(reachable[inverse], which(
            inverse, Tuple{Type{typeof(C)},Float64}))
    end
    for G in _GENERATOR_INVERSE_CASES, (_, inverse) in _INVERSE_PAIRS[1:2]
        supports_inverse(G, inverse) || continue
        push!(reachable[inverse], which(
            inverse, Tuple{Type{typeof(G)},Float64}))
    end
    for tail in _TAIL_INVERSE_CASES
        supports_inverse(tail, Copulas.τ⁻¹) || continue
        push!(reachable[Copulas.τ⁻¹], which(
            Copulas.τ⁻¹, Tuple{Type{typeof(tail)},Float64}))
    end
    @test _CHECKED_INVERSE_METHODS == reachable
end

@testset "BB6 Kendall tau identities and limits" begin
    # Interior BB6 value: exact reduction to Joe.
    θ, δ = 2.3, 1.7
    expected = 1 - (1 - Copulas.τ(JoeCopula{2}(θ))) / δ
    @test Copulas.τ(BB6Copula{2}(θ, δ)) ≈ expected

    # θ = 1 gives Gumbel: τ = 1 - 1/δ.
    for δ in (1.5, 2.0, 5.0)
        @test Copulas.τ(BB6Copula{2}(1.0, δ)) ≈ 1 - 1 / δ
    end

    # Infinite-parameter limits are comonotonic.
    @test Copulas.τ(BB6Copula{2}(1.0, Inf)) == 1
    @test Copulas.τ(BB6Copula{2}(2.3, Inf)) == 1
    @test Copulas.τ(BB6Copula{2}(Inf, 1.5)) == 1
    @test Copulas.τ(BB6Copula{2}(Inf, Inf)) == 1
end

@testset "BB6 Kendall tau Joe boundary" begin
    for θ in (1.2, 2.3, 5.0)
        @test Copulas.τ(BB6Copula{2}(θ, 1.0)) ≈
              Copulas.τ(JoeCopula{2}(θ))
    end
end

@testset "AsymLog Kendall tau boundaries" begin
    # Independence boundaries.
    @test Copulas.τ(AsymLogCopula{2}(1.0, 0.4, 0.7)) == 0
    @test Copulas.τ(AsymLogCopula{2}(2.3, 0.0, 0.7)) == 0
    @test Copulas.τ(AsymLogCopula{2}(2.3, 0.4, 0.0)) == 0

    # Symmetric logistic boundary.
    for α in (1.5, 2.0, 5.0)
        @test Copulas.τ(AsymLogCopula{2}(α, 1.0, 1.0)) ==
              1 - 1 / α
    end

    # Infinite-α / Marshall–Olkin limit.
    for (θ₁, θ₂) in ((0.4, 0.7), (0.2, 1.0), (1.0, 0.6))
        expected = θ₁ * θ₂ / (θ₁ + θ₂ - θ₁ * θ₂)
        @test Copulas.τ(AsymLogCopula{2}(Inf, θ₁, θ₂)) ≈ expected
    end

    @test Copulas.τ(AsymLogCopula{2}(Inf, 1.0, 1.0)) == 1
    @test Copulas.τ(AsymLogCopula{2}(Inf, 0.0, 0.7)) == 0
    @test Copulas.τ(AsymLogCopula{2}(Inf, 0.4, 0.0)) == 0
end

@testset "AsymLog Kendall tau interior agrees with generic EV formula" begin
    C = AsymLogCopula{2}(2.3, 0.4, 0.7)

    expected = invoke(
        Copulas.τ,
        Tuple{Copulas.ExtremeValueCopula{2}},
        C,
    )

    @test Copulas.τ(C) ≈ expected
end