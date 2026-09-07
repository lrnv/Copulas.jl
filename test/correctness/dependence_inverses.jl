# Internal fitting-kernel regression: dependence-measure inverses are not public
# API, but their dispatched implementations must remain numerically coherent.

# Internal correctness proof: verifies each dependence-measure inverse on
# representative supported families and both type- and instance-based dispatch.
const _INVERSE_PAIRS = (
    (Copulas.τ, Copulas.τ⁻¹), (Copulas.ρ, Copulas.ρ⁻¹),
    (Copulas.β, Copulas.β⁻¹), (Copulas.λᵤ, Copulas.λᵤ⁻¹),
)
const _DEPENDENCE_INVERSES = last.(_INVERSE_PAIRS)
const _CHECKED_INVERSE_METHODS =
    Dict(inverse => Set{Method}() for inverse in _DEPENDENCE_INVERSES)

function has_scalar_parameter(object)
    Base.@nospecialize object
    return length(params(object)) == 1
end

function supports_inverse(object, inverse)
    Base.@nospecialize object inverse

    return has_scalar_parameter(object) &&
           hasmethod(
               inverse,
               Tuple{Type{typeof(object)},Float64},
           )
end

function supports_inverse(C::ArchimedeanCopula, inverse)
    Base.@nospecialize C inverse

    return has_scalar_parameter(C) &&
           hasmethod(
               inverse,
               Tuple{Type{typeof(C.G)},Float64},
           )
end

function supports_inverse(C::ExtremeValueCopula, inverse)
    Base.@nospecialize C inverse

    has_scalar_parameter(C) || return false

    # The generic EV Kendall inverse forwards to the tail. Its signature is
    # therefore present for every EV copula even when that tail provides no
    # inverse (for example DiscreteSpectralTail).
    inverse === Copulas.τ⁻¹ && return hasmethod(
        inverse,
        Tuple{Type{typeof(C.tail)},Float64},
    )

    return hasmethod(
        inverse,
        Tuple{Type{typeof(C)},Float64},
    )
end

const _COPULA_INVERSE_CASES = unique(typeof,
    [fixture.copula for fixture in COPULA_FIXTURES
     if length(fixture.copula) == 2 && has_scalar_parameter(fixture.copula)])
const _GENERATOR_INVERSE_CASES = unique(typeof,
    [G for G in GENERATOR_CASES if has_scalar_parameter(G)])
const _TAIL_INVERSE_CASES = unique(typeof,
    [tail for (tail, d) in TAIL_CASES
     if d == 2 && has_scalar_parameter(tail)])

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

@testset "internal Fréchet generator reductions" begin
    @test Copulas.τ(Copulas.MGenerator()) == 1
    @test Copulas.τ(Copulas.WGenerator()) == -1
    @test ArchimedeanCopula{3}(Copulas.MGenerator()) isa MCopula{3}
    @test ArchimedeanCopula{2}(Copulas.WGenerator()) isa WCopula{2}
end
