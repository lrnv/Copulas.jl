# Correctness obligation: verifies each public dependence-measure inverse on
# representative supported families and both type- and instance-based dispatch.
const _DEPENDENCE_INVERSES =
    (Copulas.τ⁻¹, Copulas.ρ⁻¹, Copulas.β⁻¹, Copulas.λᵤ⁻¹)
const _CHECKED_INVERSE_METHODS =
    Dict(inverse => Set{Method}() for inverse in _DEPENDENCE_INVERSES)
const _EV_INVERSE_CASES = (
    (CuadrasAugeCopula{2}(0.4), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
    (GalambosCopula{2}(1.0), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
    (HuslerReissCopula{2}(1.0), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
    (LogCopula{2}(1.5), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
    (MixedCopula{2}(0.4), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
)
const _TAIL_KENDALL_INVERSE_CASES = (
    Copulas.CuadrasAugeTail(0.4), Copulas.GalambosTail(1.0),
    Copulas.HuslerReissTail(1.0), Copulas.LogTail(1.5),
    Copulas.MixedTail(0.4),
)
const _ARCHIMEDEAN_INVERSE_CASES = (
    AMHCopula{2}(0.5), ClaytonCopula{2}(1.0), FrankCopula{2}(2.0),
    GumbelCopula{2}(1.5), GumbelBarnettCopula{2}(0.5),
    InvGaussianCopula{2}(0.5), JoeCopula{2}(1.5),
)
const _GENERATOR_INVERSE_CASES = (
    Copulas.AMHGenerator(0.5), Copulas.ClaytonGenerator(1.0),
    Copulas.FrankGenerator(2.0), Copulas.GumbelGenerator(1.5),
    Copulas.GumbelBarnettGenerator(0.5),
    Copulas.InvGaussianGenerator(0.5), Copulas.JoeGenerator(1.5),
)
function _record_inverse_route!(inverse, argument_type)
    Base.@nospecialize inverse argument_type
    push!(_CHECKED_INVERSE_METHODS[inverse],
          which(inverse, Tuple{Type{argument_type},Float64}))
end

@testset "public dependence-measure inverses" begin
    inverses = Dict(Copulas.τ => Copulas.τ⁻¹, Copulas.ρ => Copulas.ρ⁻¹,
                    Copulas.β => Copulas.β⁻¹, Copulas.λᵤ => Copulas.λᵤ⁻¹)
    for (C, measures) in _EV_INVERSE_CASES
        CT = typeof(C)
        for measure in measures
            inverse = inverses[measure]
            value = measure(C)
            parameter = inverse(CT, value)
            _record_inverse_route!(inverse, CT)
            rebuilt = ExtremeValueCopula{2}(typeof(C.tail)(parameter))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end

@testset "public tail Kendall inverses" begin
    for tail in _TAIL_KENDALL_INVERSE_CASES
        C = ExtremeValueCopula{2}(tail)
        value = Copulas.τ(C)
        parameter = Copulas.τ⁻¹(typeof(tail), value)
        _record_inverse_route!(Copulas.τ⁻¹, typeof(tail))
        rebuilt = ExtremeValueCopula{2}(typeof(tail)(parameter))
        @test Copulas.τ(rebuilt) ≈ value atol=2e-6
    end
end

@testset "one-parameter copula dependence-measure inverses" begin
    for C in _ARCHIMEDEAN_INVERSE_CASES
        CT = typeof(C)
        for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                                   (Copulas.ρ, Copulas.ρ⁻¹))
            value = measure(C)
            rebuilt = CT(inverse(CT, value))
            _record_inverse_route!(inverse, CT)
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end

    C = FGMCopula{2}(0.5)
    for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                               (Copulas.ρ, Copulas.ρ⁻¹))
        value = measure(C)
        rebuilt = FGMCopula{2}(inverse(FGMCopula{2}, value))
        _record_inverse_route!(inverse, FGMCopula{2})
        @test measure(rebuilt) ≈ value atol=2e-6
    end
end


@testset "generator dependence-measure inverses" begin
    for G in _GENERATOR_INVERSE_CASES
        GT = typeof(G)
        for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                                   (Copulas.ρ, Copulas.ρ⁻¹))
            value = measure(G)
            rebuilt = GT(inverse(GT, value))
            _record_inverse_route!(inverse, GT)
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end


@testset "every public dependence inverse method has an oracle" begin
    reachable = Dict(inverse => Set{Method}() for inverse in _DEPENDENCE_INVERSES)
    inverses = Dict(Copulas.τ => Copulas.τ⁻¹, Copulas.ρ => Copulas.ρ⁻¹,
                    Copulas.β => Copulas.β⁻¹, Copulas.λᵤ => Copulas.λᵤ⁻¹)
    for (C, measures) in _EV_INVERSE_CASES, measure in measures
        inverse = inverses[measure]
        push!(reachable[inverse],
              which(inverse, Tuple{Type{typeof(C)},Float64}))
    end
    for tail in _TAIL_KENDALL_INVERSE_CASES
        push!(reachable[Copulas.τ⁻¹], which(
            Copulas.τ⁻¹, Tuple{Type{typeof(tail)},Float64}))
    end
    for object in (_ARCHIMEDEAN_INVERSE_CASES...,
                   FGMCopula{2}(0.5), _GENERATOR_INVERSE_CASES...)
        for inverse in (Copulas.τ⁻¹, Copulas.ρ⁻¹)
            push!(reachable[inverse],
                  which(inverse, Tuple{Type{typeof(object)},Float64}))
        end
    end
    @test _CHECKED_INVERSE_METHODS == reachable
end
