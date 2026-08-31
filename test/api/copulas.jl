# Public-API fixtures and proof that the central bestiary
# represents every public copula family. Operation contracts live under
# `test/operations/`.
struct CopulaContractContext
    u::Vector{Float64}
    U::Matrix{Float64}
end

# Deliberately omits the matrix sampler required from concrete copulas. It
# exercises the generic developer-facing diagnostic without duplicating a
# public operation implementation.
struct MissingSamplerContractCopula <: Copulas.Copula{2} end

@testset "copula measure-style trait" begin
    discrete_radial = WilliamsonGenerator([1.0, 2.0], [0.4, 0.6], 3)
    @test Copulas.copula_measure_style(ArchimedeanCopula{3}(discrete_radial)) isa
          Copulas.NonAbsolutelyContinuousMeasure
    # Marginalization multiplies the preserved radial by a continuous beta
    # variable, so a positive discrete source becomes absolutely continuous.
    @test Copulas.copula_measure_style(ArchimedeanCopula{2}(discrete_radial)) isa
          Copulas.AbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(
        ArchimedeanCopula{2}(WilliamsonGenerator(Uniform(1.0, 2.0), 2)),
    ) isa Copulas.AbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(ClaytonCopula{3}(-0.5)) isa
          Copulas.NonAbsolutelyContinuousMeasure

    discrete_liouville = LiouvilleCopula{2}(
        WilliamsonGenerator([1.0, 2.0], [0.4, 0.6], 2), (0.8, 1.2),
    )
    @test Copulas.copula_measure_style(discrete_liouville) isa
          Copulas.NonAbsolutelyContinuousMeasure
    discrete_archimax = ArchimaxCopula{2}(
        WilliamsonGenerator([1.0, 2.0], [0.4, 0.6], 2),
        Copulas.GalambosTail(1.0),
    )
    @test Copulas.copula_measure_style(discrete_archimax) isa
          Copulas.NonAbsolutelyContinuousMeasure

    singular = RafteryCopula{3}(0.5)
    @test Copulas.copula_measure_style(
        Copulas.SubsetCopula(singular, (1, 2)),
    ) isa Copulas.NonAbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(
        SurvivalCopula{3}(singular, (1,)),
    ) isa Copulas.NonAbsolutelyContinuousMeasure
end

function copula_contract_context(C, seed)
    Base.@nospecialize C
    d = length(C)
    u = collect(range(0.31, 0.69; length=d))
    U = rand(StableRNG(seed), C, 4)
    return CopulaContractContext(u, U)
end

@testset "public copula registry is exhaustive" begin
    public_families = Set(getfield(Copulas, symbol) for symbol in public_symbols()
        if getfield(Copulas, symbol) isa Type &&
           symbol !== :Copula &&
           getfield(Copulas, symbol) <: Copulas.Copula)
    represented = Set(typeof(fixture.copula) for fixture in COPULA_FIXTURES)
    @test all(F -> any(T -> T <: F, represented), public_families)
    @test all(T -> any(F -> T <: F, public_families), represented)
end

@testset "collection adapters preserve the public semantics" begin
    C = ClaytonCopula{3}(1.5)
    u = [0.3, 0.5, 0.7]
    @test Base.broadcastable(C)[] === C
    @test cdf(condition(C, [1], [u[1]]), u[2:3]) ≈
          cdf(condition(C, (1,), (u[1],)), u[2:3])
end
