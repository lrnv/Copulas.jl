# Public fitting-target validation for the documented
# `SklarDist{CopulaType,Tuple{MarginTypes...}}` syntax.

@testset "Sklar fitting target validates margin arity atomically" begin
    source = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    data = rand(StableRNG(52_901), source, 24)

    exact = SklarDist{ClaytonCopula,Tuple{Normal,Exponential}}
    @test fit(exact, data; copula_method=:itau) isa SklarDist
    @test fit(CopulaModel, exact, data; copula_method=:itau) isa CopulaModel

    too_few = SklarDist{ClaytonCopula,Tuple{Normal}}
    too_many = SklarDist{ClaytonCopula,Tuple{Normal,Exponential,Gamma}}
    malformed = SklarDist{ClaytonCopula,Tuple{Normal,Int}}

    for target in (too_few, too_many)
        @test_throws DimensionMismatch fit(target, data; copula_method=:itau)
        @test_throws DimensionMismatch fit(
            CopulaModel, target, data; copula_method=:itau,
        )
    end

    @test_throws ArgumentError fit(malformed, data; copula_method=:itau)
    @test_throws ArgumentError fit(
        CopulaModel, malformed, data; copula_method=:itau,
    )
end
