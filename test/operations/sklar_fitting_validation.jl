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


struct _BrokenWeightedMargin <: ContinuousUnivariateDistribution end
_broken_weighted_fit_inner(::Int) = nothing
Distributions.fit(::Type{_BrokenWeightedMargin}, x, w) =
    _broken_weighted_fit_inner(:boom)

@testset "weighted margin errors distinguish capability from implementation failures" begin
    err = try
        Copulas._fit_margin(_BrokenWeightedMargin, [1.0, 2.0], ones(2))
    catch e
        e
    end
    @test err isa MethodError
    @test err.f === _broken_weighted_fit_inner
    @test_throws ArgumentError Copulas._fit_margin(Cauchy, [0.1, 0.2], ones(2))
end
