@testset "public copula constructors require d ≥ 2" begin
    @test_throws ArgumentError IndependentCopula{1}()
    @test_throws ArgumentError IndependentCopula(1)
    @test_throws ArgumentError MCopula{0}()
    @test_throws ArgumentError MCopula(1)

    @test_throws ArgumentError RafteryCopula{1}(0.5)
    @test_throws ArgumentError RafteryCopula(1, 0.5)
    @test_throws ArgumentError FGMCopula{1}(Float64[])
    @test_throws ArgumentError FGMCopula(1, Float64[])

    Σ1 = ones(1, 1)
    @test_throws ArgumentError GaussianCopula{1}(copy(Σ1))
    @test_throws ArgumentError GaussianCopula(copy(Σ1))
    @test_throws ArgumentError TCopula{1}(4.0, copy(Σ1))
    @test_throws ArgumentError TCopula(4.0, copy(Σ1))

    data1 = reshape([0.15, 0.35, 0.65, 0.85], 1, :)
    @test_throws ArgumentError EmpiricalCopula{1}(data1)
    @test_throws ArgumentError EmpiricalCopula(data1)
    @test_throws ArgumentError BetaCopula{1}(data1)
    @test_throws ArgumentError BetaCopula(data1)
    @test_throws ArgumentError CheckerboardCopula{1}(data1; m=2)
    @test_throws ArgumentError CheckerboardCopula(data1; m=2)
    @test_throws ArgumentError BernsteinCopula{1}(data1; m=2)
    @test_throws ArgumentError BernsteinCopula(data1; m=2)

    G = Copulas.ClaytonGenerator(1.5)
    @test_throws ArgumentError ArchimedeanCopula{1}(G)
    @test_throws ArgumentError ArchimedeanCopula(1, G)
    @test_throws ArgumentError ClaytonCopula{1}(1.5)
    @test_throws ArgumentError ClaytonCopula(1, 1.5)

    # Existing family-specific restrictions remain stronger than the package
    # minimum, while ordinary bivariate constructions remain unchanged.
    @test_throws ArgumentError WCopula{3}()
    @test IndependentCopula{2}() isa IndependentCopula{2}
    @test MCopula(2) isa MCopula{2}
    @test ClaytonCopula(2, 1.5) isa ClaytonCopula{2}
    @test GaussianCopula([1.0 0.2; 0.2 1.0]) isa GaussianCopula{2}
end

@testset "one-coordinate reductions use univariate distributions" begin
    @test subsetdims(IndependentCopula{3}(), (2,)) isa Uniform
    @test subsetdims(MCopula{3}(), (1,)) isa Uniform
    @test condition(IndependentCopula{2}(), (1,), (0.4,)) isa Uniform
end
