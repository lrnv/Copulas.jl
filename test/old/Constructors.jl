# Legacy migration layer: valid public constructor forms and reconstruction are
# covered exhaustively by `contracts/constructors.jl`; only validation,
# boundary-specialization, keyword, and numeric-parameter regressions remain.

@testset "constructor validation regressions" begin
    data = [0.1 0.4 0.8 0.6; 0.3 0.9 0.2 0.7]
    @test_throws DimensionMismatch EmpiricalCopula{3}(data)
    @test_throws DimensionMismatch GaussianCopula{3}([1.0 0.2; 0.2 1.0])
    @test_throws DimensionMismatch NestedArchimedeanCopula{3}(
        Copulas.ClaytonGenerator(1.0);
        leaves=[1, 2], children=[ClaytonCopula{2}(2.0)])
end

@testset "constructor boundary and input-type regressions" begin
    @test ClaytonCopula{2}(2) isa ClaytonCopula{2}
    @test BB1Copula{2}(1, 2) isa BB1Copula{2}
    @test GalambosCopula{2}(2) isa GalambosCopula{2}
    @test tEVCopula{2}(4, 0.5) isa tEVCopula{2}
    @test GalambosCopula(2; θ=1.0) isa GalambosCopula{2}
    @test CuadrasAugeCopula{2}(0.0) isa IndependentCopula{2}
    @test CuadrasAugeCopula{2}(1.0) isa MCopula{2}
end
