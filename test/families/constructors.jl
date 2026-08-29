# Family-regression layer: valid public constructor forms and reconstruction are
# covered exhaustively by `obligations/contracts/constructors.jl`; only validation,
# boundary-specialization, keyword, and numeric-parameter regressions remain.

@testset "constructor validation regressions" begin
    data = [0.1 0.4 0.8 0.6; 0.3 0.9 0.2 0.7]
    @test_throws DimensionMismatch EmpiricalCopula{3}(data)
    @test_throws DimensionMismatch GaussianCopula{3}([1.0 0.2; 0.2 1.0])
    @test_throws DimensionMismatch NestedArchimedeanCopula{3}(
        Copulas.ClaytonGenerator(1.0);
        leaves=[1, 2], children=[ClaytonCopula{2}(2.0)])
    @test_throws ArgumentError AsymLogCopula(3, 1.5, 0.4, 0.6)
    @test_throws ArgumentError ExtremeValueCopula(1, Copulas.GalambosTail(0.7))
end

@testset "structured extreme-value dimension validation" begin
    Γ = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
    R = [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0]
    weights = [0.6, 0.7, 0.8]
    a = [0.2, 0.5, 0.8]
    λ = ones(7)
    Uemp = [
        0.20 0.40 0.70
        0.30 0.60 0.80
        0.25 0.55 0.75
    ]
    @test_throws ArgumentError HuslerReissCopula{4}(Γ)
    @test_throws ArgumentError HuslerReissCopula(4, Γ)
    @test_throws ArgumentError tEVCopula{4}(4.0, R)
    @test_throws ArgumentError tEVCopula(4, 4.0, R)
    @test_throws ArgumentError TawnCopula{4}(2.0, weights)
    @test_throws ArgumentError AsymGalambosCopula{4}(0.7, weights)
    @test_throws ArgumentError BC2Copula{4}(a)
    @test_throws ArgumentError MOCopula{4}(λ)
    @test_throws DimensionMismatch EmpiricalEVCopula{4}(Uemp; degree=1)
    @test_throws ArgumentError MOCopula(ones(5))
end

@testset "constructor input-type regressions" begin
    @test ClaytonCopula{2}(2) isa ClaytonCopula{2}
    @test BB1Copula{2}(1, 2) isa BB1Copula{2}
    @test GalambosCopula{2}(2) isa GalambosCopula{2}
    @test tEVCopula{2}(4, 0.5) isa tEVCopula{2}
    @test GalambosCopula(2; θ=1.0) isa GalambosCopula{2}
    @test params(LogCopula{2}(2)).θ == 2.0
    @test params(MixedCopula{2}(1)).θ == 1.0
    @test params(HuslerReissCopula{2}(1)).θ == 1.0
    @test params(tEVCopula{2}(4, 0.2)).ν == 4
    Cint = LogCopula{2}(2)
    @test params(typeof(Cint)(2)).θ == 2.0
    @test params(LogCopula(2, 2)).θ == 2.0
    @test_throws MethodError GalambosCopula(2.3)
    @test_throws MethodError MixedCopula(0.5)
end
