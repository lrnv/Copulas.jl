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

@testset "nested Archimedean constructor and boundary regressions" begin
    G = Copulas.ClaytonGenerator(2.0)
    invalid = (
        (; leaves=[1, 1]),
        (; leaves=[1], children=[ClaytonCopula{2}(5.0) => [1, 2]]),
        (; children=[ClaytonCopula{2}(5.0) => [1]]),
        (; children=[ClaytonCopula{2}(5.0) => [2, 3]]),
        (; leaves=[0], children=[ClaytonCopula{2}(5.0)]),
        (; leaves=[-1], children=[ClaytonCopula{2}(5.0)]),
        (; children=Any[42]),
        (; children=Any[42 => [1]]),
        (; leaves=[2], children=[ClaytonCopula{2}(5.0)]),
    )
    for kwargs in invalid
        @test_throws ArgumentError NestedArchimedeanCopula(G; kwargs...)
    end

    placed = NestedArchimedeanCopula(G;
        leaves=[3], children=[ClaytonCopula{2}(5.0)])
    @test placed.children[1][2] == [1, 2]
    @test NestedArchimedeanCopula(G,
        [ClaytonCopula{2}(5.0), ClaytonCopula{2}(6.0)]) isa
        NestedArchimedeanCopula{4}

    C = NestedArchimedeanCopula(G;
        children=[ClaytonCopula{2}(5.0), ClaytonCopula{2}(6.0)])
    u = [0.3, 0.4, 0.6, 0.7]
    @test cdf(C, [u[1], u[2], 1.0, 1.0]) ≈
          cdf(ClaytonCopula{2}(5.0), u[1:2])
    for point in ([0, 1, 1, 1], [u[1], 1.0, u[3], u[4]],
                  [u[1], -0.1, u[3], u[4]], [u[1], Inf, u[3], u[4]],
                  [u[1], NaN, u[3], u[4]])
        @test logpdf(C, point) == -Inf
    end
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

@testset "Gaussian equicorrelation constructor boundary" begin
    @test GaussianCopula{2}(0.5) isa GaussianCopula{2}
    @test GaussianCopula{3}(-0.49) isa GaussianCopula{3}
    @test_throws ArgumentError GaussianCopula{3}(-0.5)
end

@testset "structured EV tail validation" begin
    good = [[0.15], [0.20], [0.10], [0.25, 0.15], [0.20, 0.20],
            [0.25, 0.30], [0.40, 0.40, 0.40]]
    for (constructor, dep, invalid_dep) in (
        (Copulas.TawnTail, [1.4, 2.0, 1.7, 2.3], 0.8),
        (Copulas.AsymGalambosTail, [0.7, 1.3, 0.9, 1.8], -0.1),
    )
        @test_throws DimensionMismatch constructor(3, dep[1:3], good)
        badsum = deepcopy(good)
        badsum[end][1] = 0.30
        @test_throws ArgumentError constructor(3, dep, badsum)
        baddep = copy(dep)
        baddep[2] = invalid_dep
        @test_throws ArgumentError constructor(3, baddep, good)
    end

    @test_throws DimensionMismatch Copulas.tEVTail(1.5, zeros(3, 4))
    @test_throws ArgumentError Copulas.tEVTail(
        0.0, Matrix{Float64}(I, 3, 3))
    @test_throws ArgumentError Copulas.tEVTail(1.5,
        [1.0 0.3 0.0; 0.1 1.0 0.2; 0.0 0.2 1.0])
    @test_throws ArgumentError Copulas.tEVTail(1.5,
        [1.0 0.95 0.95; 0.95 1.0 -0.95; 0.95 -0.95 1.0])
end
