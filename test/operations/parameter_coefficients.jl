@testset "Paramorph coefficient structure" begin
    P = Copulas.Paramorph

    # Sklar is a Cartesian product of prefixed component spaces. Names and
    # coefficient blocks follow that structure rather than reparsing strings.
    D = SklarDist(
        ClaytonCopula(2, 1.25),
        (Normal(2.0, 3.0), Exponential(4.0)),
    )
    p = P.param_space(D)
    @test P.names(p) == (:copula_θ, :margin_1_μ, :margin_1_σ, :margin_2_θ)

    M = CopulaModel(D, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(M) == [
        "copula_θ", "margin_1_μ", "margin_1_σ", "margin_2_θ",
    ]
    @test StatsBase.coef(M) == [1.25, 2.0, 3.0, 4.0]
    blocks = Copulas._parameter_blocks(M)
    @test blocks.copula == [1]
    @test blocks.margins == ([2, 3], [4])

    # Structural constructor arguments are absent from Paramorph and therefore
    # from model coefficients. Binomial's trial count is fixed; only p is free.
    Dstruct = SklarDist(
        ClaytonCopula(2, 0.5),
        (Binomial(5, 0.4), Normal()),
    )
    Mstruct = CopulaModel(Dstruct, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(Mstruct) == [
        "copula_θ", "margin_1_p", "margin_2_μ", "margin_2_σ",
    ]
    @test StatsBase.coef(Mstruct) == [0.5, 0.4, 0.0, 1.0]
    @test StatsBase.dof(Mstruct) == 4

    # Reflections carry no parameter geometry of their own.
    R = Rotated90Copula(ClaytonCopula(2, 0.75))
    @test P.names(P.param_space(R)) == (:θ,)
    MR = CopulaModel(R, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(MR) == ["θ"]
    @test StatsBase.coef(MR) == [0.75]

    # Empirical sample points are structural state, not coefficients.
    E = EmpiricalCopula([0.2 0.8; 0.3 0.7])
    @test P.names(P.param_space(E)) == ()
    ME = CopulaModel(E, zeros(2, 1), 0.0, nothing)
    @test isempty(StatsBase.coefnames(ME))
    @test isempty(StatsBase.coef(ME))

    # Matrix and simplex coefficientization is driven by the space. A
    # correlation matrix contributes only off-diagonal entries, while a
    # simplex uses a fixed natural convention (omit its first entry) so the
    # coefficient count equals the number of free coordinates without leaking
    # Paramorph's optimization-chart anchor into the public coefficient names.
    G = GaussianCopula([
        1.0 0.4 0.2
        0.4 1.0 0.3
        0.2 0.3 1.0
    ])
    MG = CopulaModel(G, zeros(3, 1), 0.0, nothing)
    @test StatsBase.coefnames(MG) == ["Σ₁₂", "Σ₁₃", "Σ₂₃"]
    @test StatsBase.coef(MG) == [0.4, 0.2, 0.3]
    @test StatsBase.dof(MG) == P.dimension(P.param_space(G)) == 3

    p1 = P.Simplex(:p, [0.8, 0.1, 0.1])
    p2 = P.Simplex(:p, [0.1, 0.8, 0.1])
    @test p1.anchor != p2.anchor
    @test Copulas._space_coefficients(p1, ([0.8, 0.1, 0.1],)) ==
          (["p₂", "p₃"], [0.1, 0.1])
    @test Copulas._space_coefficients(p2, ([0.1, 0.8, 0.1],)) ==
          (["p₂", "p₃"], [0.8, 0.1])

    T = TawnCopula(
        2,
        [2.0],
        [0.2, 0.8],
        [0.3, 0.7],
    )
    MT = CopulaModel(T, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(MT) == ["dep₁", "weights1₂", "weights2₂"]
    @test StatsBase.coef(MT) == [2.0, 0.8, 0.7]
    @test StatsBase.dof(MT) == P.dimension(P.param_space(T)) == 3
end
