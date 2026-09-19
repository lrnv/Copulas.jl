@testset "natural StatsBase coefficients" begin
    P = Copulas.Paramorph

    @test !isdefined(Copulas, :_natural_parameters)
    @test !isdefined(Copulas, :_append_parameter!)
    @test !isdefined(Copulas, :_flatten_params)
    @test !isdefined(Copulas, :_space_coefficients)
    @test !isdefined(Copulas, :_coefficient_parameter_values)

    # Sklar keeps Paramorph only for geometry/dimension. StatsBase coefficients
    # are the natural component parameters, with copula then margins.
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
    @test StatsBase.dof(M) == 4
    blocks = Copulas._parameter_blocks(M)
    @test blocks.copula == 1:1
    @test blocks.margins == (2:3, 4:4)

    # Structural values may be present in the natural representation even when
    # Paramorph does not optimize them. Binomial n is displayed but does not
    # consume a degree of freedom.
    Dstruct = SklarDist(
        ClaytonCopula(2, 0.5),
        (Binomial(5, 0.4), Normal()),
    )
    Mstruct = CopulaModel(Dstruct, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(Mstruct) == [
        "copula_θ", "margin_1_n", "margin_1_p", "margin_2_μ", "margin_2_σ",
    ]
    @test StatsBase.coef(Mstruct) == [0.5, 5.0, 0.4, 0.0, 1.0]
    @test length(StatsBase.coef(Mstruct)) == 5
    @test StatsBase.dof(Mstruct) == 4

    # Reflections add no natural parameters of their own.
    R = Rotated90Copula(ClaytonCopula(2, 0.75))
    @test P.names(P.param_space(R)) == (:θ,)
    MR = CopulaModel(R, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(MR) == ["θ"]
    @test StatsBase.coef(MR) == [0.75]

    # Purely nonparametric state is not a StatsBase coefficient vector.
    E = EmpiricalCopula([0.2 0.8; 0.3 0.7])
    @test P.names(P.param_space(E)) == ()
    ME = CopulaModel(E, zeros(2, 1), 0.0, nothing)
    @test isempty(StatsBase.coefnames(ME))
    @test isempty(StatsBase.coef(ME))
    @test StatsBase.dof(ME) == 0

    # A natural matrix is exposed in full. Symmetry and the fixed unit diagonal
    # therefore create redundant coefficients, while dof remains the Paramorph
    # dimension of the correlation manifold.
    G = GaussianCopula([
        1.0 0.4 0.2
        0.4 1.0 0.3
        0.2 0.3 1.0
    ])
    MG = CopulaModel(G, zeros(3, 1), 0.0, nothing)
    @test StatsBase.coefnames(MG) == [
        "Σ₁₁", "Σ₂₁", "Σ₃₁",
        "Σ₁₂", "Σ₂₂", "Σ₃₂",
        "Σ₁₃", "Σ₂₃", "Σ₃₃",
    ]
    @test StatsBase.coef(MG) == vec(G.Σ)
    @test length(StatsBase.coef(MG)) == 9
    @test StatsBase.dof(MG) == P.dimension(P.param_space(G)) == 3
    MGshow = CopulaModel(
        G, zeros(3, 1), 0.0,
        Copulas._CopulaFitSpec(GaussianCopula, :mle, (;)),
    )
    report = sprint(show, MGshow)
    @test occursin("Degrees of freedom", report)
    @test occursin("Σ:", report)
    @test !occursin("Σ₁₂", report)

    # Simplex vectors likewise expose every natural probability even though one
    # entry is redundant in the optimization geometry.
    Dsimplex = SklarDist(
        ClaytonCopula(2, 0.5),
        (Categorical([0.1, 0.2, 0.7]), Normal()),
    )
    Msimplex = CopulaModel(Dsimplex, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(Msimplex)[1:4] ==
          ["copula_θ", "margin_1_p₁", "margin_1_p₂", "margin_1_p₃"]
    @test StatsBase.coef(Msimplex)[1:4] == [0.5, 0.1, 0.2, 0.7]
    @test length(StatsBase.coef(Msimplex)) == StatsBase.dof(Msimplex) + 1

    T = TawnCopula(
        2,
        [2.0],
        [0.2, 0.8],
        [0.3, 0.7],
    )
    MT = CopulaModel(T, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(MT) == [
        "dep₁", "weights1₁", "weights1₂", "weights2₁", "weights2₂",
    ]
    @test StatsBase.coef(MT) == [2.0, 0.2, 0.8, 0.3, 0.7]
    @test StatsBase.dof(MT) == P.dimension(P.param_space(T)) == 3
end
