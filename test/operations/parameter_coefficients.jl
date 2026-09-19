@testset "natural StatsBase coefficients" begin
    P = Copulas.Paramorph

    @test !isdefined(Copulas, :_natural_parameters)
    @test !isdefined(Copulas, :_append_parameter!)
    @test !isdefined(Copulas, :_flatten_params)
    @test !isdefined(Copulas, :_space_coefficients)
    @test !isdefined(Copulas, :_coefficient_parameter_values)
    @test !isdefined(Copulas, :_parameter_dof)

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

    # Empirical plug-in state is explicitly zero-dimensional in Paramorph and is
    # therefore neither a StatsBase coefficient vector nor a statistical dof.
    Xemp = [0.2 0.8; 0.3 0.7]
    E = EmpiricalCopula(Xemp)
    for C in (E, BetaCopula(Xemp), CheckerboardCopula(Xemp; m=2))
        @test P.dimension(P.param_space(C)) == 0
        @test StatsBase.dof(C) == 0
        MC = CopulaModel(C, zeros(2, 1), 0.0, nothing)
        @test isempty(StatsBase.coefnames(MC))
        @test isempty(StatsBase.coef(MC))
        @test StatsBase.dof(MC) == 0
    end
    @test P.param_space(BernsteinCopula, 2) == ()
    @test P.param_space(Copulas.EmpiricalEVTail, 2) == ()
    @test P.param_space(Copulas.EmpiricalEVMultivariateTail, 3) == ()

    # Bivariate FGM is an ordinary bounded one-dimensional chart. Multivariate
    # FGM keeps its specialized optimizer because its feasible set is coupled.
    pfgm = P.param_space(FGMCopula, 2)
    @test P.names(pfgm) == (:θ,)
    @test P.dimension(pfgm) == 1
    @test P.constrain(pfgm, [0.0]) == 0.0
    Mfgm = CopulaModel(FGMCopula(2, 0.4), zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(Mfgm) == ["θ₁"]
    @test StatsBase.coef(Mfgm) == [0.4]
    @test StatsBase.dof(Mfgm) == 1

    # A fixed nested Archimedean tree is a Cartesian product of the local
    # generator spaces. Paramorph owns its flat chart and statistical dimension;
    # Copulas only owns the tree traversal and cross-node nesting certificates.
    N = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(1.0);
        children=[ClaytonCopula{2}(2.0), ClaytonCopula{2}(3.0)],
    )
    pN = P.param_space(N)
    @test P.dimension(pN) == 3
    @test StatsBase.dof(N) == 3
    MN = CopulaModel(N, zeros(length(N), 1), 0.0, nothing)
    @test StatsBase.dof(MN) == 3
    @test length(StatsBase.coef(MN)) == 3
    Nroundtrip = Copulas._nested_rebound(N, Copulas._nested_unbound(N))
    @test Copulas._nested_coef(Nroundtrip)[2] ≈ Copulas._nested_coef(N)[2]

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
    @test !occursin("Spearman ρ", report)

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
