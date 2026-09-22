@testset "identifiable natural StatsBase coefficients" begin
    P = Copulas.Paramorph

    @test !isdefined(Copulas, :_natural_parameters)
    @test !isdefined(Copulas, :_append_parameter!)
    @test !isdefined(Copulas, :_flatten_params)
    @test !isdefined(Copulas, :_space_coefficients)
    @test !isdefined(Copulas, :_coefficient_parameter_values)
    @test !isdefined(Copulas, :_parameter_dof)

    # Sklar composes natural identifiable component coefficients, with copula
    # then margins. The coefficients remain on the model scale rather than on
    # the optimizer's unconstrained scale.
    D = SklarDist(
        ClaytonCopula(2, 1.25),
        (Normal(2.0, 3.0), Exponential(4.0)),
    )
    @test StatsBase.dof(D) == 4

    M = CopulaModel(D, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(M) == [
        "copula_θ", "margin_1_μ", "margin_1_σ", "margin_2_θ",
    ]
    @test StatsBase.coef(M) == [1.25, 2.0, 3.0, 4.0]
    @test StatsBase.dof(M) == length(StatsBase.coef(M)) == 4
    blocks = Copulas._parameter_blocks(M)
    @test blocks.copula == 1:1
    @test blocks.margins == (2:3, 4:4)

    # Structural constructor arguments are not statistical coefficients.
    Dstruct = SklarDist(
        ClaytonCopula(2, 0.5),
        (Binomial(5, 0.4), Normal()),
    )
    Mstruct = CopulaModel(Dstruct, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(Mstruct) == [
        "copula_θ", "margin_1_p", "margin_2_μ", "margin_2_σ",
    ]
    @test StatsBase.coef(Mstruct) == [0.5, 0.4, 0.0, 1.0]
    @test length(StatsBase.coef(Mstruct)) == StatsBase.dof(Mstruct) == 4

    # Reflections add no natural parameters or optimizer dimensions of their own.
    R = Rotated90Copula(ClaytonCopula(2, 0.75))
    @test Copulas._parameter_dimension(R) == 1
    MR = CopulaModel(R, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(MR) == ["θ"]
    @test StatsBase.coef(MR) == [0.75]

    # Empirical plug-in state is explicitly zero-dimensional in Paramorph and is
    # therefore neither a natural parameter tuple, a StatsBase coefficient
    # vector, nor a statistical dof.
    Xemp = [0.2 0.8; 0.3 0.7]
    E = EmpiricalCopula(Xemp)
    for C in (
        E,
        BetaCopula(Xemp),
        BernsteinCopula(Xemp; m=2),
        CheckerboardCopula(Xemp; m=2),
    )
        @test Distributions.params(C) == ()
        @test StatsBase.dof(C) == 0
        MC = CopulaModel(C, zeros(2, 1), 0.0, nothing)
        @test isempty(StatsBase.coefnames(MC))
        @test isempty(StatsBase.coef(MC))
        @test StatsBase.dof(MC) == 0
    end
    @test P.parameter_fields(Copulas.EmpiricalEVTail) == ()
    @test P.parameter_fields(Copulas.EmpiricalEVMultivariateTail) == ()

    # Ordinary parametric generators expose natural values through their
    # Paramorph logical names rather than through incidental storage order.
    Carch = ClaytonCopula(2, 1.25)
    @test P.parameter_fields(typeof(Carch.G)) == (:θ,)
    @test Distributions.params(Carch) == (1.25,)
    zarch = Copulas._parameter_coordinates(Carch)
    @test only(Distributions.params(Copulas._from_parameter_coordinates(Carch, zarch))) ≈ 1.25

    # Liouville follows the same logical representation: generator parameters
    # first, then the positive α vector. Its product chart belongs to the
    # centralized Copulas bridge; fitting remains deliberately disabled.
    L = LiouvilleCopula(Copulas.ClaytonGenerator(1.25), (0.8, 1.2))
    @test Copulas._parameter_dimension(L) == 3
    @test Distributions.params(L) == (1.25, [0.8, 1.2])
    naturalL = Copulas._from_parameter_coordinates(L, Copulas._parameter_coordinates(L))
    @test naturalL.G.θ ≈ 1.25
    @test collect(naturalL.α) ≈ [0.8, 1.2]
    @test Copulas._available_fitting_methods(typeof(L), 2) == ()

    # Bivariate FGM is an ordinary bounded one-dimensional chart. Multivariate
    # FGM keeps its specialized optimizer because its feasible set is coupled.
    @test P.parameter_fields(FGMCopula{2,Float64}) == (:θ,)
    @test P.intrinsic_dimension(FGMCopula{2,Float64}) == 1
    @test P.constraint(FGMCopula{2,Float64}, [0.0]).θ == [0.0]
    Mfgm = CopulaModel(FGMCopula(2, 0.4), zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(Mfgm) == ["θ"]
    @test StatsBase.coef(Mfgm) == [0.4]
    @test StatsBase.dof(Mfgm) == 1

    # A fixed nested Archimedean tree exposes the local generator parameters.
    N = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(1.0);
        children=[ClaytonCopula{2}(2.0), ClaytonCopula{2}(3.0)],
    )
    @test length(Copulas._nested_unbound(N)) == 3
    @test StatsBase.dof(N) == 3
    MN = CopulaModel(N, zeros(length(N), 1), 0.0, nothing)
    @test StatsBase.dof(MN) == length(StatsBase.coef(MN)) == 3
    Nroundtrip = Copulas._nested_rebound(N, Copulas._nested_unbound(N))
    @test Copulas._nested_coef(Nroundtrip)[2] ≈ Copulas._nested_coef(N)[2]

    # A custom runtime reparametrization can have fewer irreducible fitting
    # coordinates than natural generator parameters. The fit recipe controls dof,
    # but optimizer coordinates must never leak into coef/coefnames.
    runtime_recipe = Copulas._CopulaFitSpec(
        (; reparam=identity, init=[0.0], coordinates=[0.25]),
        :mle,
        (;),
    )
    Mruntime = CopulaModel(N, zeros(length(N), 1), 0.0, runtime_recipe)
    nested_names, nested_values = Copulas._nested_coef(N)
    @test StatsBase.coefnames(Mruntime) == nested_names
    @test StatsBase.coef(Mruntime) == nested_values
    @test StatsBase.dof(Mruntime) == 1
    @test length(StatsBase.coef(Mruntime)) == 3

    # Correlation matrices expose one natural off-diagonal triangle rather than
    # both symmetric halves and the fixed unit diagonal.
    G = GaussianCopula([
        1.0 0.4 0.2
        0.4 1.0 0.3
        0.2 0.3 1.0
    ])
    MG = CopulaModel(G, zeros(3, 1), 0.0, nothing)
    @test StatsBase.coefnames(MG) == ["Σ₁₂", "Σ₁₃", "Σ₂₃"]
    @test StatsBase.coef(MG) == [0.4, 0.2, 0.3]
    @test length(StatsBase.coef(MG)) == StatsBase.dof(MG) == P.intrinsic_dimension(G) == 3
    MGshow = CopulaModel(
        G, zeros(3, 1), 0.0,
        Copulas._CopulaFitSpec(GaussianCopula, :mle, (;)),
    )
    report = sprint(show, MGshow)
    @test occursin("Degrees of freedom", report)
    @test occursin("Σ:", report)
    @test !occursin("Σ₁₂", report)
    @test !occursin("Spearman ρ", report)

    # A simplex exposes k-1 natural probabilities. The omitted final probability
    # is determined by the unit-sum constraint.
    Dsimplex = SklarDist(
        ClaytonCopula(2, 0.5),
        (Categorical([0.1, 0.2, 0.7]), Normal()),
    )
    Msimplex = CopulaModel(Dsimplex, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(Msimplex)[1:3] ==
          ["copula_θ", "margin_1_p₁", "margin_1_p₂"]
    @test StatsBase.coef(Msimplex)[1:3] == [0.5, 0.1, 0.2]
    @test length(StatsBase.coef(Msimplex)) == StatsBase.dof(Msimplex)

    T = TawnCopula(
        2,
        [2.0],
        [0.2, 0.8],
        [0.3, 0.7],
    )
    MT = CopulaModel(T, zeros(2, 1), 0.0, nothing)
    @test StatsBase.coefnames(MT) == ["dep₁", "weights1₁", "weights2₁"]
    @test StatsBase.coef(MT) == [2.0, 0.2, 0.3]
    @test length(StatsBase.coef(MT)) == StatsBase.dof(MT) ==
          Copulas._parameter_dimension(T) == 3

    # Archimax composes the component natural representations instead of
    # indexing storage by Paramorph names.
    AX = ArchimaxCopula(2, Copulas.ClaytonGenerator(1.25), T.tail)
    @test Copulas._parameter_dimension(AX) == 4
    @test Distributions.params(AX) ==
          (1.25, [2.0], [0.2, 0.8], [0.3, 0.7])
    MAX = CopulaModel(AX, zeros(2, 1), 0.0, nothing)
    @test length(StatsBase.coef(MAX)) == StatsBase.dof(MAX) == 4
end
