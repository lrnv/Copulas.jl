#### Each entry is an edge of the historical constructor-reduction graph.
#### Constructors now preserve `expected_family`; numerical methods must still
#### agree with the former canonical target at every listed boundary.

#### Loaded by `runtests.jl` before the central bestiary, which consumes its
#### source constructors. The assertions are started later by
#### `reduction_graph_tests.jl`, once the test hierarchy is running.


const CONSTRUCTOR_REDUCTIONS = Any[

    # ============================================================
    # Archimedean generators -> canonical generators/copulas
    # ============================================================

    (
        name = "AMH -> Independent",
        source = constructor_spec(AMHCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),

    (
        name = "Clayton -> W",
        source = constructor_spec(ClaytonCopula, 2, -1.0),
        target = constructor_spec(WCopula, 2),
    ),
    (
        name = "Clayton -> Independent",
        source = constructor_spec(ClaytonCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "Clayton -> M",
        source = constructor_spec(ClaytonCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "Frank -> W",
        source = constructor_spec(FrankCopula, 2, -Inf),
        target = constructor_spec(WCopula, 2),
    ),
    (
        name = "Frank -> Independent",
        source = constructor_spec(FrankCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "Frank -> M",
        source = constructor_spec(FrankCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "Gumbel -> Independent",
        source = constructor_spec(GumbelCopula, 2, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "Gumbel -> M",
        source = constructor_spec(GumbelCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "Joe -> Independent",
        source = constructor_spec(JoeCopula, 2, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "Joe -> M",
        source = constructor_spec(JoeCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "GumbelBarnett -> Independent",
        source = constructor_spec(GumbelBarnettCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),

    (
        name = "InvGaussian -> Independent",
        source = constructor_spec(InvGaussianCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),


    # ============================================================
    # BB Archimedean families -> simpler Archimedean families
    # ============================================================

    (
        name = "BB1 -> Clayton",
        source = constructor_spec(BB1Copula, 2, 1.5, 1.0),
        target = constructor_spec(ClaytonCopula, 2, 1.5),
    ),
    (
        name = "BB1 -> M through Clayton",
        source = constructor_spec(BB1Copula, 2, Inf, 1.0),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "BB6 -> Joe",
        source = constructor_spec(BB6Copula, 2, 1.5, 1.0),
        target = constructor_spec(JoeCopula, 2, 1.5),
    ),
    (
        name = "BB6 -> Gumbel",
        source = constructor_spec(BB6Copula, 2, 1.0, 1.5),
        target = constructor_spec(GumbelCopula, 2, 1.5),
    ),
    (
        name = "BB6 -> Independent through Joe",
        source = constructor_spec(BB6Copula, 2, 1.0, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "BB6 -> M through Joe",
        source = constructor_spec(BB6Copula, 2, Inf, 1.0),
        target = constructor_spec(MCopula, 2),
    ),
    (
        name = "BB6 -> M through Gumbel",
        source = constructor_spec(BB6Copula, 2, 1.0, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "BB7 -> Clayton",
        source = constructor_spec(BB7Copula, 2, 1.0, 1.5),
        target = constructor_spec(ClaytonCopula, 2, 1.5),
    ),
    (
        name = "BB7 -> M through Clayton",
        source = constructor_spec(BB7Copula, 2, 1.0, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "BB8 -> Joe",
        source = constructor_spec(BB8Copula, 2, 1.5, 1.0),
        target = constructor_spec(JoeCopula, 2, 1.5),
    ),
    (
        name = "BB8 -> Independent through Joe",
        source = constructor_spec(BB8Copula, 2, 1.0, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "BB8 -> M through Joe",
        source = constructor_spec(BB8Copula, 2, Inf, 1.0),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "BB10 -> AMH",
        source = constructor_spec(BB10Copula, 2, 1.0, 0.5),
        target = constructor_spec(AMHCopula, 2, 0.5),
    ),
    (
        name = "BB10 -> Independent through AMH",
        source = constructor_spec(BB10Copula, 2, 1.0, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),


    # ============================================================
    # Extreme-value tails -> canonical tails/copulas
    # ============================================================

    (
        name = "Galambos -> Independent",
        source = constructor_spec(GalambosCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "Galambos -> M",
        source = constructor_spec(GalambosCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "Log -> Independent",
        source = constructor_spec(LogCopula, 2, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "Log -> M",
        source = constructor_spec(LogCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "HuslerReiss scalar -> Independent",
        source = constructor_spec(HuslerReissCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "HuslerReiss scalar -> M",
        source = constructor_spec(HuslerReissCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),
    (
        name = "HuslerReiss zero variogram -> M",
        source = constructor_spec(HuslerReissCopula, 3, zeros(3, 3)),
        target = constructor_spec(MCopula, 3),
    ),

    (
        name = "CuadrasAuge -> Independent",
        source = constructor_spec(CuadrasAugeCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "CuadrasAuge -> M",
        source = constructor_spec(CuadrasAugeCopula, 2, 1.0),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "Mixed -> Independent",
        source = constructor_spec(MixedCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),


    # ============================================================
    # Asymmetric EV -> simpler EV families
    # ============================================================

    (
        name = "AsymMixed -> Independent",
        source = constructor_spec(AsymMixedCopula, 2, 0.0, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymMixed -> Mixed",
        source = constructor_spec(AsymMixedCopula, 2, 0.5, 0.0),
        target = constructor_spec(MixedCopula, 2, 0.5),
    ),

    (
        name = "AsymLog alpha=1 -> Independent",
        source = constructor_spec(AsymLogCopula, 2, 1.0, 0.4, 0.6),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymLog theta1=0 -> Independent",
        source = constructor_spec(AsymLogCopula, 2, 1.5, 0.0, 0.6),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymLog theta2=0 -> Independent",
        source = constructor_spec(AsymLogCopula, 2, 1.5, 0.4, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymLog -> Log",
        source = constructor_spec(AsymLogCopula, 2, 1.5, 1.0, 1.0),
        target = constructor_spec(LogCopula, 2, 1.5),
    ),
    (
        name = "AsymLog -> Independent through Log",
        source = constructor_spec(AsymLogCopula, 2, 1.0, 1.0, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymLog -> M through Log",
        source = constructor_spec(AsymLogCopula, 2, Inf, 1.0, 1.0),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "AsymGalambos alpha=0 -> Independent",
        source = constructor_spec(AsymGalambosCopula, 2, 0.0, 0.4, 0.6),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymGalambos inactive support -> Independent",
        source = constructor_spec(AsymGalambosCopula, 2, 1.0, 0.0, 0.6),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymGalambos -> Galambos",
        source = constructor_spec(AsymGalambosCopula, 2, 1.5, 1.0, 1.0),
        target = constructor_spec(GalambosCopula, 2, 1.5),
    ),
    (
        name = "AsymGalambos -> Independent through Galambos",
        source = constructor_spec(AsymGalambosCopula, 2, 0.0, 1.0, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),
    (
        name = "AsymGalambos -> M through Galambos",
        source = constructor_spec(AsymGalambosCopula, 2, Inf, 1.0, 1.0),
        target = constructor_spec(MCopula, 2),
    ),


    # ============================================================
    # Tawn
    # ============================================================

    (
        name = "Tawn inactive logistic component -> Independent",
        source = constructor_spec(TawnCopula, 3, 1.0, [0.6, 0.7, 0.8]),
        target = constructor_spec(IndependentCopula, 3),
    ),
    (
        name = "Tawn -> Log",
        source = constructor_spec(TawnCopula, 3, 1.5, ones(3)),
        target = constructor_spec(LogCopula, 3, 1.5),
    ),
    (
        name = "Tawn -> M through Log",
        source = constructor_spec(TawnCopula, 3, Inf, ones(3)),
        target = constructor_spec(MCopula, 3),
    ),


    # ============================================================
    # extremal-t
    # ============================================================

    (
        name = "tEV scalar rho=1 -> M",
        source = constructor_spec(tEVCopula, 2, 4.0, 1.0),
        target = constructor_spec(MCopula, 2),
    ),
    # (
    #     name = "tEV all-ones correlation -> M",
    #     source = constructor_spec(tEVCopula, 3, 4.0, ones(3, 3)),
    #     target = constructor_spec(MCopula, 3),
    # ),


    # ============================================================
    # Archimax generic reductions
    # ============================================================

    (
        name = "Archimax NoTail -> Archimedean",
        source = constructor_spec(
            ArchimaxCopula,
            2,
            Copulas.ClaytonGenerator(1.5),
            Copulas.NoTail(),
        ),
        target = constructor_spec(ClaytonCopula, 2, 1.5),
    ),

    (
        name = "Archimax IndependentGenerator -> ExtremeValue",
        source = constructor_spec(
            ArchimaxCopula,
            2,
            Copulas.IndependentGenerator(),
            Copulas.GalambosTail(1.5),
        ),
        target = constructor_spec(GalambosCopula, 2, 1.5),
    ),


    # ============================================================
    # BB4 = Archimax(Clayton, Galambos)
    # ============================================================

    (
        name = "BB4 -> Clayton",
        source = constructor_spec(BB4Copula, 2, 1.5, 0.0),
        target = constructor_spec(ClaytonCopula, 2, 1.5),
    ),
    (
        name = "BB4 -> Galambos",
        source = constructor_spec(BB4Copula, 2, 0.0, 1.5),
        target = constructor_spec(GalambosCopula, 2, 1.5),
    ),
    (
        name = "BB4 -> M through Clayton",
        source = constructor_spec(BB4Copula, 2, Inf, 0.0),
        target = constructor_spec(MCopula, 2),
    ),
    (
        name = "BB4 -> M through Galambos",
        source = constructor_spec(BB4Copula, 2, 0.0, Inf),
        target = constructor_spec(MCopula, 2),
    ),


    # ============================================================
    # BB5 = Archimax(Gumbel, Galambos)
    # ============================================================

    (
        name = "BB5 -> Gumbel",
        source = constructor_spec(BB5Copula, 2, 1.5, 0.0),
        target = constructor_spec(GumbelCopula, 2, 1.5),
    ),
    (
        name = "BB5 -> Galambos",
        source = constructor_spec(BB5Copula, 2, 1.0, 1.5),
        target = constructor_spec(GalambosCopula, 2, 1.5),
    ),
    (
        name = "BB5 -> M through Gumbel",
        source = constructor_spec(BB5Copula, 2, Inf, 0.0),
        target = constructor_spec(MCopula, 2),
    ),
    (
        name = "BB5 -> M through Galambos",
        source = constructor_spec(BB5Copula, 2, 1.0, Inf),
        target = constructor_spec(MCopula, 2),
    ),


    # ============================================================
    # Liouville -> Archimedean
    # ============================================================

    (
        name = "Liouville -> Archimedean/Clayton",
        source = constructor_spec(
            LiouvilleCopula,
            2,
            Copulas.ClaytonGenerator(1.5),
            (1.0, 1.0),
        ),
        target = constructor_spec(ClaytonCopula, 2, 1.5),
    ),

    (
        name = "Liouville -> Archimedean/Gumbel",
        source = constructor_spec(
            LiouvilleCopula,
            2,
            Copulas.GumbelGenerator(1.5),
            (1.0, 1.0),
        ),
        target = constructor_spec(GumbelCopula, 2, 1.5),
    ),


    # ============================================================
    # Nested Archimedean -> flat Archimedean
    # ============================================================

    (
        name = "NestedArchimedean no children -> Archimedean",
        source = constructor_spec(
            NestedArchimedeanCopula,
            3,
            Copulas.ClaytonGenerator(1.5);
            leaves = [1, 2, 3],
            children = [],
        ),
        target = constructor_spec(ClaytonCopula, 3, 1.5),
    ),


    # ============================================================
    # Other value-dependent public constructors from the bestiary
    # ============================================================

    (
        name = "FGM -> Independent",
        source = constructor_spec(FGMCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),

    # The bivariate FGM endpoints θ = ±1 are ordinary FGM laws, not
    # Fréchet bounds, so they deliberately have no reduction-graph entries.

    (
        name = "FGM multivariate zero parameters -> Independent",
        source = constructor_spec(FGMCopula, 3, zeros(4)),
        target = constructor_spec(IndependentCopula, 3),
    ),

    (
        name = "Plackett -> Independent",
        source = constructor_spec(PlackettCopula, 2, 1.0),
        target = constructor_spec(IndependentCopula, 2),
    ),

    (
        name = "Plackett lower endpoint",
        source = constructor_spec(PlackettCopula, 2, 0.0),
        target = constructor_spec(WCopula, 2),
    ),

    (
        name = "Plackett upper endpoint",
        source = constructor_spec(PlackettCopula, 2, Inf),
        target = constructor_spec(MCopula, 2),
    ),

    (
        name = "Raftery -> Independent",
        source = constructor_spec(RafteryCopula, 2, 0.0),
        target = constructor_spec(IndependentCopula, 2),
    ),

    (
        name = "Raftery -> M",
        source = constructor_spec(RafteryCopula, 2, 1.0),
        target = constructor_spec(MCopula, 2),
    ),
]


function test_boundary_equivalence(source, target; atol = 1e-10, rtol = 1e-10)
    Base.@nospecialize source target
    @test length(source) == length(target)

    d = length(source)

    points = (
        fill(0.25, d),
        fill(0.5, d),
        fill(0.75, d),
        collect(range(0.2, 0.8; length=d)),
    )

    for u in points
        @test cdf(source, u) ≈ cdf(target, u) atol=atol rtol=rtol
    end

    # Density is only meaningful when both expose the same absolutely
    # continuous interface at the considered boundary.
    source_abscont = Copulas.copula_measure_style(source) isa Copulas.AbsolutelyContinuousMeasure
    target_abscont = Copulas.copula_measure_style(target) isa Copulas.AbsolutelyContinuousMeasure

    if source_abscont && target_abscont
        for u in points
            @test pdf(source, u) ≈ pdf(target, u) atol=atol rtol=rtol
            @test logpdf(source, u) ≈ logpdf(target, u) atol=atol rtol=rtol
        end
    end

    return nothing
end


function test_constructor_reduction_graph()
    @testset "constructor boundary equivalence" begin
        for edge in CONSTRUCTOR_REDUCTIONS
            source = build_typed(edge.source)
            target = build_typed(edge.target)
            @test source isa constructor_type(edge.source)
            @test target isa constructor_type(edge.target)
            test_boundary_equivalence(source, target)
        end
    end
    return nothing
end
