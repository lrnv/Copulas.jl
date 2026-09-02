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
        source = () -> AMHCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = AMHCopula{2}
    ),

    (
        name = "Clayton -> W",
        source = () -> ClaytonCopula{2}(-1.0),
        target = () -> WCopula{2}(),
        expected_family = ClaytonCopula{2}
    ),
    (
        name = "Clayton -> Independent",
        source = () -> ClaytonCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = ClaytonCopula{2}
    ),
    (
        name = "Clayton -> M",
        source = () -> ClaytonCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = ClaytonCopula{2}
    ),

    (
        name = "Frank -> W",
        source = () -> FrankCopula{2}(-Inf),
        target = () -> WCopula{2}(),
        expected_family = FrankCopula{2}
    ),
    (
        name = "Frank -> Independent",
        source = () -> FrankCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = FrankCopula{2}
    ),
    (
        name = "Frank -> M",
        source = () -> FrankCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = FrankCopula{2}
    ),

    (
        name = "Gumbel -> Independent",
        source = () -> GumbelCopula{2}(1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = GumbelCopula{2}
    ),
    (
        name = "Gumbel -> M",
        source = () -> GumbelCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = GumbelCopula{2}
    ),

    (
        name = "Joe -> Independent",
        source = () -> JoeCopula{2}(1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = JoeCopula{2}
    ),
    (
        name = "Joe -> M",
        source = () -> JoeCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = JoeCopula{2}
    ),

    (
        name = "GumbelBarnett -> Independent",
        source = () -> GumbelBarnettCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = GumbelBarnettCopula{2}
    ),

    (
        name = "InvGaussian -> Independent",
        source = () -> InvGaussianCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = InvGaussianCopula{2}
    ),


    # ============================================================
    # BB Archimedean families -> simpler Archimedean families
    # ============================================================

    (
        name = "BB1 -> Clayton",
        source = () -> BB1Copula{2}(1.5, 1.0),
        target = () -> ClaytonCopula{2}(1.5),
        expected_family = BB1Copula{2}
    ),
    (
        name = "BB1 -> M through Clayton",
        source = () -> BB1Copula{2}(Inf, 1.0),
        target = () -> MCopula{2}(),
        expected_family = BB1Copula{2}
    ),

    (
        name = "BB6 -> Joe",
        source = () -> BB6Copula{2}(1.5, 1.0),
        target = () -> JoeCopula{2}(1.5),
        expected_family = BB6Copula{2}
    ),
    (
        name = "BB6 -> Gumbel",
        source = () -> BB6Copula{2}(1.0, 1.5),
        target = () -> GumbelCopula{2}(1.5),
        expected_family = BB6Copula{2}
    ),
    (
        name = "BB6 -> Independent through Joe",
        source = () -> BB6Copula{2}(1.0, 1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = BB6Copula{2}
    ),
    (
        name = "BB6 -> M through Joe",
        source = () -> BB6Copula{2}(Inf, 1.0),
        target = () -> MCopula{2}(),
        expected_family = BB6Copula{2}
    ),
    (
        name = "BB6 -> M through Gumbel",
        source = () -> BB6Copula{2}(1.0, Inf),
        target = () -> MCopula{2}(),
        expected_family = BB6Copula{2}
    ),

    (
        name = "BB7 -> Clayton",
        source = () -> BB7Copula{2}(1.0, 1.5),
        target = () -> ClaytonCopula{2}(1.5),
        expected_family = BB7Copula{2}
    ),
    (
        name = "BB7 -> M through Clayton",
        source = () -> BB7Copula{2}(1.0, Inf),
        target = () -> MCopula{2}(),
        expected_family = BB7Copula{2}
    ),

    (
        name = "BB8 -> Joe",
        source = () -> BB8Copula{2}(1.5, 1.0),
        target = () -> JoeCopula{2}(1.5),
        expected_family = BB8Copula{2}
    ),
    (
        name = "BB8 -> Independent through Joe",
        source = () -> BB8Copula{2}(1.0, 1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = BB8Copula{2}
    ),
    (
        name = "BB8 -> M through Joe",
        source = () -> BB8Copula{2}(Inf, 1.0),
        target = () -> MCopula{2}(),
        expected_family = BB8Copula{2}
    ),

    (
        name = "BB10 -> AMH",
        source = () -> BB10Copula{2}(1.0, 0.5),
        target = () -> AMHCopula{2}(0.5),
        expected_family = BB10Copula{2}
    ),
    (
        name = "BB10 -> Independent through AMH",
        source = () -> BB10Copula{2}(1.0, 0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = BB10Copula{2}
    ),


    # ============================================================
    # Extreme-value tails -> canonical tails/copulas
    # ============================================================

    (
        name = "Galambos -> Independent",
        source = () -> GalambosCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = GalambosCopula{2}
    ),
    (
        name = "Galambos -> M",
        source = () -> GalambosCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = GalambosCopula{2}
    ),

    (
        name = "Log -> Independent",
        source = () -> LogCopula{2}(1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = LogCopula{2}
    ),
    (
        name = "Log -> M",
        source = () -> LogCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = LogCopula{2}
    ),

    (
        name = "HuslerReiss scalar -> Independent",
        source = () -> HuslerReissCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = HuslerReissCopula{2}
    ),
    (
        name = "HuslerReiss scalar -> M",
        source = () -> HuslerReissCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = HuslerReissCopula{2}
    ),
    (
        name = "HuslerReiss zero variogram -> M",
        source = () -> HuslerReissCopula{3}(zeros(3, 3)),
        target = () -> MCopula{3}(),
        expected_family = HuslerReissCopula{3}
    ),

    (
        name = "CuadrasAuge -> Independent",
        source = () -> CuadrasAugeCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = CuadrasAugeCopula{2}
    ),
    (
        name = "CuadrasAuge -> M",
        source = () -> CuadrasAugeCopula{2}(1.0),
        target = () -> MCopula{2}(),
        expected_family = CuadrasAugeCopula{2}
    ),

    (
        name = "Mixed -> Independent",
        source = () -> MixedCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = MixedCopula{2}
    ),


    # ============================================================
    # Asymmetric EV -> simpler EV families
    # ============================================================

    (
        name = "AsymMixed -> Independent",
        source = () -> AsymMixedCopula{2}(0.0, 0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymMixedCopula{2}
    ),
    (
        name = "AsymMixed -> Mixed",
        source = () -> AsymMixedCopula{2}(0.5, 0.0),
        target = () -> MixedCopula{2}(0.5),
        expected_family = AsymMixedCopula{2}
    ),

    (
        name = "AsymLog alpha=1 -> Independent",
        source = () -> AsymLogCopula{2}(1.0, 0.4, 0.6),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymLogCopula{2}
    ),
    (
        name = "AsymLog theta1=0 -> Independent",
        source = () -> AsymLogCopula{2}(1.5, 0.0, 0.6),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymLogCopula{2}
    ),
    (
        name = "AsymLog theta2=0 -> Independent",
        source = () -> AsymLogCopula{2}(1.5, 0.4, 0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymLogCopula{2}
    ),
    (
        name = "AsymLog -> Log",
        source = () -> AsymLogCopula{2}(1.5, 1.0, 1.0),
        target = () -> LogCopula{2}(1.5),
        expected_family = AsymLogCopula{2}
    ),
    (
        name = "AsymLog -> Independent through Log",
        source = () -> AsymLogCopula{2}(1.0, 1.0, 1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymLogCopula{2}
    ),
    (
        name = "AsymLog -> M through Log",
        source = () -> AsymLogCopula{2}(Inf, 1.0, 1.0),
        target = () -> MCopula{2}(),
        expected_family = AsymLogCopula{2}
    ),

    (
        name = "AsymGalambos alpha=0 -> Independent",
        source = () -> AsymGalambosCopula{2}(0.0, 0.4, 0.6),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymGalambosCopula{2}
    ),
    (
        name = "AsymGalambos inactive support -> Independent",
        source = () -> AsymGalambosCopula{2}(1.0, 0.0, 0.6),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymGalambosCopula{2}
    ),
    (
        name = "AsymGalambos -> Galambos",
        source = () -> AsymGalambosCopula{2}(1.5, 1.0, 1.0),
        target = () -> GalambosCopula{2}(1.5),
        expected_family = AsymGalambosCopula{2}
    ),
    (
        name = "AsymGalambos -> Independent through Galambos",
        source = () -> AsymGalambosCopula{2}(0.0, 1.0, 1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = AsymGalambosCopula{2}
    ),
    (
        name = "AsymGalambos -> M through Galambos",
        source = () -> AsymGalambosCopula{2}(Inf, 1.0, 1.0),
        target = () -> MCopula{2}(),
        expected_family = AsymGalambosCopula{2}
    ),


    # ============================================================
    # Tawn
    # ============================================================

    (
        name = "Tawn inactive logistic component -> Independent",
        source = () -> TawnCopula{3}(1.0, [0.6, 0.7, 0.8]),
        target = () -> IndependentCopula{3}(),
        expected_family = TawnCopula{3}
    ),
    (
        name = "Tawn -> Log",
        source = () -> TawnCopula{3}(1.5, ones(3)),
        target = () -> LogCopula{3}(1.5),
        expected_family = TawnCopula{3}
    ),
    (
        name = "Tawn -> M through Log",
        source = () -> TawnCopula{3}(Inf, ones(3)),
        target = () -> MCopula{3}(),
        expected_family = TawnCopula{3}
    ),


    # ============================================================
    # extremal-t
    # ============================================================

    (
        name = "tEV scalar rho=1 -> M",
        source = () -> tEVCopula{2}(4.0, 1.0),
        target = () -> MCopula{2}(),
        expected_family = tEVCopula{2}
    ),
    # (
    #     name = "tEV all-ones correlation -> M",
    #     source = () -> tEVCopula{3}(4.0, ones(3, 3)),
    #     target = () -> MCopula{3}(),
    #     expected_family = tEVCopula{3}
    # ),


    # ============================================================
    # Archimax generic reductions
    # ============================================================

    (
        name = "Archimax NoTail -> Archimedean",
        source = () -> ArchimaxCopula{2}(
            Copulas.ClaytonGenerator(1.5),
            Copulas.NoTail(),
        ),
        target = () -> ClaytonCopula{2}(1.5),
        expected_family = ArchimaxCopula{2}
    ),

    (
        name = "Archimax IndependentGenerator -> ExtremeValue",
        source = () -> ArchimaxCopula{2}(
            Copulas.IndependentGenerator(),
            Copulas.GalambosTail(1.5),
        ),
        target = () -> GalambosCopula{2}(1.5),
        expected_family = ArchimaxCopula{2}
    ),


    # ============================================================
    # BB4 = Archimax(Clayton, Galambos)
    # ============================================================

    (
        name = "BB4 -> Clayton",
        source = () -> BB4Copula{2}(1.5, 0.0),
        target = () -> ClaytonCopula{2}(1.5),
        expected_family = BB4Copula{2}
    ),
    (
        name = "BB4 -> Galambos",
        source = () -> BB4Copula{2}(0.0, 1.5),
        target = () -> GalambosCopula{2}(1.5),
        expected_family = BB4Copula{2}
    ),
    (
        name = "BB4 -> M through Clayton",
        source = () -> BB4Copula{2}(Inf, 0.0),
        target = () -> MCopula{2}(),
        expected_family = BB4Copula{2}
    ),
    (
        name = "BB4 -> M through Galambos",
        source = () -> BB4Copula{2}(0.0, Inf),
        target = () -> MCopula{2}(),
        expected_family = BB4Copula{2}
    ),


    # ============================================================
    # BB5 = Archimax(Gumbel, Galambos)
    # ============================================================

    (
        name = "BB5 -> Gumbel",
        source = () -> BB5Copula{2}(1.5, 0.0),
        target = () -> GumbelCopula{2}(1.5),
        expected_family = BB5Copula{2}
    ),
    (
        name = "BB5 -> Galambos",
        source = () -> BB5Copula{2}(1.0, 1.5),
        target = () -> GalambosCopula{2}(1.5),
        expected_family = BB5Copula{2}
    ),
    (
        name = "BB5 -> M through Gumbel",
        source = () -> BB5Copula{2}(Inf, 0.0),
        target = () -> MCopula{2}(),
        expected_family = BB5Copula{2}
    ),
    (
        name = "BB5 -> M through Galambos",
        source = () -> BB5Copula{2}(1.0, Inf),
        target = () -> MCopula{2}(),
        expected_family = BB5Copula{2}
    ),


    # ============================================================
    # Liouville -> Archimedean
    # ============================================================

    (
        name = "Liouville -> Archimedean/Clayton",
        source = () -> LiouvilleCopula{2}(
            Copulas.ClaytonGenerator(1.5),
            (1.0, 1.0),
        ),
        target = () -> ClaytonCopula{2}(1.5),
        expected_family = LiouvilleCopula{2}
    ),

    (
        name = "Liouville -> Archimedean/Gumbel",
        source = () -> LiouvilleCopula{2}(
            Copulas.GumbelGenerator(1.5),
            (1.0, 1.0),
        ),
        target = () -> GumbelCopula{2}(1.5),
        expected_family = LiouvilleCopula{2}
    ),


    # ============================================================
    # Nested Archimedean -> flat Archimedean
    # ============================================================

    (
        name = "NestedArchimedean no children -> Archimedean",
        source = () -> NestedArchimedeanCopula{3}(
            Copulas.ClaytonGenerator(1.5);
            leaves = [1, 2, 3],
            children = [],
        ),
        target = () -> ClaytonCopula{3}(1.5),
        expected_family = NestedArchimedeanCopula{3}
    ),


    # ============================================================
    # Other value-dependent public constructors from the bestiary
    # ============================================================

    (
        name = "FGM -> Independent",
        source = () -> FGMCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = FGMCopula{2}
    ),

    # The bivariate FGM endpoints θ = ±1 are ordinary FGM laws, not
    # Fréchet bounds, so they deliberately have no reduction-graph entries.

    (
        name = "FGM multivariate zero parameters -> Independent",
        source = () -> FGMCopula{3}(zeros(4)),
        target = () -> IndependentCopula{3}(),
        expected_family = FGMCopula{3}
    ),

    (
        name = "Plackett -> Independent",
        source = () -> PlackettCopula{2}(1.0),
        target = () -> IndependentCopula{2}(),
        expected_family = PlackettCopula{2}
    ),

    (
        name = "Plackett lower endpoint",
        source = () -> PlackettCopula{2}(0.0),
        target = () -> WCopula{2}(),
        expected_family = PlackettCopula{2}
    ),

    (
        name = "Plackett upper endpoint",
        source = () -> PlackettCopula{2}(Inf),
        target = () -> MCopula{2}(),
        expected_family = PlackettCopula{2}
    ),

    (
        name = "Raftery -> Independent",
        source = () -> RafteryCopula{2}(0.0),
        target = () -> IndependentCopula{2}(),
        expected_family = RafteryCopula{2}
    ),

    (
        name = "Raftery -> M",
        source = () -> RafteryCopula{2}(1.0),
        target = () -> MCopula{2}(),
        expected_family = RafteryCopula{2}
    ),
]


function test_boundary_equivalence(source, target; atol = 1e-10, rtol = 1e-10)
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
        for case in CONSTRUCTOR_REDUCTIONS
            source = case.source()
            target = case.target()
            @test source isa case.expected_family
            test_boundary_equivalence(source, target)
        end
    end
    return nothing
end
