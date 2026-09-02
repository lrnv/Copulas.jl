# Routing obligation: methods selected by `which` do not reveal value-,
# dimension-, or representation-dependent branches inside their bodies. These
# focused tests exercise such public branches without repeating the full
# per-family contract.
@testset verbose=true "non-dispatch behavioural branches" begin
    @testset "beta by dimension" begin
        C2 = FGMCopula{2}(0.4)
        @test Copulas.β(C2) ≈ 4cdf(C2, [0.5, 0.5]) - 1

        C3 = FGMCopula{3}([0.0, 0.0, 0.0, 0.4])
        u = fill(0.5, 3)
        survival = SurvivalCopula(C3, (1, 2, 3))
        expected = (4cdf(C3, u) + cdf(survival, u) - 1) / 3
        @test Copulas.β(C3) ≈ expected
    end

    @testset "Frank parameter domain by dimension" begin
        @test params(FrankCopula{2}(-2.0)).θ == -2.0
        @test params(FrankCopula{3}(2.0)).θ == 2.0
        @test_throws AssertionError FrankCopula{3}(-2.0)
    end

    @testset "Clayton negative bivariate domain" begin
        @test ClaytonCopula{2}(-0.7) isa ClaytonCopula{2}
    end

    @testset "independent conditioning output dimension" begin
        @test condition(IndependentCopula{2}(), 1, 0.4) isa Copulas.NoDistortion
        @test condition(IndependentCopula{3}(), 1, 0.4) isa IndependentCopula{2}
    end

    @testset "Survival conditional flip remapping" begin
        C = SurvivalCopula{4}(ClaytonCopula{4}(2.0), (2, 4))
        conditioned = condition(C, (1, 3), (0.25, 0.75))
        @test conditioned.C isa SurvivalCopula{2}
        @test 0.0 <= cdf(conditioned, [0.4, 0.6]) <= 1.0
    end

    @testset "elliptical EV representation by dimension" begin
        # These kernels are expensive.  Their bivariate and multivariate
        # representatives have already populated the proof ledger, so this
        # branch registry verifies that both representations are linked rather
        # than executing the same numerical identities a second time.
        fixtures = filter(ROUTING_COPULA_FIXTURES) do fixture
            fixture.copula isa Union{HuslerReissCopula,tEVCopula}
        end
        for fixture in fixtures
            case, C = fixture.case, fixture.copula
            key = dispatch_route_key(:logpdf, C)
            @test key in keys(PROVEN_DISPATCH_ROUTES[:logpdf])
        end
    end

    @testset "Gumbel--Barnett dimension-dependent validity" begin
        @test GumbelBarnettCopula{2}(0.5) isa GumbelBarnettCopula{2}
        @test GumbelBarnettCopula{3}(0.3) isa GumbelBarnettCopula{3}
        @test_throws AssertionError GumbelBarnettCopula{3}(0.5)
        @test GumbelBarnettCopula{4}(0.2) isa GumbelBarnettCopula{4}
        @test_throws AssertionError GumbelBarnettCopula{4}(0.3)
    end

    @testset "Galambos dependence-inverse boundaries" begin
        @test Copulas.β⁻¹(GalambosCopula, -0.1) == 0.0
        @test Copulas.β⁻¹(GalambosCopula, 0.0) == 0.0
        @test Copulas.β⁻¹(GalambosCopula, 1.0) == Inf
        @test Copulas.λᵤ⁻¹(GalambosCopula, 0.0) == 0.0
        @test Copulas.λᵤ⁻¹(GalambosCopula, 1.0) == Inf
    end

    @testset "parameter-dependent distortion quantile regimes" begin
        # `which` inventories one method per concrete distortion, but cannot
        # see value branches within that method.  Compare every such regime to
        # the generic inversion without repeating the full distortion contract.
        cases = Any[]
        for case in ALL_COPULA_CASES
            isnothing(case.conditional_at) && continue
            C = case.build()
            is_absolutely_continuous(C) || continue
            j, value = case.conditional_at
            push!(cases, condition(C, j, value))
        end
        for D in cases
            p = 0.63
            generic = invoke(quantile, Tuple{Copulas.Distortion,Real}, D, p)
            @test quantile(D, p) ≈ generic atol=2e-8 rtol=2e-8
        end
        @test quantile(Copulas.PlackettDistortion(1.0, Int8(1), 0.4), 0.37) ≈ 0.37
    end

    @testset "Liouville all-one Dirichlet reduction" begin
        G = Copulas.ClaytonGenerator(1.0)
        reduced = LiouvilleCopula{2}(G, (1.0, 1.0))
        native = ArchimedeanCopula{2}(G)
        @test typeof(reduced) == typeof(native)
        @test reduced.G === G
        @test cdf(reduced, [0.35, 0.7]) == cdf(native, [0.35, 0.7])
    end

    @testset "nested singleton subtree collapse" begin
        inner = NestedArchimedeanCopula(Copulas.GumbelGenerator(2.0);
            leaves=[1], children=[GumbelCopula{2}(4.0)])
        deep = NestedArchimedeanCopula(Copulas.ClaytonGenerator(1.5);
            leaves=[1], children=[inner])
        collapsed = subsetdims(deep, (1, 3))
        native = ClaytonCopula{2}(1.5)
        @test typeof(collapsed) == typeof(native)
        @test cdf(collapsed, [0.4, 0.7]) == cdf(native, [0.4, 0.7])
    end

    @testset "extremal-t fitting bounds by dimension" begin
        for (d, lower) in ((2, -1.0), (3, -0.5))
            CT = typeof(tEVCopula{d}(4.0, 0.2))
            bounded = (; ν=4.0, ρ=0.2)
            unbound = Copulas._unbound_params(CT, d, bounded)
            restored = Copulas._rebound_params(CT, d, unbound)
            @test restored.ν ≈ bounded.ν
            @test restored.ρ ≈ bounded.ρ
            @test lower < Copulas._rebound_params(CT, d, [0.0, -100.0]).ρ < 1
            @test lower < Copulas._rebound_params(CT, d, [0.0, 100.0]).ρ < 1
        end
    end

    @testset "Williamson inversion parameter branches" begin
        @test Copulas.𝒲₋₁(Copulas.AMHGenerator(0.5), 2) isa
              Copulas.WilliamsonFromFrailty
        @test !(Copulas.𝒲₋₁(Copulas.AMHGenerator(-0.5), 2) isa
                Copulas.WilliamsonFromFrailty)
        @test Copulas.𝒲₋₁(Copulas.FrankGenerator(2.0), 2) isa
              Copulas.WilliamsonFromFrailty
        @test !(Copulas.𝒲₋₁(Copulas.FrankGenerator(-2.0), 2) isa
                Copulas.WilliamsonFromFrailty)

        @test Copulas.𝒲₋₁(Copulas.ClaytonGenerator(1.0), 1.5) isa
              Distributions.ContinuousUnivariateDistribution
        @test Copulas.𝒲₋₁(Copulas.ClaytonGenerator(-0.25), 2) isa
              Copulas.ClaytonWilliamsonDistribution
        @test Copulas.𝒲₋₁(Copulas.ClaytonGenerator(-0.25), 1.5) isa
              Copulas.WilliamsonBetaProduct
    end

end
