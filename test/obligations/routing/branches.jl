# Routing obligation: methods selected by `which` do not reveal value-,
# dimension-, or representation-dependent branches inside their bodies.  This
# focused registry exercises those public branches without repeating the full
# per-family contract.
const BEHAVIOURAL_BRANCHES = (
    :beta_bivariate, :beta_multivariate,
    :frank_negative_bivariate, :frank_positive_multivariate,
    :fgm_independence_boundary, :fgm_frechet_boundaries,
    :generator_boundary_reductions, :misc_copula_boundary_reductions,
    :tail_boundary_reductions,
    :independent_scalar_condition, :independent_copula_condition,
    :subsetting_full_permutation_generic,
    :subsetting_full_permutation_elliptical,
    :husler_reiss_bivariate, :husler_reiss_multivariate,
    :tev_bivariate, :tev_multivariate,
    :tev_fitting_bivariate_bounds, :tev_fitting_multivariate_bounds,
    :gumbel_barnett_dimension_bounds,
    :distortion_quantile_parameter_regimes,
    :amh_frailty, :amh_generic_williamson,
    :frank_frailty, :frank_generic_williamson,
    :clayton_positive_real_order, :clayton_negative_integer_order,
    :clayton_negative_real_order,
)
const PROVEN_BEHAVIOURAL_BRANCHES = Set{Symbol}()
prove_branches!(branches...) = union!(PROVEN_BEHAVIOURAL_BRANCHES, branches)

@testset verbose=true "non-dispatch behavioural branches" begin
    @test allunique(BEHAVIOURAL_BRANCHES)

    @testset "beta by dimension" begin
        C2 = FGMCopula{2}(0.4)
        @test Copulas.β(C2) ≈ 4cdf(C2, [0.5, 0.5]) - 1

        C3 = FGMCopula{3}([0.0, 0.0, 0.0, 0.4])
        u = fill(0.5, 3)
        survival = SurvivalCopula(C3, (1, 2, 3))
        expected = (4cdf(C3, u) + cdf(survival, u) - 1) / 3
        @test Copulas.β(C3) ≈ expected
        prove_branches!(:beta_bivariate, :beta_multivariate)
    end

    @testset "Frank parameter domain by dimension" begin
        @test params(FrankCopula{2}(-2.0)).θ == -2.0
        @test params(FrankCopula{3}(2.0)).θ == 2.0
        @test_throws AssertionError FrankCopula{3}(-2.0)
        prove_branches!(:frank_negative_bivariate, :frank_positive_multivariate)
    end

    @testset "FGM value-dependent reductions" begin
        @test FGMCopula{2}(0.0) isa IndependentCopula{2}
        @test FGMCopula{2}(1.0) isa MCopula{2}
        @test FGMCopula{2}(-1.0) isa WCopula{2}
        @test FGMCopula{3}([0.0, 0.0, 0.0, 0.4]) isa FGMCopula{3}
        prove_branches!(:fgm_independence_boundary, :fgm_frechet_boundaries)
    end

    @testset "public constructor boundary reductions" begin
        generator_reductions = (
            (Copulas.AMHGenerator(0.0), Copulas.IndependentGenerator),
            (Copulas.ClaytonGenerator(-1.0), Copulas.WGenerator),
            (Copulas.ClaytonGenerator(0.0), Copulas.IndependentGenerator),
            (Copulas.ClaytonGenerator(Inf), Copulas.MGenerator),
            (Copulas.FrankGenerator(-Inf), Copulas.WGenerator),
            (Copulas.FrankGenerator(0.0), Copulas.IndependentGenerator),
            (Copulas.FrankGenerator(Inf), Copulas.MGenerator),
            (Copulas.GumbelBarnettGenerator(0.0), Copulas.IndependentGenerator),
            (Copulas.GumbelGenerator(1.0), Copulas.IndependentGenerator),
            (Copulas.GumbelGenerator(Inf), Copulas.MGenerator),
            (Copulas.InvGaussianGenerator(0.0), Copulas.IndependentGenerator),
            (Copulas.JoeGenerator(1.0), Copulas.IndependentGenerator),
            (Copulas.JoeGenerator(Inf), Copulas.MGenerator),
        )
        for (value, expected) in generator_reductions
            @test value isa expected
        end
        prove_branches!(:generator_boundary_reductions)

        copula_reductions = (
            (GaussianCopula{3}(0.0), IndependentCopula{3}),
            (PlackettCopula{2}(0.0), MCopula{2}),
            (PlackettCopula{2}(1.0), IndependentCopula{2}),
            (PlackettCopula{2}(Inf), WCopula{2}),
            (RafteryCopula{3}(0.0), IndependentCopula{3}),
            (RafteryCopula{3}(1.0), MCopula{3}),
        )
        for (value, expected) in copula_reductions
            @test value isa expected
        end
        prove_branches!(:misc_copula_boundary_reductions)

        tail_reductions = (
            (Copulas.CuadrasAugeTail(0.0), Copulas.NoTail),
            (Copulas.CuadrasAugeTail(1.0), Copulas.MTail),
            (Copulas.GalambosTail(0.0), Copulas.NoTail),
            (Copulas.GalambosTail(Inf), Copulas.MTail),
            (Copulas.HuslerReissTail(0.0), Copulas.NoTail),
            (Copulas.HuslerReissTail(Inf), Copulas.MTail),
            (Copulas.HuslerReissTail(zeros(3, 3)), Copulas.MTail),
            (Copulas.LogTail(1.0), Copulas.NoTail),
            (Copulas.LogTail(Inf), Copulas.MTail),
            (Copulas.MixedTail(0.0), Copulas.NoTail),
            (Copulas.tEVTail(4.0, 1.0), Copulas.MTail),
            (Copulas.tEVTail(4.0, ones(3, 3)), Copulas.MTail),
            (Copulas.AsymLogTail(1.0, 0.4, 0.6), Copulas.NoTail),
            (Copulas.AsymLogTail(1.5, 1.0, 1.0), Copulas.LogTail),
            (Copulas.AsymMixedTail(0.0, 0.0), Copulas.NoTail),
            (Copulas.AsymMixedTail(0.3, 0.0), Copulas.MixedTail),
            (Copulas.AsymGalambosTail(1.5, [0.0, 0.0]), Copulas.NoTail),
            (Copulas.AsymGalambosTail(1.5, [1.0, 1.0]), Copulas.GalambosTail),
            (Copulas.AsymGalambosTail(2, [0.7],
                [[1.0], [1.0], [0.0, 0.0]]), Copulas.NoTail),
            (Copulas.AsymGalambosTail(2, [0.7],
                [[0.0], [0.0], [1.0, 1.0]]), Copulas.GalambosTail),
            (Copulas.TawnTail(1.0, [0.4, 0.6]), Copulas.NoTail),
            (Copulas.TawnTail(1.5, [1.0, 1.0]), Copulas.LogTail),
            (Copulas.TawnTail(2, [2.0],
                [[1.0], [1.0], [0.0, 0.0]]), Copulas.NoTail),
            (Copulas.TawnTail(2, [2.0],
                [[0.0], [0.0], [1.0, 1.0]]), Copulas.LogTail),
        )
        for (value, expected) in tail_reductions
            @test value isa expected
        end
        prove_branches!(:tail_boundary_reductions)
    end

    @testset "independent conditioning output dimension" begin
        @test condition(IndependentCopula{2}(), 1, 0.4) isa Copulas.NoDistortion
        @test condition(IndependentCopula{3}(), 1, 0.4) isa IndependentCopula{2}
        prove_branches!(:independent_scalar_condition,
                        :independent_copula_condition)
    end

    @testset "full-coordinate subsetting permutations" begin
        function permuted_point(perm, u)
            v = similar(u)
            for (i, j) in enumerate(perm)
                v[j] = u[i]
            end
            return v
        end

        C = ClaytonCopula{3}(2.0)
        perm = (2, 3, 1)
        S = subsetdims(C, perm)
        u = [0.31, 0.57, 0.79]
        @test cdf(S, u) ≈ cdf(C, permuted_point(perm, u)) atol=1e-8
        @test logpdf(S, u) ≈ logpdf(C, permuted_point(perm, u)) atol=1e-8
        prove_branches!(:subsetting_full_permutation_generic)

        Σ = [1.0 0.6 0.2; 0.6 1.0 0.5; 0.2 0.5 1.0]
        G = GaussianCopula{3}(Σ)
        permuted = subsetdims(G, perm)
        @test permuted.Σ ≈ Σ[collect(perm), collect(perm)]
        @test logpdf(permuted, u) ≈
              logpdf(G, permuted_point(perm, u)) atol=1e-8
        prove_branches!(:subsetting_full_permutation_elliptical)
    end

    @testset "elliptical EV representation by dimension" begin
        # These kernels are expensive.  Their bivariate and multivariate
        # representatives have already populated the proof ledger, so this
        # branch registry verifies that both representations are linked rather
        # than executing the same numerical identities a second time.
        names = ("Husler--Reiss bivariate", "Husler--Reiss",
                 "t-EV", "t-EV multivariate")
        for name in names
            fixture = only(filter(x -> x.case.name == name,
                                  ROUTING_COPULA_FIXTURES))
            case, C = fixture.case, fixture.copula
            key = dispatch_route_key(:logpdf, C, case)
            @test key in keys(PROVEN_DISPATCH_ROUTES[:logpdf])
        end
        prove_branches!(:husler_reiss_bivariate, :husler_reiss_multivariate,
                        :tev_bivariate, :tev_multivariate)
    end

    @testset "Gumbel--Barnett dimension-dependent validity" begin
        @test GumbelBarnettCopula{2}(0.5) isa GumbelBarnettCopula{2}
        @test GumbelBarnettCopula{3}(0.3) isa GumbelBarnettCopula{3}
        @test_throws AssertionError GumbelBarnettCopula{3}(0.5)
        @test GumbelBarnettCopula{4}(0.2) isa GumbelBarnettCopula{4}
        @test_throws AssertionError GumbelBarnettCopula{4}(0.3)
        prove_branches!(:gumbel_barnett_dimension_bounds)
    end

    @testset "parameter-dependent distortion quantile regimes" begin
        # `which` inventories one method per concrete distortion, but cannot
        # see value branches within that method.  Compare every such regime to
        # the generic inversion without repeating the full distortion contract.
        cases = (
            condition(PlackettCopula{2}(0.5), 2, 0.7),
            condition(FrankCopula{2}(-2.0), 1, 0.4),
            condition(AMHCopula{2}(-0.5), 1, 0.4),
            condition(GumbelCopula{2}(1.001), 1, 0.25),
            condition(GumbelCopula{2}(8.0), 1, 0.7),
            condition(LogCopula{2}(1.001), 1, 0.25),
            condition(InvGaussianCopula{2}(0.01), 1, 0.4),
            condition(BB9Copula{2}(1.001, 0.8), 1, 0.4),
            condition(GumbelBarnettCopula{2}(0.01), 1, 0.3),
            condition(GumbelBarnettCopula{2}(0.8), 1, 0.7),
        )
        for D in cases
            p = 0.63
            generic = invoke(quantile, Tuple{Copulas.Distortion,Real}, D, p)
            @test quantile(D, p) ≈ generic atol=2e-8 rtol=2e-8
        end
        @test quantile(Copulas.PlackettDistortion(1.0, Int8(1), 0.4), 0.37) ≈ 0.37
        prove_branches!(:distortion_quantile_parameter_regimes)
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
        prove_branches!(:tev_fitting_bivariate_bounds,
                        :tev_fitting_multivariate_bounds)
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
        prove_branches!(:amh_frailty, :amh_generic_williamson,
                        :frank_frailty, :frank_generic_williamson,
                        :clayton_positive_real_order,
                        :clayton_negative_integer_order,
                        :clayton_negative_real_order)
    end

    @test PROVEN_BEHAVIOURAL_BRANCHES == Set(BEHAVIOURAL_BRANCHES)
end
