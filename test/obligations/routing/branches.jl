# Routing obligation: methods selected by `which` do not reveal value-,
# dimension-, or representation-dependent branches inside their bodies.  This
# focused registry exercises those public branches without repeating the full
# per-family contract.
const BEHAVIOURAL_BRANCHES = (
    :beta_bivariate, :beta_multivariate,
    :frank_negative_bivariate, :frank_positive_multivariate,
    :fgm_independence_boundary, :fgm_frechet_boundaries,
    :independent_scalar_condition, :independent_copula_condition,
    :husler_reiss_bivariate, :husler_reiss_multivariate,
    :tev_bivariate, :tev_multivariate,
    :gumbel_barnett_dimension_bounds,
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

    @testset "independent conditioning output dimension" begin
        @test condition(IndependentCopula{2}(), 1, 0.4) isa Uniform
        @test condition(IndependentCopula{3}(), 1, 0.4) isa IndependentCopula{2}
        prove_branches!(:independent_scalar_condition,
                        :independent_copula_condition)
    end

    @testset "elliptical EV representation by dimension" begin
        # These kernels are expensive.  Their bivariate and multivariate
        # representatives have already populated the proof ledger, so this
        # branch registry verifies that both representations are linked rather
        # than executing the same numerical identities a second time.
        names = ("Husler--Reiss bivariate", "Husler--Reiss",
                 "t-EV", "t-EV multivariate")
        for name in names
            case = only(filter(c -> c.name == name, ROUTING_COPULA_CASES))
            C = case.build()
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

    @test PROVEN_BEHAVIOURAL_BRANCHES == Set(BEHAVIOURAL_BRANCHES)
end
