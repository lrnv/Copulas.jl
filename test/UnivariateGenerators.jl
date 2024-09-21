
@testitem "UnivariateGenerators : Test of τ ∘ τ⁻¹ = Id" begin
    using Random
    using InteractiveUtils
    using StableRNGs
    rng = StableRNG(123)

    inv_works(T,tau) = Copulas.τ(T(2,Copulas.τ⁻¹(T,tau))) ≈ tau
    check_rnd(T,min,max,N) = all(inv_works(T,x) for x in min .+ (max-min) .* rand(rng,N))

    @test check_rnd(ClaytonCopula,       -1,    1,    10)
    @test check_rnd(GumbelCopula,         0,    1,    10)
    @test check_rnd(JoeCopula,            0,    1,    10)
    @test check_rnd(GumbelBarnettCopula, -0.35, 0,    10)
    @test check_rnd(AMHCopula,           -0.18, 0.33, 10)
    @test check_rnd(FrankCopula,         -1,    1,    10)
    @test check_rnd(InvGaussianCopula,    0,    1/2,  10)

    # check we did not forget to add a generator: 
    @test length(InteractiveUtils.subtypes(Copulas.UnivariateGenerator)) == 7
end

@testitem "UnivariateGenerators : Test of ρ ∘ ρ⁻¹ = Id" begin
    using Random
    using InteractiveUtils
    using StableRNGs
    rng = StableRNG(123)

    inv_works(T,rho) = Copulas.ρ(T(2,Copulas.ρ⁻¹(T,rho))) ≈ rho
    check_rnd(T,min,max,N) = all(inv_works(T,x) for x in min .+ (max-min) .* rand(rng,N))

    # Should be adapted to spearman rho and its inverse when it is possible. 
    @test_broken check_rnd(ClaytonCopula,       -1,    1,    10)
    @test_broken check_rnd(GumbelCopula,         0,    1,    10)
    @test_broken check_rnd(JoeCopula,            0,    1,    10)
    @test_broken check_rnd(GumbelBarnettCopula, -0.35, 0,    10)
    @test_broken check_rnd(AMHCopula,           -0.18, 0.33, 10)
    @test_broken check_rnd(FrankCopula,         -1,    1,    10)
    @test_broken check_rnd(InvGaussianCopula,    0,    1/2,  10)

    # check we did not forget to add a generator: 
    @test length(InteractiveUtils.subtypes(Copulas.UnivariateGenerator)) == 7
end

@testitem "Archimedeans with UnivariateGenerators" begin
    using StableRNGs
    using Distributions
    using InteractiveUtils
    rng = StableRNG(123)
    Gs = InteractiveUtils.subtypes(Copulas.UnivariateGenerator)
    for d in 2:2 # fitting only in bivariate cases due to issues with negative parameters ? 
        for G in Gs
            fit(ArchimedeanCopula{d,G{Tθ}} where {d, Tθ}, rand(rng,d,100))
        end
    end
    @test true
end

# @testitem "A few tests on bigfloats" begin
#     # using StableRNGs
#     using Random
#     using Distributions
#     using HypothesisTests
#     rng = Random.default_rng() #StableRNG(123) not availiable for bigfloats on old julias.
    
#     for C in (
#         GumbelCopula(3, BigFloat(2.)),
#         ClaytonCopula(2, BigFloat(-.5)),
#         ClaytonCopula(3, BigFloat(2.5)),
#         FrankCopula(2, BigFloat(2.)),
#         AMHCopula(3,BigFloat(.5)),
#         AMHCopula(2,BigFloat(-.3)),
#     )
#         x = rand(rng,C,100)
#         @test_broken typeof(x) == Array{BigFloat,2}
#     end
# end