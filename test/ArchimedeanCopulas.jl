

@testitem "Generic" tags=[:ArchimedeanCopula, :WilliamsonGenerator] setup=[M] begin using Distributions; M.check(ArchimedeanCopula(2,i𝒲(LogNormal(),2))) end

@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(2,-1.0)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(2,-1.0)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(2,-rand(M.rng))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(2,0.7)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(3,-rand(M.rng)*0.1)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(3,0.6)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(3,rand(M.rng))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :AMHCopula] setup=[M] begin M.check(AMHCopula(4,-0.01)) end

@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,-0.7)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,-rand(M.rng))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,7)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(3,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(3,-rand(M.rng)/2)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(4,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(4,-rand(M.rng)/3)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(4,7.)) end

@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(2,0.5)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(2,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(2,1.0)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(2,-5)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(3,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(3,1.0)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(3,12)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(4,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(4,1.0)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(4,150)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(4,30)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(4,37)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(4,6.)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :FrankCopula] setup=[M] begin M.check(FrankCopula(4,6)) end

@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(2,1.0)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(3,0.1)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(3,0.35)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(3,rand(M.rng)*0.38)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(4,0.2)) end

@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(2, 1.2)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(2,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(2,8)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(3,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,20)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,7)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,100)) end

@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,1.0)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(3,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(3,rand(M.rng))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,0.05)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,1.0)) end

@testitem "Generic" tags=[:ArchimedeanCopula, :JoeCopula] setup=[M] begin M.check(JoeCopula(2,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :JoeCopula] setup=[M] begin M.check(JoeCopula(2,3)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :JoeCopula] setup=[M] begin M.check(JoeCopula(2,Inf)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :JoeCopula] setup=[M] begin M.check(JoeCopula(3,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ArchimedeanCopula, :JoeCopula] setup=[M] begin M.check(JoeCopula(3,7)) end
@testitem "Generic" tags=[:ArchimedeanCopula, :JoeCopula] setup=[M] begin M.check(JoeCopula(4,1-log(rand(M.rng)))) end

@testitem "Boundary test for bivariate Joe" tags=[:ArchimedeanCopula, :JoeCopula] begin
    using Distributions
    θ = 1.1
    C = JoeCopula(2, θ)

    # Joe copula is zero on all borders and corners of the hypercube.
    # so as soon as there is a zero or a one it should be zero.
    us = [0,1,rand(10)...]
    for u in us
        @test pdf(C, [0, u]) == 0
        @test pdf(C, [u, 0]) == 0
        @test pdf(C, [1, u]) == 0
        @test pdf(C, [u, 1]) == 0
    end
end

@testitem "Boundary test for bivariate Gumbel" tags=[:ArchimedeanCopula, :GumbelCopula] begin
    using Distributions
    G = GumbelCopula(2, 2.5)
    @test pdf(G, [0.1,0.0]) == 0.0
    @test pdf(G, [0.0,0.1]) == 0.0
    @test pdf(G, [0.0,0.0]) == 0.0
end

@testitem "Fix values of bivariate ClaytonCopula: τ, cdf, pdf and contructor" tags=[:ArchimedeanCopula, :ClaytonCopula] begin
    using Distributions
    using HCubature

    C = ClaytonCopula(2, 2.5)
    @test hcubature(x -> pdf(C, x), zeros(2), ones(2))[1] ≈ 1.0

    # Fix a few cdf and pdf values:
    x = [0:0.25:1;]
    y = x
    cdf1 = [0.0, 0.1796053020267749, 0.37796447300922725, 0.6255432421712244, 1.0]
    cdf2 = [0.0, 0.0, 0.17157287525381, 0.5358983848622453, 1.0]
    pdf1 = [0.0, 2.2965556205046926, 1.481003649342278, 1.614508582188617, 0.0]
    pdf2 = [0.0, 0.0, 1.0, 2 / 3, 0.0]
    for i in 1:5
        @test cdf(ClaytonCopula(2,2),[x[i],y[i]]) ≈ cdf1[i]
        @test cdf(ClaytonCopula(2,-0.5),[x[i],y[i]]) ≈ cdf2[i]
        @test pdf(ClaytonCopula(2,2),[x[i],y[i]]) ≈ pdf1[i]
        @test pdf(ClaytonCopula(2,-0.5),[x[i],y[i]]) ≈ pdf2[i]
    end

    # Fix a few tau values:
    @test Copulas.τ(ClaytonCopula(2,-0.5)) == -1 / 3
    @test Copulas.τ(ClaytonCopula(2,2)) == 0.5
    @test Copulas.τ(ClaytonCopula(2,10)) == 10 / 12

    # Fix constructor behavior:
    @test isa(ClaytonCopula(2,0), IndependentCopula)
    @test isa(ClaytonCopula(2,-0.7), ClaytonCopula)
    @test isa(ClaytonCopula(2,-1), WCopula)
    @test isa(ClaytonCopula(2,Inf), MCopula)
end








@testitem "Archimedean - Fix Kendall correlation" tags=[:ArchimedeanCopula, :ClaytonCopula, :GumbelCopula, :AMHCopula, :FrankCopula] begin
    using Random
    using StableRNGs
    rng = StableRNG(123)

    @test Copulas.Debye(0.5,1) ≈ 0.8819271567906056
    @test Copulas.τ⁻¹(FrankCopula, 0.6) ≈ 7.929642284264058
    @test Copulas.τ⁻¹(GumbelCopula, 0.5) ≈ 2.
    @test Copulas.τ⁻¹(ClaytonCopula, 1/3) ≈ 1.
    @test Copulas.τ⁻¹(AMHCopula, 1/4) ≈ 0.8384520912688538
    @test Copulas.τ⁻¹(AMHCopula, 0.) ≈ 0.
    @test Copulas.τ⁻¹(AMHCopula, 1/3+0.0001) ≈ 1.
    @test Copulas.τ⁻¹(AMHCopula, -2/11) ≈ -1.
    @test Copulas.τ⁻¹(AMHCopula, -0.1505) ≈ -0.8 atol=1.0e-3
    @test Copulas.τ⁻¹(FrankCopula, -0.3881) ≈ -4. atol=1.0e-3
    @test Copulas.τ⁻¹(ClaytonCopula, -1/3) ≈ -.5 atol=1.0e-5
end

@testitem "Archimedeans - Fix Spearman correlation" tags=[:ArchimedeanCopula, :ClaytonCopula, :GumbelCopula, :AMHCopula, :FrankCopula] begin

    @test Copulas.ρ(ClaytonCopula(2,3.)) ≈ 0.78645 atol=1.0e-4
    @test Copulas.ρ(ClaytonCopula(2,0.001)) ≈ 0. atol=1.0e-2
    @test Copulas.ρ(GumbelCopula(2,3.)) ≈ 0.8489 atol=1.0e-4

    @test_broken Copulas.ρ⁻¹(ClaytonCopula, 1/3) ≈ 0.58754 atol=1.0e-5
    @test_broken Copulas.ρ⁻¹(ClaytonCopula, 0.01) ≈ 0. atol=1.0e-1
    @test_broken Copulas.ρ⁻¹(ClaytonCopula, -0.4668) ≈ -.5 atol=1.0e-3

    @test_broken Copulas.ρ⁻¹(GumbelCopula, 0.5) ≈ 1.5410704204332681
    @test_broken Copulas.ρ⁻¹(GumbelCopula, 0.0001) == 1.

    @test_broken Copulas.ρ⁻¹(FrankCopula, 1/3) ≈ 2.116497 atol=1.0e-5
    @test_broken Copulas.ρ⁻¹(FrankCopula, -0.5572) ≈ -4. atol=1.0e-3

    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.2) ≈ 0.5168580913147318
    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.) ≈ 0. atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, 0.49) ≈ 1 atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, -0.273) ≈ -1 atol=1.0e-4
    @test_broken Copulas.ρ⁻¹(AMHCopula, -0.2246) ≈ -0.8 atol=1.0e-3
end

@testitem "Testing empirical tail values of certain copula samples" tags=[:ArchimedeanCopula, :ClaytonCopula, :GumbelCopula, :AMHCopula, :FrankCopula] begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)

    function tail(v1::Vector{T}, v2::Vector{T}, tail::String, α::T = 0.002) where T <: Real
        if tail == "l"
            return sum((v1 .< α) .* (v2 .< α))./(length(v1)*α)
        elseif tail == "r"
            return sum((v1 .> (1-α)) .* (v2 .> (1-α)))./(length(v1)*α)
        end
        0.
    end

    # tail dependencies test
    v1 = vcat(zeros(5), 0.5*ones(5), zeros(5), 0.5*ones(70), ones(5), 0.5*ones(5), ones(5));
    v2 = vcat(zeros(10), 0.5*ones(80), ones(10))
    @test tail(v1, v2,  "l", 0.1) ≈ 0.5
    @test tail(v1, v2, "r", 0.1) ≈ 0.5

    # Gumbel
    rng = StableRNG(123)
    x = rand(rng,GumbelCopula(3,2.), 100_000)
    @test_broken tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
    @test_broken tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
    @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.

    # Clayton
    rng = StableRNG(123)
    x = rand(rng,ClaytonCopula(3,1.), 100_000)
    @test_broken tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test_broken tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0

    # AMH
    rng = StableRNG(123)
    x = rand(rng,AMHCopula(3,0.8), 100_000)
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    
    # Frank
    rng = StableRNG(123)
    x = rand(rng,FrankCopula(3,0.8), 100_000)
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
end


@testitem "Test of τ ∘ τ⁻¹ = Id" tags=[:ArchimedeanCopula, :ClaytonCopula, :GumbelCopula, :AMHCopula, :FrankCopula, :GumbelBarnettCopula, :InvGaussianCopula] begin
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
end

@testitem "Test of ρ ∘ ρ⁻¹ = Id" tags=[:ArchimedeanCopula, :ClaytonCopula, :GumbelCopula, :AMHCopula, :FrankCopula, :GumbelBarnettCopula, :InvGaussianCopula] begin
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
end

@testitem "williamson test" tags=[:ArchimedeanCopula, :WilliamsonCopula] begin
    using Distributions, Random
    using StableRNGs
    rng = StableRNG(12)

    Cops = (
        ArchimedeanCopula(10,i𝒲(Dirac(1),10)),
        ArchimedeanCopula(2,i𝒲(Pareto(1),5)),
        ArchimedeanCopula(2,i𝒲(LogNormal(3),5)),
    )
    for C in Cops
        rand(rng,C,10)
    end
end

@testitem "williamson test" tags=[:ArchimedeanCopula, :WilliamsonCopula] begin
    using Distributions, Random
    using StableRNGs
    rng = StableRNG(12)

    Cops = (
        ArchimedeanCopula(10,i𝒲(MixtureModel([Dirac(1), Dirac(2)]),11)), 
        ArchimedeanCopula(2,i𝒲(Pareto(1),5)),
        ArchimedeanCopula(2,i𝒲(LogNormal(3),5)),
    )
    
    for C in Cops
        u = rand(C, 10)
        v = rosenblatt(C, u)
        w = inverse_rosenblatt(C, v)
        @test u ≈ w
    end
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

