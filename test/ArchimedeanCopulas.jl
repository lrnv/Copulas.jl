

@testset "Boundary test for bivariate Joe, Gumbel and Frank" begin
    # [GenericTests integration]: Yes, valuable. A general "pdf zero on boundaries when defined" property exists for families with known boundary behavior.
    # We can add a predicate + @testif block in GenericTests that exercises boundary-zero conditions when the family declares them.

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

    G = GumbelCopula(2, 2.5)
    @test pdf(G, [0.1,0.0]) == 0.0
    @test pdf(G, [0.0,0.1]) == 0.0
    @test pdf(G, [0.0,0.0]) == 0.0
    
    # Issue 247
    @test pdf(FrankCopula(2, 2.5), [1,1]*eps()) ≈ 2.723563724584597
    @test pdf(FrankCopula(2, -2.5), [1,1]*eps()) ≈ 0.22356372458463078
    @test pdf(FrankCopula(2, -2.5), [1,1]*0.0) == 0.0
    @test pdf(FrankCopula(2, 2.5), [1,1]*0.0) == 0.0
    @test isapprox(pdf(SklarDist(FrankCopula(2,-2.5),(Normal(-2.,1),Normal(-0.3,0.1))), [2.,-2.]), 0.0, atol=eps())

end

@testset "Fix values of bivariate ClaytonCopula: τ, cdf, pdf and contructor" begin
    # [GenericTests integration]: Partially. The numeric regression values (cdf/pdf grids) are very specific but could be folded as a generic
    # "golden samples" check behind a feature flag for select baseline families. τ identities and constructor edge-cases (0, -1, Inf) can be generalized.


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


@testset "Archimedean - Fix Kendall and Spearman correlation" begin
    # [GenericTests integration]: Yes for τ ∘ τ⁻¹; we already added similar checks in GenericTests.
    # The many ρ⁻¹ broken checks are family-specific and currently broken; better to keep here until ρ⁻¹ is implemented robustly.

    Random.seed!(rng,123)

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

    @test Copulas.ρ(ClaytonCopula(2,3.)) ≈ 0.78645 atol=1.0e-4
    @test Copulas.ρ(ClaytonCopula(2,0.001)) ≈ 0. atol=1.0e-2
    @test Copulas.ρ(GumbelCopula(2,3.)) ≈ 0.8489 atol=1.0e-4

    @test Copulas.ρ⁻¹(ClaytonCopula, 1/3) ≈ 0.58754 atol=1.0e-5
    @test Copulas.ρ⁻¹(ClaytonCopula, 0.01) ≈ 0. atol=1.0e-1
    @test Copulas.ρ⁻¹(ClaytonCopula, -0.4668) ≈ -.5 atol=1.0e-3

    @test Copulas.ρ⁻¹(GumbelCopula, 0.5) ≈ 1.5410704204332681
    @test_broken Copulas.ρ⁻¹(GumbelCopula, 0.0001) == 1.

    @test Copulas.ρ⁻¹(FrankCopula, 1/3) ≈ 2.116497 atol=1.0e-5
    @test Copulas.ρ⁻¹(FrankCopula, -0.5572) ≈ -4. atol=1.0e-3

    @test Copulas.ρ⁻¹(AMHCopula, 0.2) ≈ 0.5168580913147318
    @test Copulas.ρ⁻¹(AMHCopula, 0.) ≈ 0. atol=1.0e-4
    @test Copulas.ρ⁻¹(AMHCopula, 0.49) ≈ 1 atol=1.0e-4
    @test Copulas.ρ⁻¹(AMHCopula, -0.273) ≈ -1 atol=1.0e-4
    @test Copulas.ρ⁻¹(AMHCopula, -0.2246) ≈ -0.8 atol=1.0e-3
end

@testset "Testing empirical tail values of certain copula samples" begin
    # [GenericTests integration]: Probably too stochastic and slow for generic; relies on large random samples and fragile tail estimates.
    # Keep as targeted property tests here; if needed, add a lighter tail-coherency smoke test generically.

    Random.seed!(rng,123)

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
    Random.seed!(rng,123)
    x = rand(rng,GumbelCopula(3,2.), 40_000)
    @test_broken tail(x[:,1], x[:,2], "r") ≈ 2-2^(1/2) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "l", 0.00001) ≈ 0.
    @test tail(x[:,1], x[:,3], "l", 0.00001) ≈ 0.

    # Clayton
    Random.seed!(rng,123)
    x = rand(rng,ClaytonCopula(3,1.), 40_000)
    @test_broken tail(x[:,1], x[:,2], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test_broken tail(x[:,1], x[:,3], "l") ≈ 2.0^(-1) atol=1.0e-1
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0

    # AMH
    Random.seed!(rng,123)
    x = rand(rng,AMHCopula(3,0.8), 40_000)
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
    
    # Frank
    Random.seed!(rng,123)
    x = rand(rng,FrankCopula(3,0.8), 40_000)
    @test tail(x[:,1], x[:,2], "l", 0.0001) ≈ 0
    @test tail(x[:,1], x[:,2], "r", 0.0001) ≈ 0
end


@testset "Test of τ ∘ τ⁻¹ = Id" begin
    # [GenericTests integration]: Yes. This is already covered or can be unified inside GenericTests under Archimedean-specific checks.

    Random.seed!(rng,123)

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

@testset "Test of ρ ∘ ρ⁻¹ = Id" begin
    # [GenericTests integration]: Not yet. ρ⁻¹ is not uniformly available/accurate; keep here as broken placeholders until APIs solidify.
    Random.seed!(rng,123)

    inv_works(T,rho) = Copulas.ρ(T(2,Copulas.ρ⁻¹(T,rho))) ≈ rho
    check_rnd(T,m,M,N) = all(inv_works(T, m + (M-m)*u) for u in rand(rng,N))

    # Should be adapted to spearman rho and its inverse when it is possible. 
    @test check_rnd(GumbelCopula,         0,    1,    10)
    @test check_rnd(JoeCopula,            0,    1,    10)
    @test check_rnd(GumbelBarnettCopula, -0.35, 0,    10)
    @test check_rnd(AMHCopula,           -0.18, 0.33, 10)
    @test check_rnd(FrankCopula,         -1,    1,    10)
    @test check_rnd(ClaytonCopula,       -1,    1,    10)
    @test check_rnd(InvGaussianCopula,    0,    log(2),  10)
end


@testset "Fix clayton conditionals" begin 

dist = condition(ClaytonCopula(2, 7.3), 2, 0.6)
a,b,c = cdf(dist, [0.2, 0.5, 0.8])

@test a ≈ 0.00010958096560576897
@test b ≈ 0.16963161864932144
@test c ≈ 0.8987566352893012

dist = condition(ClaytonCopula(3, 7.3), 3, 0.6951919277176142)
d = cdf(dist, [0.2, 0.3]) 
@test d ≈ 3.0484941754695964e-5

e = cdf(dist.C, [0.2, 0.3])
@test d ≈ 0.13034531809769517

end