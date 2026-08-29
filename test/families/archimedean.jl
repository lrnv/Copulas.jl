# Family-regression layer: Archimedean and Williamson reference values,
# parameter boundaries, and numerical corner cases.

@testset "Boundary test for bivariate Joe, Gumbel and Frank" begin
    θ = 1.1
    C = JoeCopula{2}(θ)

    # Joe copula is zero on all borders and corners of the hypercube.
    # so as soon as there is a zero or a one it should be zero.
    us = (0.0, 0.2, 0.5, 0.8, 1.0)
    for u in us
        @test pdf(C, [0, u]) == 0
        @test pdf(C, [u, 0]) == 0
        @test pdf(C, [1, u]) == 0
        @test pdf(C, [u, 1]) == 0
    end

    G = GumbelCopula{2}(2.5)
    @test pdf(G, [0.1,0.0]) == 0.0
    @test pdf(G, [0.0,0.1]) == 0.0
    @test pdf(G, [0.0,0.0]) == 0.0
    
    # Issue 247
    @test pdf(FrankCopula{2}(2.5), [1,1]*eps()) ≈ 2.723563724584597
    @test pdf(FrankCopula{2}(-2.5), [1,1]*eps()) ≈ 0.22356372458463078
    @test pdf(FrankCopula{2}(-2.5), [1,1]*0.0) == 0.0
    @test pdf(FrankCopula{2}(2.5), [1,1]*0.0) == 0.0
    @test isapprox(pdf(SklarDist(FrankCopula{2}(-2.5),(Normal(-2.,1),Normal(-0.3,0.1))), [2.,-2.]), 0.0, atol=eps())

end

@testset "Fix values of bivariate ClaytonCopula: τ, cdf, pdf and contructor" begin
    # Fix a few cdf and pdf values:
    x = [0:0.25:1;]
    y = x
    cdf1 = [0.0, 0.1796053020267749, 0.37796447300922725, 0.6255432421712244, 1.0]
    cdf2 = [0.0, 0.0, 0.17157287525381, 0.5358983848622453, 1.0]
    pdf1 = [0.0, 2.2965556205046926, 1.481003649342278, 1.614508582188617, 0.0]
    pdf2 = [0.0, 0.0, 1.0, 2 / 3, 0.0]
    for i in 1:5
        @test cdf(ClaytonCopula{2}(2),[x[i],y[i]]) ≈ cdf1[i]
        @test cdf(ClaytonCopula{2}(-0.5),[x[i],y[i]]) ≈ cdf2[i]
        @test pdf(ClaytonCopula{2}(2),[x[i],y[i]]) ≈ pdf1[i]
        @test pdf(ClaytonCopula{2}(-0.5),[x[i],y[i]]) ≈ pdf2[i]
    end

    # Fix a few tau values:
    @test Copulas.τ(ClaytonCopula{2}(-0.5)) == -1 / 3
    @test Copulas.τ(ClaytonCopula{2}(2)) == 0.5
    @test Copulas.τ(ClaytonCopula{2}(10)) == 10 / 12

    # Interior negative dependence remains a family-specific constructor case;
    # all boundary reductions live in the behavioural-branch ledger.
    @test isa(ClaytonCopula{2}(-0.7), ClaytonCopula)
end


@testset "Archimedean - Fix Kendall and Spearman correlation" begin
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

    @test Copulas.ρ(ClaytonCopula{2}(3.)) ≈ 0.78645 atol=1.0e-4
    @test Copulas.ρ(ClaytonCopula{2}(0.001)) ≈ 0. atol=1.0e-2
    @test Copulas.ρ(GumbelCopula{2}(3.)) ≈ 0.8489 atol=1.0e-4

    @test Copulas.ρ⁻¹(ClaytonCopula, 1/3) ≈ 0.58754 atol=1.0e-5
    @test Copulas.ρ⁻¹(ClaytonCopula, 0.01) ≈ 0. atol=1.0e-1
    @test Copulas.ρ⁻¹(ClaytonCopula, -0.4668) ≈ -.5 atol=1.0e-3
    @test Copulas.ρ⁻¹(ClaytonCopula, 1.0) == Inf

    @test Copulas.ρ⁻¹(GumbelCopula, 0.5) ≈ 1.5410704204332681
    ρweak = 1.0e-4
    θweak = Copulas.ρ⁻¹(GumbelCopula, ρweak)
    @test 1 < θweak < 1.01
    @test Copulas.ρ(GumbelCopula{2}(θweak)) ≈ ρweak atol=1.0e-7

    @test Copulas.ρ⁻¹(FrankCopula, 1/3) ≈ 2.116497 atol=1.0e-5
    @test Copulas.ρ⁻¹(FrankCopula, -0.5572) ≈ -4. atol=1.0e-3

    @test Copulas.ρ⁻¹(AMHCopula, 0.2) ≈ 0.5168580913147318
    @test Copulas.ρ⁻¹(AMHCopula, 0.) ≈ 0. atol=1.0e-4
    @test Copulas.ρ⁻¹(AMHCopula, 0.49) ≈ 1 atol=1.0e-4
    @test Copulas.ρ⁻¹(AMHCopula, -0.273) ≈ -1 atol=1.0e-4
    @test Copulas.ρ⁻¹(AMHCopula, -0.2246) ≈ -0.8 atol=1.0e-3
end

@testset "Fix clayton conditionals" begin 

dist = condition(ClaytonCopula{2}(7.3), 2, 0.6)
a,b,c = cdf(dist, [0.2, 0.5, 0.8])

@test a ≈ 0.00010958096560576897
@test b ≈ 0.16963161864932144
@test c ≈ 0.8987566352893012

dist = condition(ClaytonCopula{3}(7.3), 3, 0.6951919277176142)
d = cdf(dist, [0.2, 0.3]) 
@test d ≈ 3.0484941754695964e-5

e = cdf(dist.C, [0.2, 0.3])
@test e ≈ 0.13034531809769517

end
