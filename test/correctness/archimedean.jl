# Correctness obligation: independent Archimedean reference values, support
# boundaries, and conditional numerical anchors.

@testset "Boundary test for bivariate Joe, Gumbel and Frank" begin
    θ = 1.1
    C = JoeCopula{2}(θ)

    # Exercise both coordinate positions and both boundary branches. Interior
    # values along a given border use the same implementation path.
    @test pdf(C, [0.0, 0.5]) == 0
    @test pdf(C, [0.5, 0.0]) == 0
    @test pdf(C, [1.0, 0.5]) == 0
    @test pdf(C, [0.5, 1.0]) == 0

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

@testset "bivariate Clayton CDF/PDF numerical anchors" begin
    # Fix a few cdf and pdf values:
    x = [0:0.25:1;]
    y = x
    cdf1 = [0.0, 0.1796053020267749, 0.37796447300922725, 0.6255432421712244, 1.0]
    cdf2 = [0.0, 0.0, 0.17157287525381, 0.5358983848622453, 1.0]
    pdf1 = [0.0, 2.2965556205046926, 1.481003649342278, 1.614508582188617, 0.0]
    pdf2 = [0.0, 0.0, 1.0, 2 / 3, 0.0]
    # Endpoints are part of the universal copula contract. These three points
    # retain the negative-support cutoff and two distinct interior regimes.
    for i in 2:4
        @test cdf(ClaytonCopula{2}(2),[x[i],y[i]]) ≈ cdf1[i]
        @test cdf(ClaytonCopula{2}(-0.5),[x[i],y[i]]) ≈ cdf2[i]
        @test pdf(ClaytonCopula{2}(2),[x[i],y[i]]) ≈ pdf1[i]
        @test pdf(ClaytonCopula{2}(-0.5),[x[i],y[i]]) ≈ pdf2[i]
    end
end


@testset "Clayton conditional numerical anchors" begin
    distortion = condition(ClaytonCopula{2}(7.3), 2, 0.6)
    @test cdf(distortion, [0.2, 0.5, 0.8]) ≈
          [0.00010958096560576897, 0.16963161864932144, 0.8987566352893012]

    conditional = condition(ClaytonCopula{3}(7.3), 3, 0.6951919277176142)
    @test cdf(conditional, [0.2, 0.3]) ≈ 3.0484941754695964e-5
    @test cdf(conditional.C, [0.2, 0.3]) ≈ 0.13034531809769517
end
