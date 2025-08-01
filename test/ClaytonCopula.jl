@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,-0.7)) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,-rand(M.rng))) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(2,7)) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(3,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(3,-rand(M.rng)/2)) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(4,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(4,-rand(M.rng)/3)) end
@testitem "Generic" tags=[:ClaytonCopula] setup=[M] begin M.check(ClaytonCopula(4,7.)) end

@testitem "Fix values of bivariate ClaytonCopula: τ, cdf, pdf and contructor" begin
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