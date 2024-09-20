
@testitem "constructor" begin
    @test isa(ClaytonCopula(2,0), IndependentCopula)
    @test isa(ClaytonCopula(2,-0.7), ClaytonCopula)
    @test isa(ClaytonCopula(2,-1), WCopula)
    @test isa(ClaytonCopula(2,Inf), MCopula)
end

@testitem "Check that generators inverse are good." begin

    #### This test could be done for many more archimedeans. 
    θ = [2, 10]
    u = [0:0.1:1;]
    for ϑ in θ
        c = ClaytonCopula(2,ϑ)
        for x in u
            @test Copulas.ϕ⁻¹(c,Copulas.ϕ(c,x)) ≈ x
        end
    end
end

@testitem "τ" begin
    @test Copulas.τ(ClaytonCopula(2,-0.5)) == -1 / 3
    @test Copulas.τ(ClaytonCopula(2,2)) == 0.5
    @test Copulas.τ(ClaytonCopula(2,10)) == 10 / 12
end

@testitem "sample" begin
    using StatsBase, Random
    using StableRNGs
    rng = StableRNG(123)
    n = 10^5
    θ = [-0.5, 2, 10]
    
    for ϑ in θ
        c = ClaytonCopula(2,ϑ)
        # if ϑ < 0
        #     @test_broken rand(rng,c,n)
        # else
            u = rand(rng,c, n)
            @test corkendall(u') ≈ [1.0 Copulas.τ(c); Copulas.τ(c) 1.0] atol = 0.01
        # end
    end
end

@testitem "cdf" begin
    using Distributions
    
    x = [0:0.25:1;]
    y = x
    v1 = [0.0, 0.1796053020267749, 0.37796447300922725, 0.6255432421712244, 1.0]
    v2 = [0.0, 0.0, 0.17157287525381, 0.5358983848622453, 1.0]
    for i in 1:5
        @test cdf(ClaytonCopula(2,2),[x[i],y[i]]) ≈ v1[i]
        @test cdf(ClaytonCopula(2,-0.5),[x[i],y[i]]) ≈ v2[i]
    end
end

@testitem "density" begin
    using Distributions
    using Random
    n = 10^6
    θ = [-0.5, 2, 10]
    
    x = [0:0.25:1;]
    y = x
    v1 = [Inf, 2.2965556205046926, 1.481003649342278, 1.614508582188617, 3.0]
    v2 = [Inf, 0.0, 1.0, 2 / 3, 0.5]
    for i in 1:5
        @test pdf(ClaytonCopula(2,2),[x[i],y[i]]) ≈ v1[i]
        @test pdf(ClaytonCopula(2,-0.5),[x[i],y[i]]) ≈ v2[i]
    end  
end


