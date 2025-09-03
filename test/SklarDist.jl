@testitem "SklarDist fitting" tags=[:SklarDist] begin
    
    using Distributions
    using Random
    using StableRNGs
    rng = StableRNG(123)
    MyD = SklarDist(ClaytonCopula(3,7),(LogNormal(),Pareto(),Beta()))
    u = rand(rng,MyD,1000)
    rand!(rng, MyD,u)
    fit(SklarDist{ClaytonCopula,Tuple{LogNormal,Pareto,Beta}},u)
    fit(SklarDist{GaussianCopula,Tuple{LogNormal,Pareto,Beta}},u)
    @test 1==1
    # loglikelyhood(MyD,u)
end

@testitem "SklarDist Rosenblatt" tags=[:SklarDist] begin
    using StatsBase
    using Distributions
    using Random
    using StableRNGs
    rng = StableRNG(123) 
    
    for D in (
        SklarDist(ClaytonCopula(3,7),(LogNormal(),Pareto(),Beta())),
        SklarDist(GumbelCopula(2,7),(LogNormal(),Pareto())),
        SklarDist(GaussianCopula([1 0.5; 0.5 1]),(Pareto(),Normal())),
    )
        
        d = length(D)
        u = rand(rng, D, 1000)
        v = rosenblatt(D, u)
        w = inverse_rosenblatt(D, v)
        @test u ≈ w
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(v[i, :], v[j, :]) ≈ 0.0 atol = 0.1
            end
        end
    end
end
