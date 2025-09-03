@testitem "SklarDist fitting (IFM API)" begin
    using StatsBase
    using Distributions
    using Random
    using StableRNGs

    rng = StableRNG(123)
    d, n = 3, 1_000

    m_true = (LogNormal(0.0, 0.5), Pareto(1.0, 3.0), Beta(2.0, 5.0))
    C_true = ClaytonCopula(d, 2.0)                 # ajusta si tu ctor es (d, θ)
    S_true = SklarDist(C_true, m_true)

    X = rand(rng, S_true, n)

    S_c = fit(SklarDist, ClaytonCopula, X; margins=(LogNormal, Pareto, Beta), method=:auto)
    @test S_c isa SklarDist
    @test length(S_c) == d

    S_g = fit(SklarDist, GaussianCopula, X; margins=(LogNormal, Pareto, Beta), method=:auto)
    @test S_g isa SklarDist
    @test length(S_g) == d

    S_emp = fit(SklarDist, EmpiricalCopula, X; margins=(LogNormal, Pareto, Beta))
    @test S_emp isa SklarDist
    @test length(S_emp) == d

    for (nm, S) in [("Clayton", S_c), ("Gaussian", S_g), ("Empirical", S_emp)]
        x = X[:, 1]
        @test isfinite(logpdf(S, x))
        c = cdf(S, x)
        @test 0.0 ≤ c ≤ 1.0
        v = similar(x)
        rand!(rng, S, v)
        @test all(isfinite, v)
    end

    @test S_c.m[1] isa LogNormal
    @test S_c.m[2] isa Pareto
    @test S_c.m[3] isa Beta
end


@testitem "SklarDist Rosenblatt" begin
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
