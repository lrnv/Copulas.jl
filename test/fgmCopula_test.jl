@testitem "FGMCopula Constructor" begin
    @test isa(FGMCopula(2,0.0), IndependentCopula)
    @test_throws ArgumentError FGMCopula(1,0.5)
    @test_throws ArgumentError FGMCopula(3,[-1.5,2.0,3.1,1.2])
    @test_throws ArgumentError FGMCopula(1,[0.8,0.2,0.5,0.4])
end

@testset "FGMCopula CDF" begin
    examples = [
        ([0.1, 0.2, 0.3], [0.0100776123, 1e-4], [0.1,0.2,0.5,0.4]),
        ([0.5, 0.4, 0.3], [0.0830421321, 1e-4], [0.3,0.3,0.3,0.3]),
        ([0.1, 0.1], [0.010023, 1e-4], 0.0),
        ([0.5, 0.4], [0.224013, 1e-4], 0.5),
    ]
    
    for (u, expected) in examples
        copula = FGMCopula(length(u), expected[3])
        @test cdf(copula, u) ≈ expected[1] atol=expected[2]
    end
end

@testset "FGMCopula PDF" begin
    using StableRNGs
    rng = StableRNG(123)
    examples = [
        ([0.1, 0.2, 0.3], [1.308876232, 1e-4], [0.1,0.2,0.5,0.4]),
        ([0.5, 0.4, 0.3], [1.024123232, 1e-4], [0.3,0.3,0.3,0.3]),
        ([0.1, 0.1], [0.01, 1e-4], 0.0),
        ([0.5, 0.4], [1, 1e-4], rand(rng)),
    ]
    
    for (u, expected) in examples
        copula = FGMCopula(length(u), expected[3])
        @test cdf(copula, u) ≈ expected[1] atol=expected[2]
    end
end

@testitem "FGMCopula Sampling" begin
    using StableRNGs
    rng = StableRNG(123)
    n_samples = 100
    F = FGMCopula(3,[0.1,0.2,0.3,0.4])
    samples = rand(rng,F, n_samples)
    @test size(samples) == (3, n_samples)
end