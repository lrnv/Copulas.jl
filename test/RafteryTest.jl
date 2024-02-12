@testitem "RafteryCopula Constructor" begin
    for d in [2,3,4]
        @test isa(RafteryCopula(d,0.0), IndependentCopula)
        @test isa(RafteryCopula(d,1.0), MCopula)
    end    
    @test_throws ArgumentError RafteryCopula(3,-1.5)
    @test_throws ArgumentError RafteryCopula(2, 2.6)
end

@testitem "RafteryCopula CDF" begin
    using StableRNGs
    using Distributions
    rng = StableRNG(123)
    for d in [2, 3, 4]
        F = RafteryCopula(d, 0.5)
        cdf_value = cdf(F, rand(d))
        pdf_value = pdf(F,rand(d))
        @test cdf_value >= 0 && cdf_value <= 1
        @test pdf_value >= 0 
    end

    @test cdf(RafteryCopula(2, 0.8), [0.2, 0.5]) ≈ 0.199432 atol=1e-5
    @test cdf(RafteryCopula(2, 0.5), [0.3, 0.8]) ≈ 0.2817 atol=1e-5
    @test_broken cdf(RafteryCopula(3, 0.5), [0.1, 0.2, 0.3]) ≈ 0.08325884 atol=1e-5
    @test_broken cdf(RafteryCopula(3, 0.1), [0.4, 0.8, 0.2]) ≈ 0.120415 atol=1e-5

    @test pdf(RafteryCopula(2, 0.8), [0.2, 0.5]) ≈ 0.114055555 atol=1e-4
    @test pdf(RafteryCopula(2, 0.5), [0.3, 0.8]) ≈ 0.6325 atol=1e-4
    @test pdf(RafteryCopula(3, 0.5), [0.1, 0.2, 0.3]) ≈ 1.9945086 atol=1e-4
    @test pdf(RafteryCopula(3, 0.1), [0.4, 0.8, 0.2]) ≈ 0.939229 atol=1e-4
end

@testitem "RafteryCopula Sampling" begin
    using StableRNGs
    rng = StableRNG(123)
    n_samples = 100
    F = RafteryCopula(3,0.5)
    samples = rand(rng,F, n_samples)
    @test size(samples) == (3, n_samples)
end
