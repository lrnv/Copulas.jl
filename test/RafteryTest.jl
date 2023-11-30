@testitem "RafteryCopula Constructor" begin
    for d in [2,3,4]
        @test isa(RafteryCopula(d,0.0), IndependentCopula)
        @test isa(RafteryCopula(d,1.0), MCopula)
    end    
    @test_throws ArgumentError RafteryCopula(3,-1.5)
    @test_throws ArgumentError RafteryCopula(2, 2.6)
end

@testset "RafteryCopula CDF" begin
    
    for d in [2, 3, 4]
        F = RafteryCopula(d, 0.5)

        # Test CDF with some random values
        u = rand(d)
        cdf_value = cdf(F, u)
        @test cdf_value >= 0 && cdf_value <= 1
    end
    examples = [
        ([0.2, 0.5], [1.199432, 1e-5], 0.8),
        ([0.3, 0.8], [0.2817, 1e-5], 0.5),
        ([0.1, 0.2, 0.3], [0.08325884, 1e-5], 0.5),
        ([0.4, 0.8, 0.2], [0.120415, 1e-5], 0.1),
    ]
        
    for (u, expected, θ) in examples
        copula = RafteryCopula(length(u), θ)
        @test cdf(copula, u) ≈ expected[1] atol=expected[2]
    end    
end

@testset "RafteryCopula PDF" begin
    
    for d in [2, 3, 4]
        F = RafteryCopula(d, 0.5)

        # Test PDF with some random values
        u = rand(d)
        pdf_value = pdf(F, u)
        @test pdf_value >= 0
    end
    examples = [
        ([0.2, 0.5], [0.114055555, 1e-4], 0.8),
        ([0.3, 0.8], [0.6325, 1e-4], 0.5),
        ([0.1, 0.2, 0.3], [1.9945086, 1e-4], 0.5),
        ([0.4, 0.8, 0.2], [0.939229, 1e-4], 0.1),
    ]
        
    for (u, expected, θ) in examples
        copula = RafteryCopula(length(u), θ)
        @test pdf(copula, u) ≈ expected[1] atol=expected[2]
    end
end


@testitem "RafteryCopula Sampling" begin
    using StableRNGs
    rng = StableRNG(123)
    n_samples = 100
    F = RafteryCopula(3,0.5)
    samples = rand(rng,F, n_samples)
    @test size(samples) == (3, n_samples)
end
