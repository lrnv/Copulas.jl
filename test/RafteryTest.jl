@testitem "RafteryCopula Constructor" begin
    for d in [2,3,4]
        @test isa(RafteryCopula(d,0.0), IndependentCopula)
        @test isa(RafteryCopula(d,1.0), MCopula)
    end    
    @test_throws ArgumentError RafteryCopula(1,0.5)
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
end

@testset "RafteryCopula PDF" begin
    
    for d in [2, 3, 4]
        F = RafteryCopula(d, 0.5)

        # Test PDF with some random values
        u = rand(d)
        pdf_value = pdf(F, u)
        @test pdf_value >= 0
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
