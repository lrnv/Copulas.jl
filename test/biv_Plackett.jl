@testitem "PlackettCopula Constructor" begin
    @test_broken isa(PlackettCopula(1), Independence)
    @test_broken isa(PlackettCopula(Inf),WCopula) # should work in any dimenisons if theta is smaller than the bound.
    @test_broken isa(PlackettCopula(0),MCopula)
    @test_throws ArgumentError PlackettCopula(-0.5)
end

@testitem "PlackettCopula CDF" begin
    u = 0.1:0.18:1
    v = 0.4:0.1:0.9
    l1 = [0.0553778 ,0.1743884 ,0.3166277 ,0.4823228 ,0.6743114 ,0.9 ]
    l2 = [0.02620873, 0.1056116, 0.2349113, 0.4162573, 0.6419255 ,0.9]

    for i in 1:6
        @test cdf(PlackettCopula(2.0), [u[i], v[i]]) ≈ l1[i]
        @test cdf(PlackettCopula(0.5), [u[i], v[i]]) ≈ l2[i]
    end
end

@testitem "PlackettCopula PDF" begin
    u = 0.1:0.18:1
    v = 0.4:0.1:0.9
    l1 = [1.059211 ,1.023291 ,1.038467 ,1.110077 ,1.272959 ,1.652893 ]
    l2 = [0.8446203 ,1.023291, 1.064891, 0.9360171, 0.7346612, 0.5540166]

    for i in 1:6
        @test pdf(PlackettCopula(2.0), [u[i], v[i]]) ≈ l1[i]
        @test pdf(PlackettCopula(0.5), [u[i], v[i]]) ≈ l2[i]
    end
end

@testitem "PlackettCopula Sampling" begin
    using Random
    n_samples = 100
    c = PlackettCopula(0.8)
    samples = rand(c, n_samples)
    @test size(samples) == (2, n_samples)
end
