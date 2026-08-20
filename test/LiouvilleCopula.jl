@testset "Liouville copulas" begin
    @testset "real Williamson orders" begin
        G = Copulas.𝒲(Dirac(1.0), 5.5)
        C = LiouvilleCopula{3}(G, (0.75, 1.5, 3.0))
        @test C isa Copulas.Copula{3}
        @test Copulas.𝒲₋₁(G, sum(C.α)) isa Copulas.WilliamsonBetaProduct
        @test_throws ArgumentError LiouvilleCopula{2}(G, (3.0, 3.0))

        U = rand(rng, C, 5)
        @test size(U) == (3, 5)
        @test all(0 .<= U .<= 1)

        C13 = subsetdims(C, (1, 3))
        @test C13 isa LiouvilleCopula{2}
        @test C13.G === C.G
        @test C13.α == (C.α[1], C.α[3])
    end

    @testset "Archimedean identity" begin
        G = Copulas.ClaytonGenerator(1.0)
        L = LiouvilleCopula{2}(G, (1.0, 1.0))
        A = ArchimedeanCopula{2}(G)
        u = [0.35, 0.7]
        @test cdf(L, u) ≈ cdf(A, u) atol=1e-7
        @test logpdf(L, u) ≈ logpdf(A, u)
    end

    @testset "conditioning and Rosenblatt" begin
        integer_C = LiouvilleCopula{3}(
            Copulas.ClaytonGenerator(1.0), (1.0, 1.0, 1.0),
        )
        integer_conditional = Copulas.ConditionalCopula(integer_C, (1,), (0.4,))
        @test integer_conditional.G isa Copulas.TiltedGenerator
        @test condition(integer_C, (1,), (0.4,)) isa SklarDist

        fractional_C = LiouvilleCopula{3}(
            Copulas.𝒲(Dirac(1.0), 4.0), (0.6, 1.1, 1.3),
        )
        fractional_conditional = Copulas.ConditionalCopula(fractional_C, (1,), (0.4,))
        @test fractional_conditional.G isa Copulas.𝒲

        D = Copulas.DistortionFromCop(fractional_C, (1,), (0.4,), 2)
        p = Distributions.cdf(D, 0.6)
        @test Distributions.quantile(D, p) ≈ 0.6

        u = [0.3, 0.5, 0.8]
        v = rosenblatt(fractional_C, u)
        @test inverse_rosenblatt(fractional_C, v) ≈ u
    end
end
