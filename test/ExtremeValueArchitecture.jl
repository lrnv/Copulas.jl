@testset "Extreme-value architecture" begin
    @testset "dimension-aware constructors" begin
        for (C, d) in (
            (LogCopula(5, 2.0), 5),
            (GalambosCopula(4, 0.7), 4),
            (HuslerReissCopula(3, 1.0), 3),
            (MixedCopula(4, 0.5), 4),
            (tEVCopula(3, 4.0, 0.2), 3),
        )
            @test length(C) == d
        end

        @test_throws ArgumentError AsymLogCopula(3, 1.5, 0.4, 0.6)
        @test_throws ArgumentError Copulas.ExtremeValueCopula(1, Copulas.GalambosTail(0.7))

        Cind = LogCopula(3, 1.0)
        Cdep = LogCopula(3, Inf)
        @test length(Cind) == 3
        @test length(Cdep) == 3
        @test cdf(Cind, fill(0.5, 3)) ≈ 0.5^3
        @test cdf(Cdep, fill(0.5, 3)) ≈ 0.5
    end

    @testset "bivariate density routing" begin
        u = [0.31, 0.67]

        for C in (
            GalambosCopula(2, 0.7),
            HuslerReissCopula(2, 1.0),
            MixedCopula(2, 0.5),
            tEVCopula(2, 4.0, 0.5),
        )
            lb = Copulas._ev_logpdf_bivariate(C, u)
            lm = Copulas._ev_logpdf_from_partials(C, u)
            @test logpdf(C, u) == lb
            @test lb ≈ lm atol=5e-13 rtol=5e-13
        end
    end

    @testset "strong logistic density" begin
        for θ in (2.0, 13.5, 210.0)
            C = LogCopula(2, θ)
            G = Copulas.GumbelCopula(2, θ)
            for u in ([1e-3, 0.99], [0.01, 0.9], [0.99, 0.5], [0.99, 0.99])
                @test logpdf(C, u) ≈ logpdf(G, u) atol=2e-12 rtol=2e-12
            end
        end
    end
end
