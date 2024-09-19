@testitem "Extreme Value Copulas Tests" begin
    using InteractiveUtils
    using Copulas, Distributions
    using Random
    using StableRNGs

    rng = StableRNG(1234)

    @testset "Check bounds on parametrisations" begin
        @test_throws ArgumentError, AsymMixedCopula([0.3,0.4])
        @test_throws ArgumentError BC2Copula(-0.1, 0.2)
        @test_throws ArgumentError BC2Copula(1.1, 0.5)
        @test_throws ArgumentError BC2Copula(0.5, -0.2)
        @test_throws ArgumentError CuadrasAugeCopula(-0.1)
        @test_throws ArgumentError CuadrasAugeCopula(1.1)
        @test_throws ArgumentError GalambosCopula(-1.0)
        @test_throws ArgumentError HuslerReissCopula(-1.0)
        @test_throws ArgumentError MixedCopula(-rand(rng))
        @test_throws ArgumentError MOCopula(-0.1, 0.2, 0.3),  # Invalid case: negative parameter
        @test_throws ArgumentError MOCopula(0.1, -0.2, 0.3),  # Invalid case: negative parameter
        @test_throws ArgumentError MOCopula(0.1, 0.2, -0.3),  # Invalid case: negative parameter
        @test_throws ArgumentError tEVCopula(-2.0, 0.5),  # Invalid case: ν <= 0
        @test_throws ArgumentError tEVCopula(3.0, 1.1),  # Invalid case: ρ out of range
        @test_throws ArgumentError tEVCopula(3.0, -1.1),  # Invalid case: ρ out of range
        @test_throws ArgumentError LogCopula(0.5)
    end
    biv_ev_cops = []
    for α in [0.0, rand(rng, Uniform()), rand(rng, Uniform(5.0, 9.0)), rand(rng, Uniform(10.0, 15.0))]
        for θ in [[rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))], [0.0, 0.0], [1.0, 1.0]]
            push!(biv_ev_cops, AsymGalambosCopula(α, θ))
        end
    end
    for α in [1.0, rand(rng, Uniform(1.0, 5.0)), rand(rng, Uniform(10.0, 15.0))]
        for θ in [[rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))], [0.0, 0.0], [1.0, 1.0]]
            push!(biv_ev_cops, AsymLogCopula(α, θ))
        end
    end
    for θ in [[0.1, 0.2], [0.0, 0.0], [0.2, 0.4]]
        push!(biv_ev_cops, AsymMixedCopula(θ))
    end
    for θ in [20, 60, 70, 80, 120, 210]
        push!(biv_ev_cops, GalambosCopula(θ))
    end
    for θ in [[rand(rng), rand(rng)], [1.0, 0.0], [0.5, 0.5], ]
        push!(biv_ev_cops, BC2Copula(θ...))
    end
    for θ in [rand(rng), 0.0, 1.0, rand(rng)]
        push!(biv_ev_cops, CuadrasAugeCopula(θ))
    end
    for θ in [rand(rng), 0.0, Inf, rand(rng, Uniform(1.0, 5.0)), rand(rng, Uniform(5.0, 10.0))]
        push!(biv_ev_cops, GalambosCopula(θ))
    end
    for θ in [rand(rng), 0.0, Inf, rand(rng, Uniform(1.0, 5.0)), rand(rng, Uniform(5.0, 10.0))]
        push!(biv_ev_cops, HuslerReissCopula(θ))
    end
    for θ in [0.0, 1.0, 0.2, 0.5]
        push!(biv_ev_cops, MixedCopula(θ))
    end
    for λ in [
        (0.1, 0.2, 0.3),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (rand(rng), rand(rng), rand(rng))  # Random Params
    ]
        push!(biv_ev_cops, MOCopula(λ...))
    end
    for (ν, ρ) in [
        (2.0, 0.5),  # ν > 0 y -1 < ρ <= 1
        (5.0, -0.5),  # ν > 0 y -1 < ρ <= 1
        (10.0, 1.0),  # ν > 0 y ρ == 1, MCopula
        (3.0, 0.0),  # ν > 0 y ρ == 0, Independent Copula
        (rand(rng, Uniform(4.0, 10.0)), rand(rng, Uniform(-0.9, 1.0)))  # random params inside of range
    ]
        push!(biv_ev_cops, tEVCopula(ν, ρ))
    end
    for θ in [1.0, Inf, rand(rng, Uniform(1.0, 10.0))]
        push!(biv_ev_cops, LogCopula(θ))
    end
    append!(biv_ev_cops, [
        IndependentCopula(2),
        AsymGalambosCopula(0.6129496106778634, [0.820474440393214, 0.22304578643880224]),
        AsymGalambosCopula(8.810168494949659, [0.5987759444612732, 0.5391280234619427]),
        AsymGalambosCopula(11.647356700032505, [0.6195348270893413, 0.4197760589260566]),
        AsymLogCopula(1.0, [0.8360692316060747, 0.68704221750134]),
        AsymLogCopula(1.0, [0.0, 0.0]),
        AsymLogCopula(1.0, [1.0, 1.0]),
        AsymLogCopula(2.8130363753722403, [0.3539590866764071, 0.15146985093210463]),
        AsymLogCopula(2.8130363753722403, [0.0, 0.0]),
        AsymLogCopula(2.8130363753722403, [1.0, 1.0]),
        AsymLogCopula(12.29006035397328, [0.7036713552821277, 0.7858058549340399]),
        AsymLogCopula(12.29006035397328, [0.0, 0.0]),
        AsymLogCopula(12.29006035397328, [1.0, 1.0]),
        AsymMixedCopula([0.1, 0.2]),
        AsymMixedCopula([0.2, 0.4]),
        GalambosCopula(0.6129496106778634),
        GalambosCopula(8.810168494949659),
        GalambosCopula(11.647356700032505),
        GalambosCopula(20),
        GalambosCopula(60),
        GalambosCopula(70),
        GalambosCopula(80),
        GalambosCopula(120),
        GalambosCopula(210),
        GalambosCopula(0.40543796744015514),
        GalambosCopula(2.675150743283436),
        GalambosCopula(6.730938346629261),
        BC2Copula(0.5516353577049822, 0.33689370624999193),
        BC2Copula(1.0, 0.0),
        BC2Copula(0.5, 0.5),
        CuadrasAugeCopula(0.7103550345192344),
        CuadrasAugeCopula(0.3437537135972244),
        MCopula(2),
        HuslerReissCopula(0.256693308150987),
        HuslerReissCopula(1.6287031392529938),
        HuslerReissCopula(5.319851350643586),
        MixedCopula(1.0),
        MixedCopula(0.2),
        MixedCopula(0.5),
        MOCopula(0.1, 0.2, 0.3),
        MOCopula(1.0, 1.0, 1.0),
        MOCopula(0.5, 0.5, 0.5),
        MOCopula(0.5960710257852946, 0.3313524247810329, 0.09653466861970061),
        tEVCopula(2.0, 0.5),
        tEVCopula(5.0, -0.5),
        tEVCopula(5.466564460573727, -0.6566645244416698),
        LogCopula(4.8313231991648244)
    ])
    unique!(biv_ev_cops)

    for C in biv_ev_cops 
        @testset "$C - sampling, pdf, cdf, Pickhand function test" begin
            data = rand(rng, C, 10)
            @test size(data) == (2, 10)
            for i in 1:10

                # These tests should be true on EVERY copulas. One more argument to completely refactor the test base to be more generic and easier to maintain.

                u = data[:,i]
                cdf_value = cdf(C, u)
                pdf_value = pdf(C, u)
                @test 0.0 <= cdf_value <= 1.0 || error("CDF failure: C=$C, u=$u, cdf_value=$cdf_value")
                @test pdf_value >= 0.0 || error("PDF failure: C=$C, u=$u, cdf_value=$cdf_value")
            end    
            
            if hasmethod(Copulas.A, (typeof(C), Float64))
                @test Copulas.A(C, 0.0) == 1
                @test Copulas.A(C, 1.0) == 1
                t = rand(rng)
                A_value = Copulas.A(C, t)
                @test 0.0 <= A_value <= 1.0
                @test isapprox(A_value, max(t, 1-t); atol=1e-6) || A_value >= max(t, 1-t)
                @test A_value <= 1.0
            end
        end
    end


    
    @testset "Checking LogCopula == GumbelCopula and GumbelCopula match" begin
        for θ in [1.0, Inf, 0.5, rand(rng, Uniform(1.0, 10.0))]
            try
                C1 = LogCopula(θ)
                C2 = GumbelCopula(2, θ)
                data = rand(rng, C1, 10)
    
                for i in 1:10
                    u = data[:,i]
                    cdf_value_C1 = cdf(C1, u)
                    cdf_value_C2 = cdf(C2, u)
                    pdf_value_C1 = pdf(C1, u)
                    pdf_value_C2 = pdf(C2, u)

                    @test isapprox(cdf_value_C1, cdf_value_C2, atol=1e-6) || error("CDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, cdf_value_C1=$cdf_value_C1, cdf_value_C2=$cdf_value_C2")
                    @test isapprox(pdf_value_C1, pdf_value_C2, atol=1e-6) || error("PDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, pdf_value_C2=$pdf_value_C1, pdf_value_C2=$pdf_value_C2")
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct LogCopula with θ=$θ: ", e)
            end
        end
    end
end