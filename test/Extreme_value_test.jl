@testitem "Extreme Value Copulas Tests" begin
    using InteractiveUtils
    using Distributions
    using Random
    using StableRNGs
    rng = StableRNG(123)
    d = 100
    @testset "AsymGalambosCopula - sampling, pdf, cdf" begin
        for α in [0.0, rand(rng, Uniform()), rand(rng, Uniform(5.0, 9.0)), rand(rng, Uniform(10.0, 15.0))]
            for θ in [[rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))], [0.0, 0.0], [1.0, 1.0]]
                C = AsymGalambosCopula(α, θ)
                data = rand(rng, C, 100)
                
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    pdf_value = pdf(C, u)
                    @test 0.0 <= cdf_value <= 1.0
                    @test pdf_value >= 0.0
                end    
            end
        end
    end

    @testset "AsymLogCopula - sampling, pdf, cdf" begin
        for α in [1.0, rand(rng, Uniform(1.0, 5.0)), rand(rng, Uniform(10.0, 15.0))]
            for θ in [[rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))], [0.0, 0.0], [1.0, 1.0]]
                C = AsymLogCopula(α, θ)
                data = rand(rng, C, 100)
                
                @test size(data) == (2, 100)

                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    pdf_value = pdf(C, u)
                    @test 0.0 <= cdf_value <= 1.0
                    @test pdf_value >= 0.0
                end    
            end
        end
    end

    @testset "AsymMixedCopula - sampling, pdf, cdf" begin
        for θ in [[0.1, 0.2], [0.0, 0.0], [0.3, 0.4], [0.2, 0.4]]
            try
                C = AsymMixedCopula(θ)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)

                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    pdf_value = pdf(C, u)

                    @test 0.0 <= cdf_value <= 1.0 || error("CDF failure: θ=$θ, u=$u, cdf_value=$cdf_value")
                    @test pdf_value >= 0.0 || error("PDF failure: θ=$θ, u=$u, cdf_value=$cdf_value")
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct AsymMixedCopula with θ=$θ: ", e)
            end
        end
    end

    @testset "BC2Copula - sampling, pdf, cdf" begin
        for θ in [[rand(rng), rand(rng)], [1.0, 0.0], [0.5, 0.5], [-0.1, 0.2], [1.1, 0.5], [0.5, -0.2]]
            try
                C = BC2Copula(θ...)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    pdf_value = pdf(C, u)
    
                    @test 0.0 <= cdf_value <= 1.0 || error("CDF failure: θ=$θ, u=$u, cdf_value=$cdf_value")
                    @test pdf_value >= 0.0 || error("PDF failure: θ=$θ, u=$u, cdf_value==$pdf_value")
                end
            catch e
                @test e isa ArgumentError
                println("Could not build BC2Copula with θ=$θ: ", e)
            end
        end
    end

    @testset "CuadrasAugeCopula - sampling, pdf, cdf" begin
        for θ in [rand(rng), 0.0, 1.0, -0.1, 1.1, rand(rng)]
            try
                C = CuadrasAugeCopula(θ)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    #if θ == 1.0
                     #   @test_throws ArgumentError pdf(C, u)
                    #else
                        pdf_value = pdf(C, u)
                        @test 0.0 <= cdf_value <= 1.0 || error("CDF failure: θ=$θ, u=$u, cdf_value=$cdf_value")
                        @test pdf_value >= 0.0 || error("PDF failure: θ=$θ, u=$u, cdf_value=$pdf_value")
                    #end
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct CuadrasAugeCopula with θ=$θ: ", e)
            end
        end
    end

    @testset "GalambosCopula - sampling, pdf, cdf" begin
        for θ in [rand(rng), 0.0, Inf, -1.0, rand(rng, Uniform(1.0, 5.0)), rand(rng, Uniform(5.0, 10.0))]
            try
                C = GalambosCopula(θ)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    #if  θ == Inf
                     #   @test_throws ArgumentError pdf(C, u)
                    #else
                        pdf_value = pdf(C, u)
                        @test 0.0 <= cdf_value <= 1.0 || error("CDF failure: θ=$θ, u=$u, cdf_value=$cdf_value")
                        @test pdf_value >= 0.0 || error("PDF failure: θ=$θ, u=$u, cdf_value=$pdf_value")
                    #end
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct GalambosCopula with θ=$θ: ", e)
            end
        end
    end

    @testset "HuslerReissCopula - sampling, pdf, cdf" begin
        for θ in [rand(rng), 0.0, Inf, -1.0, rand(rng, Uniform(1.0, 5.0)), rand(rng, Uniform(5.0, 10.0))]
            try
                C = HuslerReissCopula(θ)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    #if  θ == Inf
                    #    @test_throws ArgumentError pdf(C, u)
                    #else
                        pdf_value = pdf(C, u)
                        @test 0.0 <= cdf_value <= 1.0 || error("CDF failure: θ=$θ, u=$u, cdf_value=$cdf_value")
                        @test pdf_value >= 0.0 || error("PDF failure: θ=$θ, u=$u, cdf_value=$pdf_value")
                    #end
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct HuslerReissCopula with θ=$θ: ", e)
            end
        end
    end

    @testset "LogCopula - sampling, pdf, cdf" begin
        for θ in [1.0, Inf, 0.5, rand(rng, Uniform(1.0, 10.0))]
            try
                C1 = LogCopula(θ)
                C2 = GumbelCopula(2, θ)
                data = rand(rng, C1, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value_C1 = cdf(C1, u)
                    cdf_value_C2 = cdf(C2, u)
                    #if  θ == Inf
                    #    @test_throws ArgumentError pdf(C1, u)
                    #    @test_throws ArgumentError pdf(C2, u)
                    #else
                        pdf_value_C1 = pdf(C1, u)
                        pdf_value_C2 = pdf(C2, u)
                        @test 0.0 <= cdf_value_C1 <= 1.0 || error("CDF failure LogCopula: θ=$θ, u=$u, cdf_value_C1=$cdf_value_C1")
                        @test 0.0 <= cdf_value_C2 <= 1.0 || error("CDF failure de GumbelCopula: θ=$θ, u=$u, cdf_value_C2=$cdf_value_C2")
                        @test pdf_value_C1 >= 0.0 || error("PDF failure LogCopula: θ=$θ, u=$u, pdf_value_C1=$pdf_value_C1")
                        @test pdf_value_C2 >= 0.0 || error("PDF failure GumbelCopula: θ=$θ, u=$u, pdf_value_C2=$pdf_value_C2")
    
                        @test isapprox(cdf_value_C1, cdf_value_C2, atol=1e-6) || error("CDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, cdf_value_C1=$cdf_value_C1, cdf_value_C2=$cdf_value_C2")
                        @test isapprox(pdf_value_C1, pdf_value_C2, atol=1e-6) || error("PDF LogCopula and GumbelCopula do not match: θ=$θ, u=$u, pdf_value_C2=$pdf_value_C1, pdf_value_C2=$pdf_value_C2")
                    #end
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct LogCopula with θ=$θ: ", e)
            end
        end
    end

    @testset "MixedCopula - sampling, pdf, cdf" begin
        for θ in [0.0, 1.0, 0.2, -rand(rng), 0.5]
            try
                C = MixedCopula(θ)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    pdf_value = pdf(C, u)
                    @test 0.0 <= cdf_value <= 1.0 || error("CDF failure: θ=$θ, u=$u, cdf_value=$cdf_value")
                    @test pdf_value >= 0.0 || error("PDF failure: θ=$θ, u=$u, cdf_value=$pdf_value")
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct MixedCopula with θ=$θ: ", e)
            end
        end
    end

    @testset "MOCopula - sampling, pdf, cdf" begin
        param_sets = [
            (0.1, 0.2, 0.3),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (-0.1, 0.2, 0.3),  # Invalid case: negative parameter
            (0.1, -0.2, 0.3),  # Invalid case: negative parameter
            (0.1, 0.2, -0.3),  # Invalid case: negative parameter
            (rand(rng), rand(rng), rand(rng))  # Random Params
        ]
    
        for λ in param_sets
            try
                C = MOCopula(λ...)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    cdf_value = cdf(C, u)
                    pdf_value = pdf(C, u)
    
                    @test 0.0 <= cdf_value <= 1.0 || error("cdf failure: λ=$λ, u=$u, cdf_value=$cdf_value")
                    @test pdf_value >= 0.0 || error("pdf failure: λ=$λ, u=$u, pdf_value=$pdf_value")
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct MOCopula with λ=$λ: ", e)
            end
        end
    end

    @testset "tEVCopula - sampling, pdf, cdf" begin
        param_sets = [
            (2.0, 0.5),  # ν > 0 y -1 < ρ <= 1
            (5.0, -0.5),  # ν > 0 y -1 < ρ <= 1
            (10.0, 1.0),  # ν > 0 y ρ == 1, MCopula
            (3.0, 0.0),  # ν > 0 y ρ == 0, Independent Copula
            (-2.0, 0.5),  # Invalid case: ν <= 0
            (3.0, 1.1),  # Invalid case: ρ out of range
            (3.0, -1.1),  # Invalid case: ρ out of range
            (rand(rng, Uniform(4.0, 10.0)), rand(rng, Uniform(-0.9, 1.0)))  # random params inside of range
        ]
    
        for (ν, ρ) in param_sets
            try
                C = tEVCopula(ν, ρ)
                data = rand(rng, C, 100)
    
                @test size(data) == (2, 100)
    
                for i in 1:d
                    u = data[:,i]
                    
                    cdf_value = cdf(C, u)
                    #if  ρ == 1
                    #    @test_throws ArgumentError pdf(C, u)
                    #else
                        pdf_value = pdf(C, u)
                        @test 0.0 <= cdf_value <= 1.0 || error("CDF Failure: ν=$ν, ρ=$ρ, u=$u, cdf_value=$cdf_value")
                        @test pdf_value >= 0.0 || error("PDF Failure: ν=$ν, ρ=$ρ, u=$u, pdf_value=$pdf_value")
                    #end
                end
            catch e
                @test e isa ArgumentError
                println("Could not construct tEVCopula with ν=$ν, ρ=$ρ: ", e)
            end
        end
    end
end

@testitem "Pickands Function test" begin
    cops = (
        AsymGalambosCopula(5.0, [0.8, 0.3]),
        AsymLogCopula(1.5, [0.5, 0.2]),
        AsymMixedCopula([0.1, 0.2]),
        BC2Copula(0.5, 0.3),
        CuadrasAugeCopula(0.8),
        GalambosCopula(4.3),
        HuslerReissCopula(3.5),
        LogCopula(5.5),
        MixedCopula(0.5),
        MOCopula(0.1, 0.2, 0.3),
        tEVCopula(4.0, 0.5))

        for C in cops
            for C in cops
                @test Copulas.A(C, 0.0) == 1
                @test Copulas.A(C, 1.0) == 1
                t = rand(rng, Uniform())
                A_value = Copulas.A(C, t)
                @test 0.0 <= A_value <= 1.0
                @test max(t, 1-t) <= A_value <= 1.0
            end
        end
end