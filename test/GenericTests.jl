@testmodule M begin
    using Copulas
    using HypothesisTests, Distributions, Random, WilliamsonTransforms
    using InteractiveUtils
    using ForwardDiff
    using StatsBase: corkendall
    using StableRNGs
    using Test
    rng = StableRNG(123)

    function _subtypes(type::Type)
        out = Any[]
        _subtypes!(out, type)
    end
    function _subtypes!(out, type::Type)
        if !isabstracttype(type)
            push!(out, type)
        else
            foreach(T->_subtypes!(out, T), InteractiveUtils.subtypes(type))
        end
        out
    end
    # @testset "Generic tests: missing copulas in the list" begin
    #     @test all(any(isa(C,CT) for C in bestiary) for CT in _subtypes(Copulas.Copula))
    #     @test all(any(isa(C.G,TG) for C in bestiary if typeof(C)<:Copulas.ArchimedeanCopula) for TG in _subtypes(Copulas.Generator))
    # end

    #### methods to numerically derivate the pdf from the cdf :
    # Not really efficient as in some cases this return zero while the true pdf is clearly not zero.
    function _v(u,j,uj)
        return [(i == j ? uj : u[i]) for i in 1:length(u)]
    end
    function _der(j,C,u)
        if j == 1
            return ForwardDiff.derivative(u1 -> cdf(C,_v(u,1,u1)), u[1])
        else
            return ForwardDiff.derivative(uj -> _der(j-1,C,_v(u,j,uj)),u[j])
        end
    end
    function get_numerical_pdf(C,u)
        _der(length(C),C,u)
    end

    # Filter on archimedeans for fitting tests.
    function is_archimedean_with_agenerator(CT)
        if CT<:ArchimedeanCopula
            GT = Copulas.generatorof(CT)
            if !isnothing(GT)
                if !(GT<:Copulas.ZeroVariateGenerator)
                    if !(GT<:Copulas.WilliamsonGenerator)
                        return true
                    end
                end
            end
        end
        return false
    end

    function check(C::Copulas.Copula{d}) where d
        Random.seed!(rng,123) # set seed.
        CT = typeof(C)
        D = SklarDist(C, Tuple(Normal() for i in 1:d))
        spl10 = rand(rng,C,10)
        spl1000 = rand(rng,C,1000)

        # Sampling shapes: 
        @test length(rand(rng,C))==d # to sample only one value
        @test size(spl10) == (d,10)

        # Check samples in  [0,1]" begin
        @test all(0 .<= spl10 .<= 1)
        @test all(0 .<= spl1000 .<= 1)
        
        # Check cdf > 0 and bounday conditions."
        @test iszero(cdf(C,zeros(d)))
        @test isone(cdf(C,ones(d)))
        @test 0 <= cdf(C,rand(rng,d)) <= 1
        @test all(0 .<= spl10 .<= 1)
        @test cdf(D,zeros(d)) >= 0
    
        # Test that the measure function works correctly:
        # Check measure >0 and boundary conditions
        @test Copulas.measure(C,zeros(d),ones(d)) ≈ 1
        u_mes = ones(d)*0.2
        v_mes = ones(d)*0.4
        @test Copulas.measure(C, u_mes, v_mes) >= 0

        # Check cdf's margins uniformity" begin
        if !(CT<:EmpiricalCopula) # this one is not a true copula :)
            for i in 1:d
                for val in [0,1,0.5,rand(rng,5)...]
                    u = ones(d)
                    u[i] = val
                    @test cdf(C,u) ≈ val atol=1e-5
                end
                # extra check for zeros:
                u = rand(rng,d)
                u[i] = 0
                @test iszero(cdf(C,u))
            end
        end
        
        # @testset "Check pdf > 0" begin
        has_pdf(C) = applicable(Distributions._logpdf,C,rand(length(C),3))
        if has_pdf(C)
            @test pdf(C,ones(length(C))/2) >= 0
            @test all(pdf(C, spl10) .>= 0)
            @test pdf(D, zeros(d)) >= 0 # also the sklardist pdf should be strictily positive there.
        end

        if hasmethod(Copulas.A, (typeof(C), Float64))
            # ExtremeValues: Check A's boundary conditions" begin
            @test Copulas.A(C, 0.0) == 1
            @test Copulas.A(C, 1.0) == 1
            t = rand(rng)
            A_value = Copulas.A(C, t)
            @test 0.0 <= A_value <= 1.0
            @test isapprox(A_value, max(t, 1-t); atol=1e-6) || A_value >= max(t, 1-t)
            @test A_value <= 1.0

            # Maybe we should also check the \ell and C connections to A ?
            # to be certain to catch any mistake.
        end

        # Match theoretical and empirical kendall taus" begin
        K = corkendall(spl1000')
        Kth = corkendall(C)
        @test all(-1 .<= Kth .<= 1)
        @test all(isapprox.(Kth, K; atol=0.2))
    
        # Extra checks, only for archimedeans.
        if is_archimedean_with_agenerator(CT)
            if applicable(Copulas.τ, C.G) && applicable(Copulas.τ⁻¹,typeof(C.G), 1.0)
                # Archimedean with generator: check τ ∘ τ_inv == Id" begin
                tau = Copulas.τ(C)
                @test Copulas.τ(Copulas.generatorof(CT)(Copulas.τ⁻¹(CT,tau))) ≈ tau
            end

            if applicable(Copulas.ρ, C.G) && applicable(Copulas.ρ⁻¹,typeof(C.G), 1.0)
                # Archimedean with generator: check ρ ∘ ρ_inv == Id" begin
                rho = Copulas.ρ(C)
                @test -1 <= rho <= 1
                @test Copulas.ρ(Copulas.generatorof(CT)(Copulas.ρ⁻¹(CT,rho))) ≈ rho
            end

            # Check fit() runs" begin
            fit(CT,spl10)

            # Check ϕ ∘ ϕ_inv == Id" begin
            for x in 0:0.1:1
                @test Copulas.ϕ⁻¹(C,Copulas.ϕ(C,x)) ≈ x
            end

            # Check that williamson_dist and ϕ⁻¹ actually corresponds" begin
            splW_method1 = dropdims(sum(Copulas.ϕ⁻¹.(Ref(C),spl1000),dims=1),dims=1)
            splW_method2 = rand(rng,Copulas.williamson_dist(C),1000)
            @test pvalue(ApproximateTwoSampleKSTest(splW_method1,splW_method2),tail=:right) > 0.01
        end
    end
end

