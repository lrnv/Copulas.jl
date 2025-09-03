@testmodule M begin
    using Copulas
    using HypothesisTests, Distributions, Random, WilliamsonTransforms
    using InteractiveUtils
    using ForwardDiff
    using StatsBase: corkendall
    using StableRNGs
    using HCubature
    using Test
    using LogExpFunctions

    rng = StableRNG(123)

    # Filter on archimedeans for fitting tests.
    function is_archimedean_with_agenerator(CT)
        if CT<:ArchimedeanCopula
            GT = Copulas.generatorof(CT)
            if !isnothing(GT)
                if !(GT<:Copulas.WilliamsonGenerator)
                    return true
                end
            end
        end
        return false
    end

    function mc_pdf_integral_rect(rng, C::Copulas.Copula{d}; a=zeros(d), b=ones(d), N=100_000) where d
        logS, logS2, n_eff  = -Inf, -Inf, 0
        u, x = zeros(d), zeros(d)
        ba = b .- a
        logvol = log(prod(ba))
        @inbounds for _ in 1:N
            rand!(rng, u)            # U ~ Unif(0,1)^d
            x .= a .+ ba .* u        # X en [a,b]
            lp = logpdf(C, x)
            if isfinite(lp)
                log_fx = lp + logvol # pdf_X(x) = pdf(C,x) * vol
                logS   = LogExpFunctions.logaddexp(logS,  log_fx)
                logS2  = LogExpFunctions.logaddexp(logS2, 2*log_fx)
                n_eff += 1
            end
        end
        μ   = exp(logS  - log(n_eff))
        m2  = exp(logS2 - log(n_eff))
        var = max(m2 - μ^2, 0.0)
        return μ, var / n_eff, n_eff
    end

    function integrate_pdf_rect(rng, C::Copulas.Copula{d}; a=zeros(d), b=ones(d), maxevals=10_000, nmc=100_000) where d
        dim = length(C)
        if dim <= 3
            try
                v, abs_err = hcubature(x -> pdf(C, x), a, b; maxevals=maxevals)
                    v_true = Copulas.measure(C, a, b)
                    if isapprox(v, v_true; atol=10*abs_err)
                        return v, abs_err^2, :hcubature   # r = SE²
                end
            catch
            end
        end
        μ, r, _ = mc_pdf_integral_rect(rng, C; a=a, b=b, N=nmc)
        return μ, r, :mc
    end

    has_pdf(C) = applicable(Distributions._logpdf,C,rand(rng,length(C),3))
    function check_density_intergates_to_cdf(C::CT) where CT
        return has_pdf(C) && 
                !(CT<:Union{MOCopula, EmpiricalCopula, WCopula, MCopula, BC2Copula}) && 
                !(CT<:ClaytonCopula) &&
                !(CT<:Union{CuadrasAugeCopula, GalambosCopula, AsymGalambosCopula}) &&
                !((CT<:AMHCopula) && (C.G.θ == -1.0)) &&
                !((CT<:FrankCopula) && (C.G.θ >= 100)) &&
                !((CT<:ArchimedeanCopula) && length(C)==Copulas.max_monotony(C.G))
    end

    function check(C::Copulas.Copula{d}) where d

        @testset "Testing $C" begin
            @info "start $C"
            CT = typeof(C)
            Random.seed!(rng,123)
            D = SklarDist(C, Tuple(Normal() for i in 1:d))
            spl10 = rand(rng,C,10)
            spl1000 = rand(rng,C,1000)
            @testset "General tests" begin 
                # Sampling shape and support
                @test length(rand(rng,C))==d
                @test size(spl10) == (d,10)
                @test all(0 .<= spl10 .<= 1)
                @test all(0 .<= spl1000 .<= 1)
            end

            @testset "CDF boundary and measure" begin 
                @test iszero(cdf(C,zeros(d)))
                @test isone(cdf(C,ones(d)))
                @test 0 <= cdf(C,rand(rng,d)) <= 1
                @test cdf(D,zeros(d)) >= 0
                @test Copulas.measure(C, zeros(d),    ones(d)) ≈ 1
                @test Copulas.measure(C, ones(d)*0.2, ones(d)*0.4) >= 0
            end

            sC = Copulas.subsetdims(C,(1,2))
            sD = Copulas.subsetdims(D,(2,1))
            @testset "Subsetdims" begin
                @test isa(Copulas.subsetdims(C,(1,)), Distributions.Uniform)
                @test isa(Copulas.subsetdims(D,1), Distributions.Normal)
                @test all(0 .<= cdf(sC,rand(sC,10)) .<= 1)
                @test all(0 .<= cdf(sD,rand(sD,10)) .<= 1)
                @test sD.C == Copulas.subsetdims(C,(2,1)) # check for coherence. 
            end

            # Margins uniformity
            @testset "Margins uniformity" begin
                if !(CT<:EmpiricalCopula)
                    for i in 1:d
                    for val in [0,1,0.5,rand(rng,5)...]
                        u = ones(d)
                        u[i] = val
                        @test cdf(C,u) ≈ val atol=1e-5
                    end
                    u = rand(rng,d)
                    u[i] = 0
                    @test iszero(cdf(C,u))
                    end
                end
            end

            if has_pdf(C)
                @testset "PDF positivity" begin
                    @test pdf(C,ones(length(C))/2) >= 0
                    @test all(pdf(C, spl10) .>= 0)
                    @test pdf(D, zeros(d)) >= 0
                end
            end

            # Archimedean-specific tests
            if is_archimedean_with_agenerator(CT)
                @testset "Archimedean specifics" begin 
                    @testset "Specific implementation of derivatives match generics" begin
                        # Derivative tests for Archimedean generators
                        specialized_ϕ1 = which(Copulas.ϕ⁽¹⁾, (typeof(C.G), Float64)) != which(Copulas.ϕ⁽¹⁾, (Copulas.Generator, Float64))
                        specialized_ϕk = which(Copulas.ϕ⁽ᵏ⁾, (typeof(C.G), Val{1}, Float64)) != which(Copulas.ϕ⁽ᵏ⁾, (Copulas.Generator, Val{1}, Float64))
                        specialized_ϕinv = which(Copulas.ϕ⁻¹, (typeof(C.G), Float64)) != which(Copulas.ϕ⁻¹, (Copulas.Generator, Float64))
                        specialized_ϕinv1 = which(Copulas.ϕ⁻¹⁽¹⁾, (typeof(C.G), Float64)) != which(Copulas.ϕ⁻¹⁽¹⁾, (Copulas.Generator, Float64))
                        specialized_ϕkinv = which(Copulas.ϕ⁽ᵏ⁾⁻¹, (typeof(C.G), Val{1}, Float64)) != which(Copulas.ϕ⁽ᵏ⁾⁻¹, (Copulas.Generator, Val{1}, Float64))

                        if specialized_ϕinv
                            @testset "ϕ⁻¹ ∘ ϕ == Id" begin
                                for x in 0:0.1:1
                                    @test Copulas.ϕ⁻¹(C.G,Copulas.ϕ(C.G,x)) ≈ x atol=1e-10
                                end
                            end
                        end

                        if specialized_ϕ1
                            @testset "Check d(ϕ) == ϕ⁽¹⁾" begin 
                                @test ForwardDiff.derivative(x -> Copulas.ϕ(C.G, x), 1.2) ≈ Copulas.ϕ⁽¹⁾(C.G, 1.2)
                            end
                        end

                        if specialized_ϕk
                            @testset "d(ϕ) == ϕ⁽ᵏ⁾(k=1)" begin
                                @test ForwardDiff.derivative(x -> Copulas.ϕ(C.G, x), 1.2) ≈ Copulas.ϕ⁽ᵏ⁾(C.G, Val(1), 1.2)
                                if applicable(Copulas.ϕ⁽¹⁾, C, 1.2) && applicable(Copulas.ϕ⁽ᵏ⁾, C, Val(2), 1.2)
                                    @testset "Check ϕ⁽¹⁾ == ϕ⁽ᵏ⁾(k=1)" begin
                                        @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(C.G, x), 1.2) ≈ Copulas.ϕ⁽ᵏ⁾(C.G, Val(2), 1.2)
                                    end
                                end
                            end
                        end
                        
                        if specialized_ϕinv1
                            @testset "Check d(ϕ⁻¹) == ϕ⁻¹⁽¹⁾" begin
                                @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(C.G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(C.G, 0.5)
                            end
                        end

                        if specialized_ϕkinv
                            @testset "Check ϕ⁻¹ ∘ ϕ == Id" begin
                                for x in 0.1:0.1:0.5
                                    @test Copulas.ϕ⁽ᵏ⁾⁻¹(C.G,Val{d-1}(), Copulas.ϕ⁽ᵏ⁾(C.G, Val{d-1}(), x)) ≈ x
                                end
                            end
                        end
                    end
                    
                    if !((CT <: Copulas.GumbelBarnettCopula) && d>2) && !(CT<:Copulas.AMHCopula && d > 2)
                         if applicable(Copulas.τ, C.G) && applicable(Copulas.τ⁻¹,typeof(C.G), 1.0)
                            @testset "Check τ ∘ τ⁻¹ == Id" begin
                                tau = Copulas.τ(C)
                                @test Copulas.τ(Copulas.generatorof(CT)(Copulas.τ⁻¹(CT,tau))) ≈ tau
                            end
                            @testset "Check fitting the copula" begin
                                fit(CT,spl10)
                            end
                        end
                        
                        if applicable(Copulas.ρ, C.G) && applicable(Copulas.ρ⁻¹,typeof(C.G), 1.0)
                            @testset  "Check ρ ∘ ρ⁻¹ == Id" begin
                                rho = Copulas.ρ(C)
                                @test -1 <= rho <= 1
                                @test Copulas.ρ(Copulas.generatorof(CT)(Copulas.ρ⁻¹(CT,rho))) ≈ rho
                            end
                        end


                    end
                    
                    @testset "Check kendall distribution coherence between ϕ⁻¹+rand and williamson_dist" begin
                        splW_method1 = dropdims(sum(Copulas.ϕ⁻¹.(C.G,spl1000),dims=1),dims=1)
                        splW_method2 = rand(rng,Copulas.williamson_dist(C.G, Val{d}()),1000)
                        @test pvalue(ApproximateTwoSampleKSTest(splW_method1,splW_method2),tail=:right) > 0.01
                    end
                end
            end

            # Extreme value copula-specific tests
            if CT<:Copulas.ExtremeValueCopula
                @testset "Extreme value copulas specifics" begin
                    @testset "A function basics" begin
                        @test Copulas.A(C, 0.0) ≈ 1
                        @test Copulas.A(C, 1.0) ≈ 1
                        t = rand(rng)
                        A_value = Copulas.A(C, t)
                        @test 0.0 <= A_value <= 1.0
                        @test isapprox(A_value, max(t, 1-t); atol=1e-6) || A_value >= max(t, 1-t)
                        @test A_value <= 1.0
                    end

                    # Only run derivative and related tests if the methods are specialized for this type
                    specialized_dA =        which(Copulas.dA, (typeof(C), Float64))        != which(Copulas.dA, (Copulas.ExtremeValueCopula, Float64))
                    specialized_d²A =       which(Copulas.d²A, (typeof(C), Float64))       != which(Copulas.d²A, (Copulas.ExtremeValueCopula, Float64))
                    specialized__A_dA_d²A = which(Copulas._A_dA_d²A, (typeof(C), Float64)) != which(Copulas._A_dA_d²A, (Copulas.ExtremeValueCopula, Float64))
                    specialized_ℓ =         which(Copulas.ℓ, (typeof(C), Float64, Float64))         != which(Copulas.ℓ, (Copulas.ExtremeValueCopula, Float64, Float64))

                    if specialized_dA || specialized_d²A || specialized__A_dA_d²A
                        @testset "Testing derivatives of A" begin
                            for t in (0.05, 0.5, 0.95)
                                if !(CT<:tEVCopula)
                                    @test isapprox(Copulas.dA(C, t), ForwardDiff.derivative(x -> Copulas.A(C, x), t); atol=1e-6)
                                    @test isapprox(Copulas.d²A(C, t), ForwardDiff.derivative(x -> Copulas.dA(C, x), t); atol=1e-6)
                                end
                                a, da, d2a = Copulas._A_dA_d²A(C, t)
                                @test isapprox(a, Copulas.A(C, t); atol=1e-8)
                                @test isapprox(da, Copulas.dA(C, t); atol=1e-8)
                                @test isapprox(d2a, Copulas.d²A(C, t); atol=1e-8)
                            end
                        end
                    end

                    if specialized_dA || specialized_d²A || specialized__A_dA_d²A || specialized_ℓ
                        @testset "Testing ℓ and cdf for Extreme Value Copula" begin 
                            u, v = rand(rng), rand(rng)
                            x, y = -log(u), -log(v)
                            s = y / (x + y)
                            expected_ℓ = Copulas.A(C, s) * (x + y)
                            @test isapprox(Copulas.ℓ(C, x, y), expected_ℓ; atol=0.1)
                            expected_cdf = exp(-expected_ℓ)
                            @test isapprox(cdf(C, [u, v]), expected_cdf; atol=0.1)

                            if !(CT<:tEVCopula)
                                u, v = rand(rng), rand(rng)
                                num_pdf = ForwardDiff.derivative(u_ -> ForwardDiff.derivative(v_ -> cdf(C, [u_, v_]), v), u)
                                ana_pdf = pdf(C, [u, v])
                                @test isapprox(ana_pdf, num_pdf; atol=0.1)
                            end
                        end
                    end
                end
            end

            # Generic tests
            @testset "Testing corkendall coeherency" begin
                K = corkendall(spl1000')
                Kth = corkendall(C)
                @test all(-1 .<= Kth .<= 1)
                @test all(isapprox.(Kth, K; atol=0.2))
            end

            if check_density_intergates_to_cdf(C)
                @testset "Testing pdf integration" begin
                    dim = length(C)

                    # 1) ∫_{[0,1]^d} pdf = 1  (hcubature if d≤3; si no, MC)
                    v, r, _ = integrate_pdf_rect(rng, C)
                    @test isapprox(v, 1; atol=5*sqrt(r))

                    # 2) ∫_{[0,0.5]^d} pdf = C(0.5,…,0.5)
                    b = fill(0.5, dim)
                    v2, r2, _ = integrate_pdf_rect(rng, C; b=b)
                    @test isapprox(v2, cdf(C, b); atol=10*sqrt(r2))

                    # 3) random rectangle and comparation with measure
                    a = rand(rng, dim)
                    b = a .+ rand(rng, dim) .* (1 .- a)
                    v3, r3, _ = integrate_pdf_rect(rng, C; a=a, b=b)
                    @test isapprox(v3, Copulas.measure(C, a, b); atol=10*sqrt(r3))
                end
            end

            if applicable(rosenblatt, C, spl10) && 
                applicable(inverse_rosenblatt, C, spl10) && 
                !(CT<:Union{Copulas.WCopula, Copulas.MCopula}) && 
                !((CT<:Copulas.GumbelCopula) && (C.G.θ > 50)) && 
                !((CT<:Copulas.FrankCopula{4}) && (C.G.θ > 50)) &&
                !((CT<:Copulas.ArchimedeanCopula) && (typeof(C.G)<:Copulas.WilliamsonGenerator))
                @testset "Testing Rosenblatt and inverse Rosenblatt transforms" begin
                    U = rosenblatt(C, spl1000)
                    for i in 1:(d - 1)
                        for j in (i + 1):d
                            @test corkendall(U[i, :], U[j, :]) ≈ 0.0 atol = 0.1
                        end
                    end
                    @test spl10 ≈ inverse_rosenblatt(C, rosenblatt(C, spl10)) atol=1e-4
                end
            end
        end
    end
end

