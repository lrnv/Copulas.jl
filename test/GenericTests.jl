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

    function approx_measure_mc(rng, C::Copulas.Copula{d}, a, b, N) where d
        ba = b .- a
        logvol = log(prod(ba))
        logS  = -Inf
        logS2 = -Inf
        u = zeros(d)
        x = similar(a)
        @inbounds for _ in 1:N
            rand!(rng, u)
            x .=  a .+ ba .* u
            lp = logpdf(C, x)
            if isfinite(lp)
                log_fx = lp + logvol
                logS   = LogExpFunctions.logaddexp(logS,  log_fx)
                logS2  = LogExpFunctions.logaddexp(logS2, 2 * log_fx)
            end
        end
        μ  = exp(logS  - log(N))
        m2 = exp(logS2 - log(N))
        r  = max(m2 - μ^2, 0.0) / N
        return μ, r, :mc_pdf
    end
    function approx_measure_hcub(rng, C::Copulas.Copula{d}, a, b, maxevals) where d
        v, abs_err = hcubature(x -> pdf(C, x), a, b; maxevals=maxevals)
        return v, abs_err^2, :hcub
    end
    function integrate_pdf_rect(rng, C::Copulas.Copula{d}, a, b, maxevals, N) where d
        if d == 2
            try
                v, r, how = approx_measure_hcub(rng, C, a, b, maxevals)
                v_true = Copulas.measure(C, a, b)
                isapprox(v, v_true; atol=10 * sqrt(r)) && return v, r, how
            catch
            end
        end
        return approx_measure_mc(rng, C, a, b, N) 
    end

    is_absolutely_continuous(C::CT) where CT =  
        !(CT<:Union{MCopula,WCopula,MOCopula,CuadrasAugeCopula,RafteryCopula,EmpiricalCopula, BC2Copula}) && 
        !((CT<:FGMCopula) && (length(C)==3)) &&
        !((CT<:FrankCopula) && (C.G.θ >= 100)) &&
        !((CT<:GumbelCopula) && (C.G.θ >= 100)) &&
        !((CT<:ArchimedeanCopula) && length(C)>=Copulas.max_monotony(C.G))

    check_rosenblatt(C::CT) where CT = 
        !(CT<:Union{WCopula, MCopula}) && 
        !((CT<:GumbelCopula) && ((C.G.θ > 50) || length(C)>=4)) && 
        !((CT<:FrankCopula{4}) && (C.G.θ > 50)) &&
        !((CT<:ArchimedeanCopula) && (typeof(C.G)<:Copulas.WilliamsonGenerator))

    
    is_archimedean_with_agenerator(C::CT) where CT =
        (CT<:ArchimedeanCopula) && (typeof(C.G)<:Copulas.WilliamsonGenerator)

    has_pdf(C::CT) where CT = applicable(Distributions._logpdf, C, ones(length(C),2)./2)
    
    function check(C::Copulas.Copula{d}) where d

        @testset "Testing $C" begin
            @info "Testing $C..."
            CT = typeof(C)
            Random.seed!(rng,123)

            spl10 = rand(rng,C,10)
            spl1000 = rand(rng,C,1000)

            D = SklarDist(C, Tuple(LogNormal() for i in 1:d))
            splD10 = rand(rng, D, 10)
            
            @testset "Shape and support" begin 
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
                @test isa(Copulas.subsetdims(D,1), Distributions.LogNormal)
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
                    @test pdf(D, ones(d)) >= 0
                end
            end

            # Generic tests
            @testset "CorKendall coeherency" begin
                K = corkendall(spl1000')
                Kth = corkendall(C)
                @test all(-1 .<= Kth .<= 1)
                @test all(isapprox.(Kth, K; atol=0.2))
            end

            if is_absolutely_continuous(C) && has_pdf(C)
                @testset "Testing pdf integration" begin

                    # 1) ∫_{[0,1]^d} pdf = 1  (hcubature if d≤3; si no, MC)
                    v, r, _ = integrate_pdf_rect(rng, C, zeros(d), ones(d), 10_000, 100_000)
                    v_true = 1
                    @test isapprox(v, v_true; atol=5*sqrt(r))

                    # 2) ∫_{[0,0.5]^d} pdf = C(0.5,…,0.5)
                    b = ones(d)/2
                    v2, r2, _ = integrate_pdf_rect(rng, C, zeros(d), b, 10_000, 100_000)
                    v2_true = cdf(C, b)
                    @test isapprox(v2, v2_true; atol=10*sqrt(r2))

                    # 3) random rectangle, compare with measure (cdf based)
                    a = rand(rng, d)
                    b = a .+ rand(rng, d) .* (1 .- a)
                    v3, r3, _ = integrate_pdf_rect(rng, C, a, b, 10_000, 100_000)
                    v3_true = Copulas.measure(C, a, b)
                    @test isapprox(v3, v3_true; atol=20*sqrt(r3)) || max(v3, v3_true) < eps(Float64) # wide tolerence, should pass. 
                end
            end

            if check_rosenblatt(C) && applicable(rosenblatt, C, spl10) && applicable(inverse_rosenblatt, C, spl10) 
                @testset "rosenblatt ∘ inver_rosenblatt = Id" begin
                    U = rosenblatt(C, spl1000)
                    for i in 1:(d - 1)
                        for j in (i + 1):d
                            @test corkendall(U[i, :], U[j, :]) ≈ 0.0 atol = 0.1
                        end
                    end
                    @test spl10 ≈ inverse_rosenblatt(C, rosenblatt(C, spl10)) atol=1e-4
                    @test splD10 ≈ inverse_rosenblatt(D, rosenblatt(D, splD10)) atol=0.1 # also on the sklar level. 
                end

            end

            if is_archimedean_with_agenerator(CT)
                @testset "ArchimedeanCopula specific tests" begin 
                
                    # Only test things if there are specilized versions of the functions. 
                    spe_ϕ1 = which(Copulas.ϕ⁽¹⁾, (typeof(C.G), Float64)) != which(Copulas.ϕ⁽¹⁾, (Copulas.Generator, Float64))
                    spe_ϕk = which(Copulas.ϕ⁽ᵏ⁾, (typeof(C.G), Val{1}, Float64)) != which(Copulas.ϕ⁽ᵏ⁾, (Copulas.Generator, Val{1}, Float64))
                    spe_ϕinv = which(Copulas.ϕ⁻¹, (typeof(C.G), Float64)) != which(Copulas.ϕ⁻¹, (Copulas.Generator, Float64))
                    spe_ϕinv1 = which(Copulas.ϕ⁻¹⁽¹⁾, (typeof(C.G), Float64)) != which(Copulas.ϕ⁻¹⁽¹⁾, (Copulas.Generator, Float64))
                    spe_ϕkinv = which(Copulas.ϕ⁽ᵏ⁾⁻¹, (typeof(C.G), Val{1}, Float64)) != which(Copulas.ϕ⁽ᵏ⁾⁻¹, (Copulas.Generator, Val{1}, Float64))
                    
                    can_τinv = applicable(Copulas.τ, C.G) && applicable(Copulas.τ⁻¹,typeof(C.G), 1.0)
                    can_ρinv = applicable(Copulas.ρ, C.G) && applicable(Copulas.ρ⁻¹,typeof(C.G), 1.0)
                    GT = Copulas.generatorof(CT)

                    if spe_ϕinv
                        @testset "Check ϕ ∘ ϕ⁻¹ == Id over [0,1]" begin
                            for x in 0:0.1:1
                                @test Copulas.ϕ(C.G,Copulas.ϕ⁻¹(C.G,x)) ≈ x atol=1e-10
                            end
                        end
                    end

                    if spe_ϕ1
                        @testset "Check d(ϕ) == ϕ⁽¹⁾" begin 
                            @test ForwardDiff.derivative(x -> Copulas.ϕ(C.G, x), 1.2) ≈ Copulas.ϕ⁽¹⁾(C.G, 1.2)
                        end
                    end

                    if spe_ϕk
                        @testset "Check ϕ == ϕ⁽ᵏ⁾(k=0)" begin
                            @test Copulas.ϕ(C.G, 1.2) ≈ Copulas.ϕ⁽ᵏ⁾(C.G, Val{0}(), 1.2)
                        end

                        @testset "Check d(ϕ) == ϕ⁽ᵏ⁾(k=1)" begin
                            @test ForwardDiff.derivative(x -> Copulas.ϕ(C.G, x), 1.2) ≈ Copulas.ϕ⁽ᵏ⁾(C.G, Val{1}(), 1.2)
                        end
                    end

                    if spe_ϕ1 || spe_ϕk 
                        @testset "Check ϕ⁽¹⁾ == ϕ⁽ᵏ⁾(k=1)" begin
                            @test Copulas.ϕ⁽¹⁾(C.G, 1.2) ≈ Copulas.ϕ⁽ᵏ⁾(C.G, Val{1}(), 1.2)
                        end
                        @testset "Check d(ϕ⁽¹⁾) == ϕ⁽ᵏ⁾(k=2)" begin
                            @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(C.G, x), 1.2) ≈ Copulas.ϕ⁽ᵏ⁾(C.G, Val{2}(), 1.2)
                        end
                    end
                    
                    if spe_ϕinv1 
                        @testset "Check d(ϕ⁻¹) == ϕ⁻¹⁽¹⁾" begin
                            @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(C.G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(C.G, 0.5)
                        end
                    end

                    if spe_ϕkinv 
                        @testset "Check ϕ⁽ᵏ⁾⁻¹ ∘ ϕ⁽ᵏ⁾ == Id for k in 1:d-1" begin
                            for k in 0:d-1
                                @test Copulas.ϕ⁽ᵏ⁾⁻¹(C.G,Val{k}(), Copulas.ϕ⁽ᵏ⁾(C.G, Val{k}(), 1.2)) ≈ 1.2
                            end
                        end
                    end

                    if can_τinv
                        @testset "Check τ ∘ τ⁻¹ == Id" begin
                            tau = Copulas.τ(C)
                            @test Copulas.τ(GT(Copulas.τ⁻¹(CT,tau))) ≈ tau
                        end
                        @testset "Check fitting the copula" begin
                            fit(CT,spl10)
                        end
                    end
                    
                    if can_ρinv
                        @testset  "Check ρ ∘ ρ⁻¹ == Id" begin
                            rho = Copulas.ρ(C)
                            @test -1 <= rho <= 1
                            @test Copulas.ρ(GT(Copulas.ρ⁻¹(CT,rho))) ≈ rho
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
                @testset "ExtremeValueCopula specific tests" begin
                    @testset "A function basics" begin
                        @test Copulas.A(C, 0.0) ≈ 1
                        @test Copulas.A(C, 1.0) ≈ 1
                        t = rand(rng)
                        A_value = Copulas.A(C, t)
                        @test 0.0 <= A_value <= 1.0
                        @test isapprox(A_value, max(t, 1-t); atol=1e-6) || A_value >= max(t, 1-t)
                        @test A_value <= 1.0
                    end

                    # Only run derivative and related tests if the methods are spe for this type
                    spe_dA =        which(Copulas.dA, (typeof(C), Float64))        != which(Copulas.dA, (Copulas.ExtremeValueCopula, Float64))
                    spe_d²A =       which(Copulas.d²A, (typeof(C), Float64))       != which(Copulas.d²A, (Copulas.ExtremeValueCopula, Float64))
                    spe__A_dA_d²A = which(Copulas._A_dA_d²A, (typeof(C), Float64)) != which(Copulas._A_dA_d²A, (Copulas.ExtremeValueCopula, Float64))
                    spe_ℓ =         which(Copulas.ℓ, (typeof(C), Float64, Float64))         != which(Copulas.ℓ, (Copulas.ExtremeValueCopula, Float64, Float64))

                    if spe_dA || spe_d²A || spe__A_dA_d²A
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

                    if spe_dA || spe_d²A || spe__A_dA_d²A || spe_ℓ
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
        end
    end
end

