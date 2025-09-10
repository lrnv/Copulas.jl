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
    using Roots
    rng = StableRNG(123)

    can_pdf(C::CT) where CT = 
        applicable(Distributions._logpdf, C, ones(length(C),2)./2) &&
        !(CT<:EmpiricalCopula) &&
        !((CT<:ArchimedeanCopula) && length(C)==Copulas.max_monotony(C.G))

    check_rosenblatt(C::CT) where CT = 
        !((CT<:ArchimedeanCopula) && length(C)==Copulas.max_monotony(C.G)) &&
        !((CT<:FrankCopula) && (C.G.θ >= 100)) &&
        !((CT<:GumbelCopula) && (C.G.θ >= 20)) &&
        !(CT<:MCopula{4}) &&
        !(CT<:EmpiricalCopula) &&
        !(CT<:BC2Copula)

    check_corkendall(C::CT) where CT = 
        !((CT<:FrankCopula) && (C.G.θ >= 100)) &&
        !((CT<:GumbelCopula) && (C.G.θ >= 100)) &&
        !(CT<:Union{MCopula, WCopula}) &&
        !(CT<:EmpiricalCopula) &&
        !(CT<:BC2Copula) &&
        !(CT<:CuadrasAugeCopula) &&
        !(CT<:MOCopula)

    is_archimedean_with_generator(C::CT) where CT =
       (CT<:ArchimedeanCopula) && !(typeof(C.G)<:Copulas.WilliamsonGenerator)

    can_integrate_pdf(C::CT) where CT =  
        # This list might be longer than necessary, it should be trimmed.
        can_pdf(C) &&
        !((CT<:ArchimedeanCopula) && (typeof(C.G)<:Copulas.WilliamsonGenerator) && (C.G.d > length(C))) &&
        !((CT<:FrankCopula) && (C.G.θ >= 100)) &&
        !((CT<:GumbelCopula) && (C.G.θ >= 100)) &&
        !((CT<:FGMCopula) && (length(C)==3)) &&
        !(CT<:MCopula) && 
        !(CT<:WCopula) && 
        !(CT<:MOCopula) && 
        !(CT<:CuadrasAugeCopula) && 
        !(CT<:RafteryCopula) && 
        !(CT<:EmpiricalCopula) && 
        !(CT<:BC2Copula)


    can_ad(C::CT) where CT = 
        # This list might be longer than necessary, it should be trimmed.
        can_pdf(C) &&
        !((CT<:ArchimedeanCopula) && (typeof(C.G)<:Copulas.WilliamsonGenerator) && (C.G.d > length(C))) && # discontinuous. 
        !((CT<:FrankCopula) && (C.G.θ >= 100)) && # too extreme
        !((CT<:GumbelCopula) && ((C.G.θ >= 100)) || length(C) > 2) && # too extreme
        !(CT<:MCopula) && # not abs cont
        !(CT<:WCopula) && # not abs cont
        !(CT<:tEVCopula) && # requires derivatives of beta_inc_inv that forwardiff doesnt have. 
        !(CT<:TCopula) && # same
        !(CT<:CuadrasAugeCopula) # discontinuous
        # !(CT<:BC2Copula) && # is discontinuous
        # !(CT<:MOCopula) # discontinuous
        # !(CT<:AsymGalambosCopula) && 
        # !(CT<:GalambosCopula) && 
        # !(CT<:AsymLogCopula) && 
        # !(CT<:AsymMixedCopula) && 
        # !(CT<:GalambosCopula) && 
        # !(CT<:HuslerReissCopula) && 
        # !(CT<:MixedCopula) && 
        # !(CT<:LogCopula)

    function check(C::Copulas.Copula{d}) where d
        @testset "Testing $C" begin
            @info "Testing $C..."

            CT = typeof(C)
            Random.seed!(rng,123)

            Z = SklarDist(C, Tuple(Normal() for i in 1:d))

            spl1 = rand(rng, C)
            spl10 = rand(rng, C, 10)
            spl1000 = rand(rng, C, 1000)
            splZ10 = rand(rng, Z, 10)

            @testset "Basics" begin 
                @testset "Shape and support" begin 
                    @test length(spl1)==d
                    @test size(spl10) == (d,10)
                    @test all(0 .<= spl10 .<= 1)
                    @test all(0 .<= spl1000 .<= 1)
                end

                @testset "CDF boundary and measure" begin 
                    @test iszero(cdf(C,zeros(d)))
                    @test isone(cdf(C,ones(d)))
                    @test 0 <= cdf(C,rand(rng,d)) <= 1
                    @test cdf(Z,zeros(d)) >= 0
                    @test Copulas.measure(C, zeros(d),    ones(d)) ≈ 1
                    @test Copulas.measure(C, ones(d)*0.2, ones(d)*0.4) >= 0
                end

                if d >= 3
                    @testset "Subsetdims" begin
                        sC = Copulas.subsetdims(C,(2,1))
                        @test all(0 .<= cdf(sC, spl10[1:2,:]) .<= 1)
                        @test sC == Copulas.subsetdims(Z,(2,1)).C
                    end
                end

                # Margins uniformity
                if !(CT<:EmpiricalCopula)
                    @testset "Margins uniformity" begin
                        for i in 1:d
                            for val in [0,1,0.5,rand(rng,5)...]
                                u = ones(d)
                                u[i] = val
                                @test cdf(C,u) ≈ val atol=1e-5
                            end
                            u = rand(rng,d)
                            u[i] = 0
                            @test iszero(cdf(C,u))

                            # This pvalue test fails sometimes.. which is normal since its random, but its anoying ^^
                            # @test pvalue(ApproximateOneSampleKSTest(spl1000[i,:], Uniform())) > 0.005 # fails for Frank(4, 0.5), Gumbel(2, 2.98), Gumbel(2, 1.2)... ???
                        end
                    end
                end

                if can_pdf(C)
                    @testset "PDF positivity" begin
                        r10 = pdf(C, spl10)
                        @test pdf(C, zeros(d) .+ 1e-5) >= 0
                        @test pdf(C, ones(d)/2) >= 0
                        @test pdf(C, ones(d) .- 1e-5) >= 0
                        @test all(r10 .>= 0) && all(isfinite.(r10))
                    end 
                end

                @testset "Corkendall coeherency" begin
                    K = corkendall(spl1000')
                    Kth = corkendall(C)
                    @test all(-1 .<= Kth .<= 1)
                    @test all(isapprox.(Kth, K; atol=0.2))
                end
            end

            if can_integrate_pdf(C)
                @testset "Testing pdf integration" begin
                    # 1) ∫_{[0,1]^d} pdf = 1  (hcubature if d≤3; si no, MC)
                    v, r, _ = integrate_pdf_rect(rng, C, zeros(d), ones(d), 10_000, 10_000)
                    @test isapprox(v, 1; atol=max(5*sqrt(r), 1e-3))

                    # 2) ∫_{[0,0.5]^d} pdf = C(0.5,…,0.5)
                    b = ones(d)/2
                    v2, r2, _ = integrate_pdf_rect(rng, C, zeros(d), b, 10_000, 10_000)
                    @test isapprox(v2, cdf(C, b); atol=max(10*sqrt(r2), 1e-3))

                    # 3) random rectangle, compare with measure (cdf based)
                    a = rand(rng, d)
                    b = a .+ rand(rng, d) .* (1 .- a)
                    v3, r3, _ = integrate_pdf_rect(rng, C, a, b, 10_000, 10_000)
                    @test isapprox(v3, Copulas.measure(C, a, b); atol=max(20*sqrt(r3), 1e-3)) || max(v3, Copulas.measure(C, a, b)) < eps(Float64) # wide tolerence, should pass. 
                end
            end

            if check_rosenblatt(C)
                @testset "rosenblatt ∘ inver_rosenblatt = Id" begin
                    @test spl10 ≈ inverse_rosenblatt(C, rosenblatt(C, spl10)) atol=1e-2
                end
            end

            if check_corkendall(C)
                @testset "corkendall ∘ rosenblatt = I" begin
                    U = rosenblatt(C, spl1000)
                    for i in 1:(d - 1)
                        for j in (i + 1):d
                            @test corkendall(U[i, :], U[j, :]) ≈ 0.0 atol = 0.15
                        end
                    end
                end
            end

            @testset "Conditionning" begin
                # Conditioning tests (p = 1), validate against AD ratio and compare fast-paths to fallback
                if d == 2 && !(CT<:Copulas.tEVCopula) && !(CT<:Copulas.EmpiricalCopula)
                    @testset "Condition(2 | 1): Check Distortion vs AD" begin
                        us = (0.2, 0.5, 0.8)
                        for j in 1:2
                            i = 3 - j
                            for v in (0.3, 0.7) # interior points to avoid boundary issues
                                Dd = Copulas.condition(C, j, v)
                                vals = cdf.(Ref(Dd), us)
                                
                                if can_ad(C)
                                    refs = [ad_ratio_biv(C, i, j, ui, v) for ui in us]
                                    for (v,r) in zip(vals, refs)
                                        @test isapprox(v, r, atol=1e-3, rtol=1e-3)
                                    end
                                    # Compare fast path vs generic fallback only if a specialization exists
                                    m_fast = which(Copulas.DistortionFromCop, (CT,                NTuple{1,Int}, NTuple{1,Float64}, Int))
                                    m_gen  = which(Copulas.DistortionFromCop, (Copulas.Copula{2}, NTuple{1,Int}, NTuple{1,Float64}, Int))
                                    if m_fast != m_gen
                                        Dgen = @invoke Copulas.DistortionFromCop(C::Copulas.Copula{2}, (j,), (Float64(v),), i)
                                        vals2 = cdf.(Ref(Dgen), us)
                                        for (v2,r) in zip(vals2, refs)
                                            @test isapprox(v2, r, atol=1e-3, rtol=1e-2)
                                        end
                                        for (v2,v) in zip(vals2, vals)
                                            @test isapprox(v2, v, atol=1e-3, rtol=1e-3)
                                        end
                                    end
                                else
                                    # No pdf available -> use exact checks for Fréchet bounds if applicable
                                    if CT <: Copulas.MCopula
                                        @test all(vals .≈ min.(collect(us) ./ v, 1))
                                    elseif CT <: Copulas.WCopula
                                        @test all(vals .≈ max.(collect(us) .+ v .- 1, 0) ./ v)
                                    end
                                end
                                # Even if we cant AD, let's check the basics on the conditional cdf:
                                @test all(0.0 .<= vals .<= 1.0)
                                @test all(diff(collect(vals)) .>= -1e-10) # Tolerence for rounding error.
                            end
                        end
                    end
                end

                if d > 2 && can_ad(C)
                    @testset "Condition(d|d-1): Check Distortion vs AD" begin
                        # Spot-check a single index pair (i,j) using the 2D projection via subsetdims
                        j, i, v, us = 1, 2, 0.6, (0.2, 0.5, 0.8)

                        # Distortion for Ui | Uj=v computed from full model
                        Dᵢ = condition(C, j, v)
                        vals = cdf.(Ref(Dᵢ), us)
                        @test all(0.0 .<= vals .<= 1.0)
                        @test all(diff(vals) .>= -1e-10) # increasingness with tolerence.

                        # Validate marginal distortion against AD on the 2D subset (i maps to 1, j maps to 2)
                        Cproj = Copulas.subsetdims(C, (i, j))
                        if can_ad(Cproj)
                            refs = [ad_ratio_biv(Cproj, 1, 2, u, v) for u in us]
                            for (v,r) in zip(vals, refs)
                                @test isapprox(v, r, atol=1e-5, rtol=1e-5)
                            end
                        end
                    end

                    @testset "Condition (d|d-2): Check conditional copula vs AD" begin
                        js = collect(3:d)
                        ujs = [0.25 + 0.5*rand(rng) for _ in js]  # interior values
                        CC = condition(C, js, ujs)

                        # Local AD-based reference for this (C, js, ujs)
                        js_t = Tuple(js); ujs_t = Tuple(ujs)
                        # Small epsilon to avoid boundary artifacts at 1.0
                        EPS = 1e-9
                        Hi_local = function (i::Int, u::Float64)
                            num = _der(uvec -> cdf(C, uvec), _assemble(d, (i,), js_t, (u,), ujs_t), js_t)
                            den = _der(uvec -> cdf(C, uvec), _assemble(d, (i,), js_t, (1.0 - EPS,), ujs_t), js_t)
                            return num / den
                        end
                        invHi_local = function (i::Int, α::Float64)
                            f(u) = Hi_local(i, u) - α
                            a, b = EPS, 1.0 - EPS
                            try
                                return Roots.find_zero(f, (a, b), Roots.Brent(); xatol=1e-10, atol=1e-10)
                            catch
                                # Fallback to bisection if Brent fails to bracket
                                for _ in 1:80
                                    m = (a + b) / 2
                                    (Hi_local(i, m) < α) ? (a = m) : (b = m)
                                end
                                return (a + b) / 2
                            end
                        end
                        H12_local = function (u1::Float64, u2::Float64)
                            num = _der(uvec -> cdf(C, uvec), _assemble(d, (1,2), js_t, (u1,u2), ujs_t), js_t)
                            den = _der(uvec -> cdf(C, uvec), _assemble(d, (1,2), js_t, (1.0 - EPS, 1.0 - EPS), ujs_t), js_t)
                            return num / den
                        end

                        # test grid on [0,1]^2
                        pts = [(0.2,0.3), (0.5,0.5), (0.8,0.6)]
                        for (v1,v2) in pts
                            val_fast = cdf(CC, [v1, v2]) # our implementation
                            val_ref = H12_local(invHi_local(1, v1), invHi_local(2, v2)) # AD reference:
                            @test isapprox(val_fast, val_ref; atol=5e-5, rtol=5e-5)
                        end
                        # compare specialized ConditionalCopula vs generic fallback when specialization exists
                        let m_fast = which(Copulas.ConditionalCopula, (CT,                Any, Any)),
                            m_gen  = which(Copulas.ConditionalCopula, (Copulas.Copula{d}, Any, Any))
                            if m_fast != m_gen
                                CC_gen = @invoke Copulas.ConditionalCopula(C::Copulas.Copula{d}, js, ujs)
                                for (v1,v2) in pts
                                    @test cdf(CC, [v1,v2]) ≈ cdf(CC_gen, [v1,v2]) atol=1e-8 rtol=1e-8
                                end
                            end
                        end
                    end
                end
            end
            
            if is_archimedean_with_generator(C)
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
                            for k in 1:d-1
                                @test Copulas.ϕ⁽ᵏ⁾⁻¹(C.G,Val{k}(), Copulas.ϕ⁽ᵏ⁾(C.G, Val{k}(), 1.2)) ≈ 1.2
                            end
                        end
                    end

                    if can_τinv
                        @testset "Check τ ∘ τ⁻¹ == Id" begin
                            tau = Copulas.τ(C)
                            @test Copulas.τ(GT(Copulas.τ⁻¹(CT,tau))) ≈ tau
                        end
                    end
                    
                    if can_ρinv
                        @testset  "Check ρ ∘ ρ⁻¹ == Id" begin
                            rho = Copulas.ρ(C)
                            @test -1 <= rho <= 1
                            @test Copulas.ρ(GT(Copulas.ρ⁻¹(CT,rho))) ≈ rho
                        end
                    end

                    if C.G isa Copulas.FrailtyGenerator
                        F = frailty(C.G)
                        spe_ϕ = which(Copulas.ϕ, (typeof(C.G), Float64)) != which(Copulas.ϕ, (Copulas.FrailtyGenerator, Float64))
                        if spe_ϕ && applicable(mgf, F, -1.0)
                            @testset "Check frailty matches ϕ" begin
                                for t in 0:0.1:2
                                    @test ϕ(C.G, t) == mgf(F, -t)
                                end
                            end
                        end
                    end

                    @testset "Kendall-Radial coherency test" begin
                        # On radial-level: 
                        R1 = dropdims(sum(Copulas.ϕ⁻¹.(C.G,spl1000),dims=1),dims=1)
                        R2 = rand(rng,Copulas.williamson_dist(C.G, Val{d}()),1000)
                        @test pvalue(ApproximateTwoSampleKSTest(R1,R2)) > 0.005

                        # On kendall-level: 
                        U1 = Distributions.cdf(C, spl1000)
                        U2 = Copulas.ϕ.(Ref(C.G), rand(rng,Copulas.williamson_dist(C.G, Val{d}()),1000))
                        @test pvalue(ApproximateTwoSampleKSTest(U1, U2)) > 0.005
                    end
                end
            end

            @testset "Check Fitting the copula" begin
                # This will largely change soon when the new fitting interface comes aroud. 
                # For the moement the testing interface is mostly Ok for some copulas, and completely broken for all others. 
                # We could do much better at it. 
                try 
                    fit(CT,spl10) # sometimes fails for non-monotonicity of the obtained generator. 
                    # this is something we should fix :)
                catch
                end
            end

            # Extreme value copula-specific tests
            if CT<:Copulas.ExtremeValueCopula{2} # only bivariates !!! 
                @testset "ExtremeValueCopula specific tests" begin
                    @testset "A function basics" begin
                        @test Copulas.A(C.tail, 0.0) ≈ 1
                        @test Copulas.A(C.tail, 1.0) ≈ 1
                        t = rand(rng)
                        A_value = Copulas.A(C.tail, t)
                        @test 0.0 <= A_value <= 1.0
                        @test isapprox(A_value, max(t, 1-t); atol=1e-6) || A_value >= max(t, 1-t)
                        @test A_value <= 1.0
                    end

                    # Only run derivative and related tests if the methods are specialized for this type
                    spe_dA =        which(Copulas.dA, (typeof(C.tail), Float64))        != which(Copulas.dA, (Copulas.Tail2, Float64))
                    spe_d²A =       which(Copulas.d²A, (typeof(C.tail), Float64))       != which(Copulas.d²A, (Copulas.Tail2, Float64))
                    spe__A_dA_d²A = which(Copulas._A_dA_d²A, (typeof(C.tail), Float64)) != which(Copulas._A_dA_d²A, (Copulas.Tail2, Float64))
                    spe_ℓ =         which(Copulas.ℓ, (typeof(C.tail), Tuple{Float64, Float64}))         != which(Copulas.ℓ, (Copulas.Tail2, Tuple{Float64, Float64}))

                    if spe_dA || spe_d²A || spe__A_dA_d²A
                        @testset "Testing derivatives of A" begin
                            for t in (0.05, 0.5, 0.95)
                                if !(CT<:tEVCopula)
                                    @test isapprox(Copulas.dA(C.tail, t), ForwardDiff.derivative(x -> Copulas.A(C.tail, x), t); atol=1e-6)
                                    @test isapprox(Copulas.d²A(C.tail, t), ForwardDiff.derivative(x -> Copulas.dA(C.tail, x), t); atol=1e-6)
                                end
                                a, da, d2a = Copulas._A_dA_d²A(C.tail, t)
                                @test isapprox(a, Copulas.A(C.tail, t); atol=1e-8)
                                @test isapprox(da, Copulas.dA(C.tail, t); atol=1e-8)
                                @test isapprox(d2a, Copulas.d²A(C.tail, t); atol=1e-8)
                            end
                        end
                    end

                    if spe_dA || spe_d²A || spe__A_dA_d²A || spe_ℓ
                        @testset "Testing ℓ and cdf for Extreme Value Copula" begin 
                            u, v = rand(rng), rand(rng)
                            x, y = -log(u), -log(v)
                            s = y / (x + y)
                            expected_ℓ = Copulas.A(C.tail, s) * (x + y)
                            @test isapprox(Copulas.ℓ(C.tail, x, y), expected_ℓ; atol=0.1)
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

            # Archimax specific tests
            if CT<:Copulas.ArchimaxCopula
                @testset "ArchimaxCopula specific tests" begin

                    for (u1,u2) in ((0.2,0.3), (0.7,0.6), (0.9,0.4))
                        @test isapprox(cdf(C, [u1,u2]), _archimax_cdf_mockup(C, u1, u2); rtol=1e-12, atol=1e-12)
                    end

                    for (u1,u2) in ((0.25,0.4), (0.6,0.6))
                        lp = logpdf(C, [u1,u2])
                        @test isfinite(lp)
                        c_h = _archimax_pdf_hess(C, u1, u2)
                        @test isapprox(exp(lp), c_h; rtol=1e-6, atol=1e-8)
                    end

                    for r in _archimax_mc_rectangles_cdf(C; N=20_000, seed=321)
                        @test abs(r.p_hat - r.p_th) ≤ max(5*r.se, 2e-3)
                    end
                end
            end
        end
    end

    _archimax_cdf_mockup(C::Copulas.ArchimaxCopula, u1::Real, u2::Real) = begin
        G, E = C.gen, C.tail
        (u1≤0 || u2≤0)  && return 0.0
        (u1≥1 && u2≥1)  && return 1.0
        x = Copulas.ϕ⁻¹(G, u1); y = Copulas.ϕ⁻¹(G, u2)
        S = x + y
        S == 0 && return 1.0
        t = y / S
        Copulas.ϕ(G, S * Copulas.A(E, t))
    end
    _archimax_pdf_hess(C, u1, u2) = begin
        f(z) = cdf(C, z)                      
        H = ForwardDiff.hessian(f, [u1, u2])  # ∂²/∂u1∂u2
        max(H[1,2], 0.0)                      # numerical clip 
    end
    function _archimax_mc_rectangles_cdf(C; N::Int=300_000, seed::Integer=123,
                            rects::Tuple{Vararg{Tuple{<:Real,<:Real}}}=((0.5,0.5),(0.3,0.7),(0.8,0.2)))
        rng = StableRNG(seed)
        U = rand(rng, C, N)
        results = Vector{NamedTuple}(undef, length(rects))
        @inbounds for (k,(a,b)) in pairs(rects)
            p_th = cdf(C, [a,b])
            cnt = 0
            for i in 1:N
                (U[1,i] ≤ a && U[2,i] ≤ b) && (cnt += 1)
            end
            p_hat = cnt / N
            se = sqrt(p_th*(1-p_th)/N)
            results[k] = (rect=(a,b), p_hat=p_hat, p_th=p_th, se=se)
        end
        return results
    end

    function integrate_pdf_rect(rng, C::Copulas.Copula{d}, a, b, maxevals, N) where d
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

    # AD-based reference for the univariate conditional CDF H_{i|j}(u | U_j=v)
    # Theory (Sklar): H_{i|j}(u|v) = (∂_j C)(..., u_i=u, u_j=v) / (∂_j C)(..., u_i=1, u_j=v).
    function ad_ratio_biv(C2::Copulas.Copula{2}, i::Int, j::Int, u::Float64, v::Float64)
        # For a bivariate copula, the denominator is always one.
        @assert (i == 1 && j == 2) || (i == 2 && j == 1)
        if j == 2
            # Condition on dim 2 at v: differentiate w.r.t arg2
            return ForwardDiff.derivative(t -> cdf(C2, [u, t]), v)
        else # j == 1
            # Condition on dim 1 at v: differentiate w.r.t arg1
            return ForwardDiff.derivative(t -> cdf(C2, [t, u]), v)
        end
    end
    # helpers to compute AD-based reference for H_{I|J}
    function _assemble(Dtot, is, js_, uis, ujs_)
        # Determine an element type compatible with Duals by promoting all inputs
        T = Float64
        for x in uis;   T = promote_type(T, typeof(x)); end
        for x in ujs_;  T = promote_type(T, typeof(x)); end
        w = fill(one(T), Dtot)
        @inbounds for (k,ii) in pairs(is);  w[ii] = uis[k];  end
        @inbounds for (k,jj) in pairs(js_); w[jj] = ujs_[k]; end
        return w
    end
    # Promote vector element type when replacing index i with a dual-valued x
    function _swap_promote(u::AbstractVector, i::Int, x)
        T = promote_type(eltype(u), typeof(x))
        v = Vector{T}(undef, length(u))
        @inbounds for k in eachindex(u); v[k] = u[k]; end
        v[i] = x
        return v
    end
    function _der(f, u::AbstractVector, idxs::Tuple{Vararg{Int}})
        if length(idxs) == 1
            i = idxs[1]
            return ForwardDiff.derivative(x -> f(_swap_promote(u, i, x)), u[i])
        else
            return _der(u_ -> _der(f, u_, (idxs[end],)), u, idxs[1:end-1])
        end
    end
    # Note: Global AD reference helpers Hi/invHi/H12 were removed; each test that
    # needs them now defines local closures bound to its (C, js, ujs) context.

end

