@testitem "Generic" tags=[:BBCops,:BB1Copula] setup=[M] begin M.check(BB1Copula(2, 1.2, 1.5)) end
@testitem "Generic" tags=[:BBCops,:BB1Copula] setup=[M] begin M.check(BB1Copula(2, 2.5, 1.5)) end
@testitem "Generic" tags=[:BBCops,:BB1Copula] setup=[M] begin M.check(BB1Copula(2, 0.35, 1.0)) end

#@testitem "Generic" tags=[:BBCops,:BB2Copula] setup=[M] begin M.check(BB2Copula(2, 1.2, 0.5)) end
#@testitem "Generic" tags=[:BBCops,:BB2Copula] setup=[M] begin M.check(BB2Copula(2, 1.5, 1.8)) end
#@testitem "Generic" tags=[:BBCops,:BB2Copula] setup=[M] begin M.check(BB2Copula(2, 2.1, 1.5)) end

#@testitem "Generic" tags=[:BBCops,:BB3Copula] setup=[M] begin M.check(BB3Copula(2, 2.0, 1.6)) end
#@testitem "Generic" tags=[:BBCops,:BB3Copula] setup=[M] begin M.check(BB3Copula(2, 2.5, 0.4)) end
#@testitem "Generic" tags=[:BBCops,:BB3Copula] setup=[M] begin M.check(BB3Copula(2, 5.0, 0.1)) end

# @testitem "Generic" tags=[:BBCops,:BB4Copula] setup=[M] begin M.check(BB4Copula(2, 0.5, 1.6)) end
# @testitem "Generic" tags=[:BBCops,:BB4Copula] setup=[M] begin M.check(BB4Copula(2, 2.5, 0.4)) end
# @testitem "Generic" tags=[:BBCops,:BB4Copula] setup=[M] begin M.check(BB4Copula(2, 3.0, 2.1)) end

# @testitem "Generic" tags=[:BBCops,:BB5Copula] setup=[M] begin M.check(BB5Copula(2, 1.5, 1.6)) end
# @testitem "Generic" tags=[:BBCops,:BB5Copula] setup=[M] begin M.check(BB5Copula(2, 2.5, 0.4)) end
# @testitem "Generic" tags=[:BBCops,:BB5Copula] setup=[M] begin M.check(BB5Copula(2, 5.0, 0.5)) end

@testitem "Generic" tags=[:BBCops,:BB6Copula] setup=[M] begin M.check(BB6Copula(2, 1.2, 1.6)) end
@testitem "Generic" tags=[:BBCops,:BB6Copula] setup=[M] begin M.check(BB6Copula(2, 1.5, 1.4)) end
@testitem "Generic" tags=[:BBCops,:BB6Copula] setup=[M] begin M.check(BB6Copula(2, 2.0, 1.5)) end

@testitem "Generic" tags=[:BBCops,:BB7Copula] setup=[M] begin M.check(BB7Copula(2, 1.2, 1.6)) end
@testitem "Generic" tags=[:BBCops,:BB7Copula] setup=[M] begin M.check(BB7Copula(2, 1.5, 0.4)) end
@testitem "Generic" tags=[:BBCops,:BB7Copula] setup=[M] begin M.check(BB7Copula(2, 2.0, 1.5)) end

@testitem "Generic" tags=[:BBCops,:BB8Copula] setup=[M] begin M.check(BB8Copula(2, 1.2, 0.4)) end
@testitem "Generic" tags=[:BBCops,:BB8Copula] setup=[M] begin M.check(BB8Copula(2, 1.5, 0.6)) end
@testitem "Generic" tags=[:BBCops,:BB8Copula] setup=[M] begin M.check(BB8Copula(2, 2.5, 0.8)) end

@testitem "Generic" tags=[:BBCops,:BB9Copula] setup=[M] begin M.check(BB9Copula(2, 2.8, 2.6)) end
@testitem "Generic" tags=[:BBCops,:BB9Copula] setup=[M] begin M.check(BB9Copula(2, 1.5, 2.4)) end
@testitem "Generic" tags=[:BBCops,:BB9Copula] setup=[M] begin M.check(BB9Copula(2, 2.0, 1.5)) end

@testitem "Generic" tags=[:BBCops,:BB10Copula] setup=[M] begin M.check(BB10Copula(2, 1.5, 0.7)) end
@testitem "Generic" tags=[:BBCops,:BB10Copula] setup=[M] begin M.check(BB10Copula(2, 4.5, 0.6)) end
@testitem "Generic" tags=[:BBCops,:BB10Copula] setup=[M] begin M.check(BB10Copula(2, 3.0, 0.8)) end

@testitem "Bivariate BB_specific" tags=[:BBCops] begin
# ===================== test/BB_specific.jl =====================
    using Test
    using Random, StatsBase, HypothesisTests
    using Distributions, ForwardDiff, HCubature
    using StableRNGs
    using LogExpFunctions
    using Copulas

    const _EPS = 1e-12

    function rand_clamped(rng, C, n; ϵ=_EPS)
        U = rand(rng, C, n)
        @inbounds @views (U .= clamp.(U, ϵ, 1-ϵ))
        return U
    end

# ---------- Integration with warp specific to BB2/BB3 (peaks on axes and corners) ----------
# --- CDF MC on rectangles: P(U1≤a, U2≤b) ≈ C(a,b) ---
    function mc_rectangles_cdf(C; N::Int=300_000, seed::Integer=123,
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

    function mc_pdf_integral(C; N::Int=2*10^6, seed::Integer=42)
        rng = StableRNG(seed)
        d = length(C)
        u = Vector{Float64}(undef, d)

        logS  = -Inf
        logS2 = -Inf
        n_eff = 0

        @inbounds for _ in 1:N
            rand!(rng, u)
            lp = logpdf(C, u)
            if isfinite(lp)
                logS  = LogExpFunctions.logaddexp(logS,  lp)
                logS2 = LogExpFunctions.logaddexp(logS2, 2lp)
                n_eff += 1
            end
        end
        n_eff == 0 && error("mc_pdf_integral: There were no finite evaluations.")

        Ĩ  = exp(logS  - log(n_eff))
        m2 = exp(logS2 - log(n_eff))
        var̂ = max(m2 - Ĩ^2, 0.0)
        SE = sqrt(var̂ / n_eff)
        return Ĩ, SE, n_eff
    end

    function check_generator_calculus!(G; atol=1e-10)
        xs = range(1e-12, 1-1e-12; length=11)

        # ϕ ∘ ϕ⁻¹ ≈ Id
        f(x) = Copulas.ϕ⁻¹(G, Copulas.ϕ(G, x))
        vals = f.(xs)
        @test maximum(abs.(vals .- xs)) ≤ atol

        if which(Copulas.ϕ⁽¹⁾, (typeof(G), Float64)) != which(Copulas.ϕ⁽¹⁾, (Copulas.Generator, Float64))
            for s in (0.1, 0.5, 1.2, 3.0)
                ad = ForwardDiff.derivative(x->Copulas.ϕ(G,x), s)
                @test isapprox(ad, Copulas.ϕ⁽¹⁾(G,s); atol=1e-12, rtol=1e-10)
            end
        end
        if hasmethod(Copulas.ϕ⁽ᵏ⁾, (typeof(G), Float64))
            for s in (0.1, 0.5, 1.2, 3.0)
                ad2 = ForwardDiff.derivative(x->ForwardDiff.derivative(y->Copulas.ϕ(G,y), x), s)
                @test isapprox(ad2, Copulas.ϕ⁽ᵏ⁾(G, Val{2}(), s); atol=1e-10, rtol=1e-8)
            end
        end
        if which(Copulas.ϕ⁻¹⁽¹⁾, (typeof(G), Float64)) != which(Copulas.ϕ⁻¹⁽¹⁾, (Copulas.Generator, Float64))
            for t in (0.2, 0.5, 0.8)
                ad = ForwardDiff.derivative(x->Copulas.ϕ⁻¹(G,x), t)
                @test isapprox(ad, Copulas.ϕ⁻¹⁽¹⁾(G,t); atol=1e-12, rtol=1e-10)
            end
        end
    end

    function williamson_KS_pvalue(C; n=10000, rng=StableRNG(123))
        U = rand_clamped(rng, C, n)
        w1 = vec(dropdims(sum(Copulas.ϕ⁻¹.(C.G, U), dims=1), dims=1))
        w2 = rand(rng, Copulas.williamson_dist(C.G, Val(2)), n)
        pvalue(ApproximateTwoSampleKSTest(w1, w2), tail=:both)
    end

    function kendall_emp_vs_teo(C; n=5000, rng=StableRNG(123))
        U = rand(rng, C, n)
        τ_emp = corkendall(U')[1,2]
        τ_th = try
            Copulas.τ(C)[1,2]
        catch
            Ubig = rand(rng, C, 50_000)
            corkendall(Ubig')[1,2]
        end
        return τ_emp, τ_th
    end

    # ------------------ families and parameters to test ------------------

    const BB_PARAMS = Dict(
        :BB1 => [(1.20, 1.50), (2.50, 3.00), (0.35, 1.00)],
        :BB2 => [(1.20, 0.50), (1.5, 1.80), (2.1, 1.5)],
        :BB3 => [(2.00, 1.60), (2.50, 0.40), (5.0, 0.1)],
        # :BB4 => [(0.50, 1.60), (2.50, 0.40), (3.0, 2.1)],
        # :BB5 => [(1.50, 1.60), (2.50, 0.40), (5.0, 0.5)], 
        :BB6 => [(1.20, 1.60), (1.50, 1.40), (2.0, 1.5)],
        :BB7 => [(1.20, 1.60), (1.50, 0.40), (2.0, 0.5)],
        :BB8 => [(1.20, 0.40), (1.50, 0.60), (2.5, 0.8)],
        :BB9 => [(2.80, 2.60), (1.50, 2.40), (2.0, 1.5)],
        :BB10 => [(1.50, 0.70), (4.50, 0.60), (3.0, 0.8)],
    )

    # map constructor
    const BB_CTOR = Dict(
        :BB1  => (θ,δ) -> BB1Copula(2, θ,δ),
        :BB2  => (θ,δ) -> BB2Copula(2, θ,δ),
        :BB3  => (θ,δ) -> BB3Copula(2, θ,δ),
        # :BB4  => (θ,δ) -> BB4Copula(2, θ,δ),  
        # :BB5  => (θ,δ) -> BB5Copula(2, θ,δ), 
        :BB6  => (θ,δ) -> BB6Copula(2, θ,δ),
        :BB7  => (θ,δ) -> BB7Copula(2, θ,δ),
        :BB8  => (θ,δ) -> BB8Copula(2, θ,δ),
        :BB9  => (θ,δ) -> BB9Copula(2, θ,δ),
        :BB10 => (θ,δ) -> BB10Copula(2, θ,δ),
    )

    @testset "Bivariate BB-specific" begin
        for fam in (:BB1, :BB2, :BB3, :BB6, :BB7, :BB8, :BB9, :BB10,)
            @testset "BB $(fam)" begin
                for (θ,δ) in BB_PARAMS[fam]
                    C = BB_CTOR[fam](θ,δ)
                    G = C.G
                    @test length(C) == 2

                    @testset "ϕ/ϕ⁻¹ y derivadas | θ=$(θ), δ=$(δ)" begin
                        check_generator_calculus!(G)
                        @test isapprox(Copulas.ϕ⁻¹(G, Copulas.ϕ(G, 0.0)), 0.0; atol=1e-14)
                        @test isapprox(Copulas.ϕ(G, Copulas.ϕ⁻¹(G, 1.0)), 1.0; atol=1e-14)
                    end

                    @testset "Kendall | θ=$(θ), δ=$(δ)" begin
                        τ_emp, τ_th = kendall_emp_vs_teo(C)
                        @test -1 ≤ τ_th ≤ 1
                        @test isapprox(τ_emp, τ_th; atol=0.02)
                    end

                    @testset "Williamson | θ=$(θ), δ=$(δ)" begin
                        p = williamson_KS_pvalue(C)
                        if θ ≤ 0.4
                            p = williamson_KS_pvalue(C; n=3000)
                            @test p > 0.005
                        else
                            @test p > 0.01
                        end
                    end

                    @testset "Propiedades por MC | θ=$(θ), δ=$(δ)" begin
                        if θ*δ < 2.0
                            @testset "PDF (MC uniforme)" begin
                                Ĩ, SE, _ = mc_pdf_integral(C; N=1_000_000, seed=42)
                            @test abs(Ĩ - 1) ≤ max(5*SE, 1e-2)
                        end
                    end

                    @testset "CDF (MC sobre la cópula)" begin
                        for r in mc_rectangles_cdf(C; N=300_000, seed=123)
                            @test abs(r.p_hat - r.p_th) ≤ max(5*r.se, 2e-3)
                        end
                    end
                end

                end
            end
        end
    end
    # ===================== end test/BB_specific.jl =====================
end