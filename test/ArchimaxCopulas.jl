# @testitem "Generic" tags=[:ArchimaxCopula, :BB4Copula] setup=[M] begin M.check(BB4Copula(0.50, 1.60)) end
# @testitem "Generic" tags=[:ArchimaxCopula, :BB4Copula] setup=[M] begin M.check(BB4Copula(2.50, 0.40)) end
# @testitem "Generic" tags=[:ArchimaxCopula, :BB4Copula] setup=[M] begin M.check(BB4Copula(3.0, 2.1)) end

# @testitem "Generic" tags=[:ArchimaxCopula, :BB5Copula] setup=[M] begin M.check(BB5Copula(1.50, 1.60)) end
# @testitem "Generic" tags=[:ArchimaxCopula, :BB5Copula] setup=[M] begin M.check(BB5Copula(2.50, 0.40)) end
# @testitem "Generic" tags=[:ArchimaxCopula, :BB5Copula] setup=[M] begin M.check(BB5Copula(5.0, 0.5)) end

# For each model below, we shoudl move them to generic testing
# @testitem "Generic" tags=[:ArchimaxCopula] setup=[M] begin M.check(ArchimaxCopula(2, ClaytonGenerator(1.2), AsymLogTail(1.2, [0.2, 0.5]))) end


@testitem "Archimax_specific" tags=[:ArchimaxCopula] begin
# ===================== test/Archimax_selected.jl =====================
    using Test
    using Random, StatsBase, HypothesisTests
    using Distributions, ForwardDiff, HCubature
    using StableRNGs
    using LogExpFunctions
    using Copulas

    _EPS = 1e-12
    # Qualified aliases to avoid UndefVarError
    AC   = Copulas.ArchimedeanCopula
    CG   = Copulas.ClaytonGenerator
    GG   = Copulas.GumbelGenerator
    FG   = Copulas.FrankGenerator
    JG   = Copulas.JoeGenerator
    BB1G = Copulas.BB1Generator
    BB6G = Copulas.BB6Generator
    Aevd = Copulas.A          # Pickands A(t) of EVC
    ϕ     = Copulas.ϕ
    ϕinv  = Copulas.ϕ⁻¹

    cdf_def(C::Copulas.ArchimaxCopula, u1::Real, u2::Real) = begin
        G, E = C.gen, C.tail
        (u1≤0 || u2≤0)  && return 0.0
        (u1≥1 && u2≥1)  && return 1.0
        x = ϕinv(G, u1); y = ϕinv(G, u2)
        S = x + y
        S == 0 && return 1.0
        t = y / S
        ϕ(G, S * Aevd(E, t))
    end

# Hessian cross as oracle pdf (at interior points)
    pdf_hess(C, u1, u2) = begin
        f(z) = cdf(C, z)                      
        H = ForwardDiff.hessian(f, [u1, u2])  # ∂²/∂u1∂u2
        max(H[1,2], 0.0)                      # numerical clip 
    end

    function rand_clamped(rng, C, n; ϵ=_EPS)
        U = rand(rng, C, n)
        @inbounds @views (U .= clamp.(U, ϵ, 1-ϵ))
        return U
    end

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

    function mc_pdf_integral(C; N::Int=1_000_000, seed::Integer=42)
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
        Ĩ  = exp(logS  - log(N))
        m2 = exp(logS2 - log(N))
        var̂ = max(m2 - Ĩ^2, 0.0)
        SE = sqrt(var̂ / N)
        return Ĩ, SE, N
    end
    
    ARCH_SPECS = Dict{Symbol,Any}(
        :Clayton => (θs=[1.5,3.0],  ctor = θ -> AC(2, CG(θ))),
        :Gumbel  => (θs=[2.0,4.0],  ctor = θ -> AC(2, GG(θ))),
        :Frank   => (θs=[0.8,6.0],  ctor = θ -> AC(2, FG(θ))),
        :Joe     => (θs=[1.2,2.5],  ctor = θ -> AC(2, JG(θ))),
        :BB1     => (pairs=[(1.3,1.4),(2.0,2.0)], ctor=(θ,δ)-> AC(2, BB1G(θ,δ))),
    )

    EVC_SPECS = Dict{Symbol,Any}(
        :Log        => (θs = [2.0, 1.5],       ctor = θ -> Copulas.LogTail(θ)),            # includeA≡1 (θ=1)
        :Galambos   => (θs = [0.7, 2.5],       ctor = θ -> Copulas.GalambosTail(θ)),
        :HS         => (θs = [0.6, 1.8],       ctor = θ -> Copulas.HuslerReissTail(θ)),             
        :AsymGal    => (triples = [(0.35, 0.65, 0.3)], ctor = (a,b,θ) -> Copulas.AsymGalambosTail(a, [b,θ])),
    )

    KS_ALPHA = 0.005
    ks_done = Dict{Symbol,Bool}(a => false for a in keys(ARCH_SPECS))

    @testset "Archimax: representative selection (AMH, Clayton, Gumbel, Frank, Joe, BB1, BB6) × (EVCs)" begin
        rng0 = StableRNG(2025)

        for (aname, ainfo) in ARCH_SPECS, (ename, einfo) in EVC_SPECS
            @testset "$(aname) × $(ename)" begin
                # iteradores de parámetros
                arch_iter = haskey(ainfo, :θs)    ? ((θ,) for θ in ainfo[:θs]) :
                            haskey(ainfo, :pairs)  ? ainfo[:pairs] :
                            error("ARCH_SPECS incorrectly defined for $aname")
                evc_iter  = haskey(einfo, :θs)    ? ((θ,) for θ in einfo[:θs]) :
                            haskey(einfo, :pairs)  ? einfo[:pairs] :
                            haskey(einfo, :triples) ? einfo[:triples] :
                            error("EVC_SPECS incorrectly defined for $ename")

                for apars in arch_iter, epars in evc_iter
                    A = (length(apars)==1) ? ainfo[:ctor](apars[1]) : ainfo[:ctor](apars...)
                    E = (length(epars)==1) ? einfo[:ctor](epars[1]) : einfo[:ctor](epars...)
                    C = ArchimaxCopula(A, E)

                    @test length(C) == 2
                    @test !isempty(params(C))

                    for (u1,u2) in ((0.2,0.3), (0.7,0.6), (0.9,0.4))
                        @test isapprox(cdf(C, [u1,u2]), cdf_def(C, u1, u2); rtol=1e-12, atol=1e-12)
                    end

                    for (u1,u2) in ((0.25,0.4), (0.6,0.6))
                        lp = logpdf(C, [u1,u2])
                        @test isfinite(lp)
                        c_h = pdf_hess(C, u1, u2)
                        @test isapprox(exp(lp), c_h; rtol=1e-6, atol=1e-8)
                    end

                    Ĩ, SE, _ = mc_pdf_integral(C; N=300_000, seed=123)
                    @test abs(Ĩ - 1) ≤ max(5SE, 1e-2)

                    for r in mc_rectangles_cdf(C; N=300_000, seed=321)
                        @test abs(r.p_hat - r.p_th) ≤ max(5*r.se, 2e-3)
                    end

                    U = rand(rng0, C, 20_000)
            
                    if !ks_done[aname]
                        @test pvalue(ExactOneSampleKSTest(view(U,1,:), Uniform())) > KS_ALPHA
                        @test pvalue(ExactOneSampleKSTest(view(U,2,:), Uniform())) > KS_ALPHA
                        ks_done[aname] = true
                    end

                    τ_em = corkendall(U[1,:], U[2,:])
                    τ_th = Copulas.τ(C)
                    @test isapprox(τ_th, τ_em; atol=1e-2)
                end
            end
        end
    end
end