# Family-regression layer: historical conditional-distribution and
# distortion regressions until component and family replacements are complete.

@testset "Bivariate scalar condition fast path" begin
    # Scalar/tuple equivalence is part of `obligations/contracts/copulas.jl`; retain only
    # inference, numeric-type propagation, and input-validation regressions.
    C = GaussianCopula{2}(0.4)
    @test @inferred(condition(C, 1, 0.4)) isa Copulas.GaussianDistortion
    for j in 1:2, uⱼ in (0.2f0, big"0.8")
        @test typeof(condition(C, j, uⱼ)) ==
              typeof(condition(C, (j,), (float(uⱼ),)))
    end

    @test_throws ArgumentError condition(C, 0, 0.4)
    @test_throws ArgumentError condition(C, 3, 0.4)
    @test_throws ArgumentError condition(C, 1, -0.1)
    @test_throws ArgumentError condition(C, 1, 1.1)
end

@testset "Plackett distortion closed-form quantile" begin
    for θ in (0.5, 2.0), j in 1:2
        C = PlackettCopula{2}(θ)
        uⱼ = j == 1 ? 0.3 : 0.7
        D = condition(C, (j,), (uⱼ,))
        @test D isa Copulas.PlackettDistortion
        @test isfinite(D.logden)

        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            @test isapprox(cdf(D, q), α; atol=5e-12, rtol=5e-12)
        end
        for u in (0.2, 0.6)
            reference = ForwardDiff.derivative(t -> cdf(D, t), u)
            @test logpdf(D, u) ≈ log(reference) atol = 2e-11
        end
        @test quantile(D, 0.0) == 0.0
        @test quantile(D, 1.0) == 1.0
        @test quantile(D, big"0.37") isa BigFloat
        @test logpdf(D, -0.1) == -Inf
        @test logpdf(D, 1.1) == -Inf
    end

    Dind = Copulas.PlackettDistortion(1.0, Int8(1), 0.4)
    @test quantile(Dind, 0.37) ≈ 0.37
end

@testset "Algebraic Archimedean distortion quantiles" begin
    copulas = (
        FrankCopula{2}(-2.0),
        FrankCopula{2}(3.0),
        AMHCopula{2}(-0.5),
        AMHCopula{2}(0.5),
    )
    for C in copulas
        D = condition(C, (1,), (0.4,))
        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            generic = @invoke quantile(D::Copulas.Distortion, α::Real)
            @test isapprox(cdf(D, q), α; atol=2e-11, rtol=2e-11)
            @test isapprox(q, generic; atol=2e-8, rtol=2e-8)
        end
        @test quantile(D, big"0.37") isa BigFloat
    end
end

@testset "Gumbel and Log distortion closed-form quantiles" begin
    for θ in (1.001, 1.2, 2.5, 8.0), uⱼ in (0.25, 0.7)
        Dg = condition(GumbelCopula{2}(θ), (1,), (uⱼ,))
        Dl = condition(LogCopula{2}(θ), (1,), (uⱼ,))
        for α in (0.1, 0.5, 0.9)
            qg = quantile(Dg, α)
            ql = quantile(Dl, α)
            generic = @invoke quantile(Dg::Copulas.Distortion, α::Real)
            @test isapprox(cdf(Dg, qg), α; atol=2e-11, rtol=2e-11)
            @test isapprox(cdf(Dl, ql), α; atol=2e-11, rtol=2e-11)
            @test isapprox(qg, ql; atol=2e-11, rtol=2e-11)
            @test isapprox(qg, generic; atol=2e-8, rtol=2e-8)
        end
    end
end

@testset "Lambert-W Archimedean distortion quantiles" begin
    copulas = (
        InvGaussianCopula{2}(0.01),
        InvGaussianCopula{2}(0.5),
        InvGaussianCopula{2}(2.0),
        BB9Copula{2}(1.0, 0.8),
        BB9Copula{2}(1.001, 0.8),
        BB9Copula{2}(2.5, 0.8),
    )
    for C in copulas
        D = condition(C, (1,), (0.4,))
        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            generic = @invoke quantile(D::Copulas.Distortion, α::Real)
            @test isapprox(cdf(D, q), α; atol=3e-11, rtol=3e-11)
            @test isapprox(q, generic; atol=2e-8, rtol=2e-8)
        end
        @test quantile(D, big"0.37") isa BigFloat
    end
end

@testset "Gumbel-Barnett distortion closed-form quantile" begin
    for θ in (0.01, 0.2, 0.8), uⱼ in (0.3, 0.7)
        D = condition(GumbelBarnettCopula{2}(θ), (1,), (uⱼ,))
        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            generic = @invoke quantile(D::Copulas.Distortion, α::Real)
            @test isapprox(cdf(D, q), α; atol=3e-11, rtol=3e-11)
            @test isapprox(q, generic; atol=2e-8, rtol=2e-8)
        end
        @test quantile(D, big"0.37") isa BigFloat
    end
end

@testset "Gaussian distortion log-scale formulas" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))
    N = Normal()
    for u in (1e-12, 0.2, 0.5, 0.8)
        q = quantile(N, u)
        z = (q - D.μz) / D.σz
        reference = logpdf(N, z) - log(abs(D.σz)) - logpdf(N, q)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 1e-13
        @test logpdf(D, u) ≈ reference atol = 1e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
    @test logpdf(D, -0.1) == -Inf
end

@testset "Student distortion logcdf" begin
    D = condition(TCopula{2}(4, [1.0 0.5; 0.5 1.0]), (1,), (0.3,))
    @test D.Tu isa TDist
    @test D.Tcond isa TDist
    for u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "Elliptical conditioning shares matrix factorizations" begin
    Σ = [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]
    for C in (GaussianCopula{3}(Σ), TCopula{3}(4, Σ))
        conditioned = condition(C, (1,), (0.35,))
        @test length(conditioned.m) == 2
        for (k, i) in enumerate((2, 3)), u in (0.2, 0.7)
            reference = Copulas.DistortionFromCop(C, (1,), (0.35,), i)
            @test cdf(conditioned.m[k], u) ≈ cdf(reference, u) atol = 2e-12
        end
    end
end

@testset "Distorted distribution logcdf" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))(Logistic())
    @test D isa Copulas.DistortedDist
    for x in (-8.0, -0.5, 1.0)
        @test logcdf(D, x) ≈ logcdf(D.D, cdf(D.X, x)) atol = 2e-13
    end
end

@testset "Extreme-value conditioning caches fixed transforms" begin
    DEV = condition(GalambosCopula{2}(2.5), (1,), (0.3,))
    @test DEV.negloguⱼ == -log(DEV.uⱼ)

    DAM = condition(ArchimaxCopula{2}(Copulas.FrankGenerator(0.8),
                                  Copulas.HuslerReissTail(0.6)), (1,), (0.3,))
    @test DAM.yⱼ == Copulas.ϕ⁻¹(DAM.gen, DAM.uⱼ)
    @test DAM.invderivⱼ == Copulas.ϕ⁻¹⁽¹⁾(DAM.gen, DAM.uⱼ)
end

@testset "Archimedean distortion logcdf" begin
    distortions = (
        condition(ClaytonCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(FrankCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(GumbelCopula{3}(2.0), (1, 2), (0.3, 0.6)),
    )
    for D in distortions, u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 3e-12
    end
    @test all(logcdf(D, 0.0) == -Inf for D in distortions)
    @test all(logcdf(D, 1.0) == 0.0 for D in distortions)
end

@testset "Flip distortion logcdf" begin
    S = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (2,))
    D = condition(S, (1,), (0.3,))
    @test D isa Copulas.FlipDistortion
    for u in (0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-12
    end
    u = 1e-12
    @test logcdf(D, u) ≈ LogExpFunctions.log1mexp(logcdf(D.base, 1 - u)) atol = 2e-12
    @test isfinite(logcdf(D, u))
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "FGM distortion log-scale formulas" begin
    for θ in (-0.8, 0.8), uⱼ in (0.2, 0.7)
        D = condition(FGMCopula{2}(θ), (1,), (uⱼ,))
        for u in (1e-12, 0.2, 0.5, 0.8)
            @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-14
        end
        @test logcdf(D, 0.0) == -Inf
        @test logcdf(D, 1.0) == 0.0
        @test logpdf(D, -0.1) == -Inf
        @test logpdf(D, 1.1) == -Inf
    end
end

@testset "Generic ConditionalCopula density" begin
    C = GaussianCopula{3}([
        1.0 0.35 0.20
        0.35 1.0 0.25
        0.20 0.25 1.0
    ])
    js = (3,)
    ujs = (0.4,)
    generic = @invoke Copulas.ConditionalCopula(C::Copulas.Copula{3}, js, ujs)
    Cgeneric = FGMCopula{3}([0.1, 0.2, 0.3, 0.4])
    conditioned = condition(Cgeneric, js, ujs)
    @test conditioned.C isa Copulas.ConditionalCopula
    @test conditioned.m === conditioned.C.distortions
    @test conditioned.C.is == (1, 2)
    @test generic.logden == log(generic.den)
    specialized = Copulas.ConditionalCopula(C, js, ujs)

    for u in ([0.25, 0.35], [0.5, 0.5], [0.75, 0.65])
        @test isapprox(logpdf(generic, u), logpdf(specialized, u); atol=1e-8, rtol=1e-8)
        @test isapprox(pdf(generic, u), pdf(specialized, u); atol=1e-8, rtol=1e-8)
    end
    @test pdf(generic, [-0.1, 0.5]) == 0

    Cclayton = ClaytonCopula{3}(2.0)
    generic_big = @invoke Copulas.ConditionalCopula(
        Cclayton::Copulas.Copula{3},
        (3,),
        (big"0.4",),
    )
    value_big = logpdf(generic_big, BigFloat[0.35, 0.65])
    @test value_big isa BigFloat
    @test isfinite(value_big)
end

@testset "Checkerboard distortion supports multiple conditioning dimensions" begin
    C = CheckerboardCopula{3}(randn(rng, 3, 30); pseudo_values=false)
    D = Copulas.DistortionFromCop(C, (1, 2), (0.3, 0.7), 3)

    @test D isa Copulas.HistogramBinDistortion
    @test all(0 .<= cdf.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
    @test all(pdf.(Ref(D), (0.2, 0.5, 0.8)) .>= 0)
    @test all(0 .<= quantile.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
end

@testset "Bernstein distortion quantiles use bounded bisection" begin
    C = BernsteinCopula{2}(GaussianCopula{2}(0.3); m=5)
    D = condition(C, (1,), (0.4,))
    @test D isa Copulas.BernsteinDistortion
    for p in (0.1, 0.5, 0.9)
        q = quantile(D, p)
        @test 0 <= q <= 1
        @test cdf(D, q) ≈ p atol = 2e-12
    end
end

@testset "Bivariate Archimedean conditional (generator formula across families)" begin
    # [GenericTests integration]: Yes. We already added a similar Archimedean conditional check using generator identities in GenericTests.
    # Known bivariate Archimedean identity:
    # H(u | v) = ϕ'(ϕ^{-1}(u) + ϕ^{-1}(v)) / ϕ'(ϕ^{-1}(v))
    # Test it across multiple families by looping instead of duplicating code.
    examples = (
        ClaytonCopula{2}(1.2),
        FrankCopula{2}(1.0),
        GumbelCopula{2}(1.2),
    )
    J = (2,)
    tol = 5e-5
    for C in examples
        for v in (0.2, 0.5, 0.8)
            D = condition(C, J, (v,))
            inv_v = Copulas.ϕ⁻¹(C.G, v)
            for u in (1e-6, 0.1, 0.4, 0.8, 1 - 1e-6)
                t = Copulas.ϕ⁻¹(C.G, u) + inv_v
                num = Copulas.ϕ⁽¹⁾(C.G, t)
                den = Copulas.ϕ⁽¹⁾(C.G, inv_v)
                expected = num / den
                @test isfinite(expected) && 0.0 <= expected <= 1.0
                @test isapprox(cdf(D, u), expected; atol=tol, rtol=tol)
            end
        end
    end
end

@testset "GaussianCopula conditional copula vs MVN" begin
    # [GenericTests integration]: Maybe. It depends on MvNormalCDF and is moderately heavy; could be a behind-flag exhaustive check.
    Random.seed!(rng,42)
    d = 4
    # build correlation matrix
    A = randn(rng, d, d)
    Σ = A*A'
    # normalize to correlation
    s = sqrt.(diag(Σ))
    Σ = Symmetric(Σ ./ (s*s'))
    C = GaussianCopula{4}(Matrix(Σ))
    # Choose J and uJ
    J = (2,4)
    uJ = (0.3, 0.8)
    CC = condition(C, J, uJ)
    # Compare to MVNormal conditioning on z-scale
    I = Tuple(setdiff(1:d, J))
    dI = length(I)
    Iv = collect(I); Jv = collect(J)
    ΣII = Σ[Iv, Iv]; ΣJJ = Σ[Jv, Jv]; ΣIJ = Σ[Iv, Jv]; ΣJI = Σ[Jv, Iv]
    L = cholesky(ΣJJ)
    zJ = quantile.(Normal(), collect(uJ))
    y = L \ zJ
    μ = ΣIJ * (L' \ y)
    K = L \ ΣJI
    Σcond = ΣII - ΣIJ * (L'\K)
    for _ in 1:3
        uI = rand(rng, dI)./5 .+ 2/5
        zI = quantile.(Normal(), uI)
        p_mvn = MvNormalCDF.mvnormcdf(vec(μ), Matrix(Σcond), fill(-Inf, dI), zI)[1]
        p_cc = cdf(CC, uI)
        @test isapprox(p_cc, p_mvn; atol=5e-3)
    end
end

@testset "Higher-dim Archimedean conditional (3|2 via generator derivatives)" begin
    # [GenericTests integration]: Yes. This extends the Archimedean conditional identity to higher p; can be parameterized and integrated.
    # For Archimedean C(u) = ϕ(Σ ϕ⁻¹(u_i)), conditioning on J with |J|=p gives
    # H_{I|J}(u_I|u_J) = ϕ^{(p)}(Σ_{i∈I} ϕ⁻¹(u_i) + Σ_{j∈J} ϕ⁻¹(u_j)) / ϕ^{(p)}(Σ_{j∈J} ϕ⁻¹(u_j))
    # We'll test in d=5 with |J|=2, so |I|=3.
    families = [
        (ClaytonCopula, 1.1, 1e-5),
        (FrankCopula,   2.0, 1e-5),
        # (GumbelCopula,  1.5, 5e-5),
    ]
    d = 5
    J = (2, 4)
    p = length(J)
    for (Ctor, θ, tol) in families
        C = Ctor(d, θ)
        # a couple of moderate conditioning points away from 0/1 to avoid singularities
        for uJ in ((0.2, 0.7), (0.3, 0.8))
            CC = condition(C, J, uJ)
            # test a few uI points
            for uI in ((0.1, 0.4, 0.8), (0.25, 0.5, 0.75), (0.2, 0.6, 0.9))
                # Compute expected via generator-derivative ratio
                SJ = sum(Copulas.ϕ⁻¹(C.G, v) for v in uJ)
                SI = sum(Copulas.ϕ⁻¹(C.G, u) for u in uI)
                S_full = SJ + SI
                num = Copulas.ϕ⁽ᵏ⁾(C.G, p, S_full)
                den = Copulas.ϕ⁽ᵏ⁾(C.G, p, SJ)
                expected = num / den
                # Evaluate model
                got = cdf(CC, collect(uI))
                @test isfinite(expected) && 0.0 <= expected <= 1.0
                @test isapprox(got, expected; atol=tol, rtol=tol)
            end
        end
    end
end

@testset "Gaussian Sklar conditional vs MVN with normal marginals" begin
# [GenericTests integration]: Yes. This validates SklarDist conditioning against MVN algebra; belongs in GenericTests under conditioning.
    Random.seed!(rng,43)
    d = 3
    Σ = [1 0.7 0.3;0.7 1 0.7; 0.3 0.7 1]
    C = GaussianCopula{3}(Σ)
    μ = zeros(d)
    
    X = SklarDist(C, Tuple(Normal(μ[i],Σ[i,i]) for i in 1:d))
    X_mock = MvNormal(μ, Σ)

    # check that X and X_mock are indeed the same distribution: 
    for _ in 1:5
        t = rand(rng, 3)
        A, r = mvnormcdf(X_mock, fill(-Inf, d), t)
        B = cdf(X, t)
        @test A ≈ B atol=10sqrt(r)
    end

    
    # Now condition using the known gaussian conditionning algebra: 
    xⱼₛ = [0]
    is, js = 2:3, 1:1 
    μ_Y    = μ[is] .+ Σ[is, js] * inv(Σ[js, js]) * (xⱼₛ - μ[js])
    Σ_Y = Σ[is,is] .- Σ[is,js] * inv(Σ[js,js]) * Σ[js, is]
    Y_mock = MvNormal(μ_Y, Σ_Y)

    # And construct the conditioning using the generic paths: 
    J = Tuple(reverse(collect(js)))
    Y = condition(X, J, xⱼₛ)

    for _ in 1:3
        t = randn(rng, 2)
        A, r = mvnormcdf(Y_mock, fill(-Inf, 2), t)
        B = cdf(Y, t)
        @test A ≈ B atol=10sqrt(r)
    end
end
